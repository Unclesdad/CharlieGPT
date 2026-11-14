# CharlieGPT v2

Feed it your iMessage / Discord / Instagram history, fine-tune an LLM to text like you, and run bots that reply as you.

```
raw exports ──ingest──▶ uniform format ──caption/index/profiles──▶ dataset ──train──▶ LoRA ──serve──▶ bots
 chat.db                data/processed/*.jsonl                   data/dataset      models/       discord / imessage / instagram
 DCE json
 IG json
```

**Model:** Qwen3-14B (thinking off) with QLoRA. It's trained with Unsloth on the RTX 3090 and served by vLLM (AWQ base + LoRA), producing a full multi-bubble reply in about 5–10s. On Apple Silicon (18GB) it trains and serves Qwen3-8B (4-bit) with MLX. Models are swappable in `configs/default.yaml`.

## Layout

| Path | What |
|---|---|
| `src/charliegpt/schema.py` | The uniform format: `Conversation → Message (+Attachment, Reaction)`, `Turn` = a burst of bubbles |
| `src/charliegpt/ingest/` | Parsers per platform, normalization (dedupe, sessions, bursts, replies), image captioning |
| `src/charliegpt/prompt.py` | **The** prompt format, shared by training and inference (a test enforces parity) |
| `src/charliegpt/memory/` | Per-chat profiles + RAG over past "they said → you said" exchanges |
| `src/charliegpt/dataset/` | Sessions → chat-format JSONL |
| `src/charliegpt/train/` | `cuda.py` (Unsloth QLoRA), `mlx.py` (mlx-lm LoRA) |
| `src/charliegpt/infer/` | OpenAI-compatible client, reply engine, guardrails |
| `src/charliegpt/bots/` | Shared bot logic + Discord / iMessage / Instagram adapters |
| `deploy/` | Docker Compose (Ubuntu), launchd (Mac iMessage) |

## Setup

```bash
cp .env.example .env            # DISCORD_TOKEN etc.
# Mac
uv sync --extra mlx --extra rag --extra imessage --extra dev
# Ubuntu / 3090
uv sync --extra cuda --extra rag --extra discord --extra vision
```

Edit `configs/default.yaml` → `identity`: your name, aliases, Discord user id, Instagram display name, and an optional `contacts` map of phone numbers to names.

## 1. Get your data

| Platform | How to export | Command |
|---|---|---|
| iMessage | Nothing to export. Give your terminal **Full Disk Access** (System Settings → Privacy & Security) | `charliegpt ingest imessage` |
| Discord | [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter): `--format Json --media` for the DMs/channels you want | `charliegpt ingest discord path/to/exports` |
| Instagram | Accounts Center → Download your information → **JSON**, messages | `charliegpt ingest instagram path/to/unzipped` |

Discord's official data package only contains your side of conversations, so it can't teach context and isn't supported.

Then run `charliegpt stats` to check message counts and how many trainable replies you have.

## 2. Enrich + build the dataset

```bash
charliegpt caption         # image → [image: "caption"] (vision server or vision.backend: local)
charliegpt index           # RAG index of past exchanges
charliegpt serve &         # base model up (profiles need the helper model)
charliegpt profiles        # per-chat summaries → data/memory/profiles.json (edit `notes` freely)
charliegpt build-dataset   # → data/dataset/{train,valid}.jsonl
```

## 3. Train

```bash
charliegpt train                 # auto: Unsloth on CUDA, mlx-lm on Mac
charliegpt train --merge         # (CUDA) also save a merged 16-bit model
```

To run the 3090-trained 14B on the Mac: `train --merge`, copy `models/charlie-merged` over, then `mlx_lm convert --hf-path models/charlie-merged -q`. The easier route is to point the Mac at the Ubuntu server.

## 4. Serve + try it

```bash
charliegpt serve           # vLLM on :8000 (models "charlie" and "qwen3-14b"), or mlx_lm server on Mac
charliegpt chat            # text it yourself
charliegpt bench           # p50/p95 latency; target < 20s
```

## 5. Bots

```bash
charliegpt bot discord --dry-run     # prints replies instead of sending; always start here
charliegpt bot imessage              # macOS only
charliegpt pause / resume / status   # kill switch, checked before every send
```

Behavior (all in `bots:` config):
- replies to every DM; in group chats only when @mentioned or named (`identity.aliases`)
- waits `debounce_seconds` for the other person to finish, then sends bubbles with typing delays
- **human takeover:** if you text in a chat yourself, the bot goes quiet there for `takeover_minutes`
- stays in character; replies that leak "as an AI" are regenerated or dropped
- **commitments** (plans, money, favors) get an absurd excuse instead of a yes

### Run in the background

**Ubuntu (3090):** Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```bash
docker compose -f deploy/docker-compose.yml up -d                  # inference + vision + discord bot
docker compose -f deploy/docker-compose.yml --profile demo up -d   # + instagram (demo only)
```
`charliegpt pause` on the host stops the containerized bots too, since `state/` is mounted.

**Mac (iMessage):** set `inference.base_url` and `vision.base_url` to the Ubuntu box (LAN or Tailscale IP, e.g. `http://100.x.y.z:8000/v1` and `:8001/v1`), then:
```bash
./deploy/launchd/install.sh             # starts at login, restarts on crash
./deploy/launchd/install.sh uninstall
```
Grant Full Disk Access to the Python binary the script prints. Allow Messages automation when macOS asks.

### Instagram

`instagram_bot.py` uses the private API (instagrapi). Automating a personal account **violates Instagram's Terms of Use** and can get it banned. It exists as a demo and won't start unless `instagram.i_understand_this_violates_tos: true`.

## Smoke test (Mac, about 2 minutes)

```bash
uv run pytest
charliegpt -c configs/smoke.yaml build-dataset && charliegpt -c configs/smoke.yaml train   # Qwen3-0.6B, 50 iters
charliegpt -c configs/smoke.yaml serve & charliegpt -c configs/smoke.yaml bench -n 6
```
