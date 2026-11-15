# CharlieGPT v2

Feed it your iMessage / Discord / Instagram history, fine-tune an LLM to text like you, and run bots that reply as you.

```
data/<source> ──ingest──▶ features/ ──caption/index/profiles──▶ dataset ──train──▶ LoRA ──serve──▶ bots
 messages/                 uniform JSONL                       data/dataset      models/      discord / imessage / instagram
 discord/
 instagram/
```

**Model:** Qwen3-14B (thinking off) with QLoRA. It's trained with Unsloth on the RTX 3090 and served by vLLM (AWQ base + LoRA), producing a full multi-bubble reply in about 5–10s. On Apple Silicon (18GB) it trains and serves Qwen3-8B (4-bit) with MLX. Models are swappable in `configs/default.yaml`.

## Models (downloaded automatically)

Nothing to download by hand. Each model is pulled from Hugging Face the first time it's used and cached in `~/.cache/huggingface` (the Docker containers mount that same cache). No account or token is needed.

| Model | Used for | Where | Size |
|---|---|---|---|
| `unsloth/Qwen3-14B-unsloth-bnb-4bit` | training (QLoRA) | 3090 | ~10 GB |
| `Qwen/Qwen3-14B-AWQ` | serving the persona + helper model | 3090 | ~10 GB |
| `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | image captions (vision server) | 3090 | ~3 GB |
| `BAAI/bge-small-en-v1.5` | RAG embeddings | both | ~130 MB |
| `mlx-community/Qwen3-8B-4bit` | training/serving on Mac | Mac | ~4.6 GB |
| `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | captions on Mac, only if `vision.backend: local` | Mac | ~2 GB |
| `mlx-community/Qwen3-0.6B-4bit` | smoke test | Mac | ~350 MB |

To fetch ahead of time (e.g. before going offline): `uvx hf download <model>`. Your fine-tune itself (the LoRA adapter) is created locally in `models/` by `charliegpt train` and never leaves your machines.

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

Edit `configs/default.yaml` → `identity`: your name, aliases, Discord user id, Instagram display name, and optional `contacts` overrides (phone/email → name; the `.vcf` in `data/messages` is picked up automatically, or set `contacts_vcf`).

## 1. Get your data

Each source gets its own folder under `data/` (gitignored). Parsers only read their own folder and write the uniform format to `features/<source>.jsonl` (also gitignored).

```
data/
  messages/    imessage-exporter HTML (one file per chat) + your contacts .vcf
  discord/     DiscordChatExporter JSON
  instagram/   unzipped Instagram export
features/      uniform-format output
```

| Folder | How to export |
|---|---|
| `data/messages` | `brew install imessage-exporter && imessage-exporter -f html -o data/messages`, plus Contacts.app → select all → File → Export vCard into the same folder |
| `data/discord` | [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter): `--format Json --media` for the DMs/channels you want |
| `data/instagram` | Accounts Center → Download your information → **JSON**, messages. Copy the thread folders from `your_instagram_activity/messages/inbox/` straight into `data/instagram/` (keeping the whole export also works) |

```bash
charliegpt ingest all          # every non-empty folder
charliegpt ingest imessage     # or one at a time: imessage | discord | instagram
charliegpt stats               # messages, your share, and trainable replies per platform
```

### What each parser handles

**iMessage** (`ingest/imessage_html.py`). This parses imessage-exporter's HTML, one file per chat:
- **Chat type:** a file named after one handle (`+15551234567.html`) is a DM. Files named after several handles (`+1555…, +1555….html`) or with a group name (`Trip Crew - 123.html`) are group chats.
- **Kept:** text (final version if edited), photos/videos/voice memos (with transcriptions) pointing at the originals in `~/Library/Messages/Attachments`, link previews, stickers, tapbacks (`Loved` → ❤️, custom emoji reactions), and threaded replies (linked to the message they answer).
- **Skipped:** unsent messages, polls, and group announcements ("X named the conversation").
- **Duplicates:** the exporter prints threaded replies twice (nested + timeline); they're merged.
- **Contacts:** phone numbers and emails are resolved through the `.vcf` (`ingest/vcard.py`). It handles folded lines, `itemN.` groups, escapes, and contacts with only N/ORG/NICKNAME. Numbers are normalized to E.164 (`(201) 555-0100` → `+12015550100`). Anyone missing can be added under `identity.contacts`, which overrides the vCard.
- **Alternative:** `charliegpt ingest imessage --from-db` reads `~/Library/Messages/chat.db` directly (needs Full Disk Access).

**Instagram** (`ingest/instagram.py`). This parses Meta's JSON: each thread folder has `message_1.json` (+ `message_2.json`… for long chats) and `photos/ videos/ audio/`:
- **Discovery:** threads are found anywhere under `data/instagram`. Media URIs still carry the original `your_instagram_activity/…` prefix and get re-anchored to the thread folder; media files have no extensions, which is fine.
- **Emoji:** Meta's broken encoding (`ð\u009f\u0098\u0082` → 😂) is fixed.
- **Shared posts/reels:** these keep the author and caption: `[shared https://instagram.com/p/…: "@account: caption…"]`.
- **Dropped notices:** "liked a message", "Reacted 😂 to your message", "sent an attachment" (when the attachment itself is kept), joins/leaves/adds, renames, theme/nickname/photo changes, calls, polls, and pins.
- **Your name:** set `identity.instagram_name` to your display name exactly as it appears in `participants`.

**Discord** (`ingest/discord.py`). This parses DiscordChatExporter JSON. Set `identity.discord_user_ids` so your messages are recognized. Discord's official data package only contains your side of conversations, so it can't teach context and isn't supported.

### The uniform format (`features/<source>.jsonl`)

One JSON object per chat thread (`schema.Conversation`):

```jsonc
{
  "platform": "instagram", "thread_id": "alex_123", "title": "Alex", "is_group": false,
  "participants": [{"id": "me", "name": "Charlie", "is_me": true}, {"id": "alex", "name": "Alex"}],
  "messages": [{
    "id": "…", "sender_id": "me", "sender_name": "Charlie", "is_me": true,
    "timestamp": "2026-09-20T14:13:40+00:00", "text": "lfg",
    "attachments": [{"kind": "image", "path": "data/instagram/…/photos/123", "caption": null}],
    "reactions": [{"emoji": "😂", "sender_id": "alex"}],
    "reply_to": null
  }]
}
```

The model sees each message as a single line: text plus placeholders like `[image: "caption"]`, `[voice memo: "transcript"]`, `[shared url: "…"]`, `[sticker]` and `(replying to "…")`. At load time, threads are split into sessions (after `dataset.session_gap_hours` of silence) and back-to-back messages from one person are grouped into a turn of several texts.

## 2. Enrich + build the dataset

```bash
charliegpt caption         # image → [image: "caption"] (vision server or vision.backend: local); cached, re-runnable
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
