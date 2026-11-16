"""Image captions, so the model "sees" photos as text: [image: "a dog wearing sunglasses"].

Used offline over exports and live by the bots. Captions are cached by file hash.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import threading
from pathlib import Path
from typing import Iterable, Protocol

from charliegpt.config import Settings, VisionCfg
from charliegpt.schema import AttachmentKind, Conversation

PROMPT = (
    "Describe this image in one short sentence, the way you'd tell a friend what was sent. "
    "If it's a meme or screenshot, include the important text."
)


class Captioner(Protocol):
    def caption(self, path: Path) -> str: ...


def _to_jpeg_bytes(path: Path) -> tuple[bytes, str]:
    """HEIC (iPhone) and other formats -> JPEG so any backend can read them."""
    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime in ("image/jpeg", "image/png", "image/webp"):
        return path.read_bytes(), mime
    from io import BytesIO

    from PIL import Image

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


class OpenAICaptioner:
    """Any OpenAI-compatible vision endpoint (the vLLM vision server in docker-compose)."""

    def __init__(self, cfg: VisionCfg, api_key: str = "EMPTY"):
        from openai import OpenAI

        self.client = OpenAI(base_url=cfg.base_url, api_key=api_key, timeout=60)
        self.model = cfg.model

    def caption(self, path: Path) -> str:
        data, mime = _to_jpeg_bytes(path)
        url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT}]}],
            max_tokens=60,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()


class LocalCaptioner:
    """In-process model: transformers on CUDA, mlx-vlm on Apple Silicon."""

    def __init__(self, cfg: VisionCfg):
        from charliegpt.config import detect_device

        self.device = detect_device()
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        if self.device == "mps":
            from mlx_vlm import load

            self._model, self._processor = load(self.cfg.local_model_mlx)
        else:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            name = self.cfg.local_model_cuda
            self._processor = AutoProcessor.from_pretrained(name)
            self._model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")

    def caption(self, path: Path) -> str:
        with self._lock:
            self._load()
            data, _ = _to_jpeg_bytes(path)
            from io import BytesIO

            from PIL import Image

            image = Image.open(BytesIO(data)).convert("RGB")
            if self.device == "mps":
                from mlx_vlm import generate
                from mlx_vlm.prompt_utils import apply_chat_template

                prompt = apply_chat_template(self._processor, self._model.config, PROMPT, num_images=1)
                out = generate(self._model, self._processor, prompt, image=[image], max_tokens=60, verbose=False)
                return str(getattr(out, "text", out)).strip()
            msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
            inputs = self._processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self._model.device)
            ids = self._model.generate(**inputs, max_new_tokens=60, do_sample=False)
            return self._processor.batch_decode(ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()


class CachedCaptioner:
    def __init__(self, inner: Captioner, cache_path: Path):
        self.inner = inner
        self.cache_path = cache_path
        self.cache: dict[str, str] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._lock = threading.Lock()

    def caption(self, path: Path) -> str:
        key = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        text = self.inner.caption(Path(path))
        with self._lock:
            self.cache[key] = text
            self.save()
        return text

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False))


def make_captioner(settings: Settings) -> Captioner | None:
    v = settings.vision
    if v.backend == "none":
        return None
    inner: Captioner = OpenAICaptioner(v, settings.inference.api_key) if v.backend == "openai" else LocalCaptioner(v)
    return CachedCaptioner(inner, settings.paths.captions)


def resolve(path: str, path_map: dict[str, str] | None = None) -> Path:
    """Where an attachment lives on this machine (features keep the original path)."""
    for src, dst in (path_map or {}).items():
        if path.startswith(src):
            return Path(dst + path[len(src):])
    return Path(path).expanduser()


def caption_threads(convs: Iterable[Conversation], captioner: Captioner, on_error=None, path_map: dict[str, str] | None = None) -> int:
    n = 0
    for c in convs:
        for m in c.messages:
            for a in m.attachments:
                if a.kind != AttachmentKind.image or a.caption or not a.path:
                    continue
                local = resolve(a.path, path_map)
                if not local.exists():
                    continue
                try:
                    a.caption = captioner.caption(local)
                    n += 1
                except Exception as e:  # one bad file shouldn't stop a 10k-image run
                    if on_error:
                        on_error(a.path, e)
    return n
