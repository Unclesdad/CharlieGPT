"""LoRA fine-tune on Apple Silicon with mlx-lm, using the same JSONL as the CUDA path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from charliegpt.config import Settings


def mlx_train_command(settings: Settings, data_dir: Path, out_dir: Path) -> list[str]:
    m, t = settings.model, settings.train
    return [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", m.mlx_base,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(out_dir / f"{m.adapter_name}-mlx"),
        "--iters", str(t.mlx_iters),
        "--batch-size", str(t.mlx_batch_size),
        "--num-layers", str(t.mlx_num_layers),
        "--learning-rate", str(t.learning_rate),
        "--max-seq-length", str(m.max_seq_length),
        "--mask-prompt",  # loss on your reply only
        "--grad-checkpoint",
        "--steps-per-eval", "100",
        "--save-every", "200",
    ]  # fmt: skip


def train_mlx(settings: Settings, data_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(mlx_train_command(settings, data_dir, out_dir), check=True)
    return out_dir / f"{settings.model.adapter_name}-mlx"
