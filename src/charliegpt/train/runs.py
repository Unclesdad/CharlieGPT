"""Each training run gets its own folder named for when it started, e.g. models/charlie-2026-11-15_14-32-05.

Serving picks the newest finished run unless you name another one (`serve --model`, or `model.adapter` in the config).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

STAMP = "%Y-%m-%d_%H-%M-%S"
# Written after the final save. Adapter files alone don't prove a run finished: a cancelled one can have them too.
DONE = "TRAINED"


def _tag(name: str, backend: str) -> str:
    return f"{name}-mlx" if backend == "mlx" else name


def new_run_dir(models: Path, name: str, backend: str, now: datetime | None = None) -> Path:
    return Path(models) / f"{_tag(name, backend)}-{(now or datetime.now()).strftime(STAMP)}"


def list_runs(models: Path, name: str, backend: str) -> list[Path]:
    """Finished runs, oldest first."""
    pattern = re.compile(rf"{re.escape(_tag(name, backend))}-\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d")
    runs = [p for p in Path(models).glob("*") if pattern.fullmatch(p.name) and (p / DONE).exists()]
    return sorted(runs, key=lambda p: p.name)


def find_adapter(models: Path, name: str, backend: str, choice: str | None = "latest") -> Path | None:
    """`choice` is "latest", a run name like charlie-2026-11-15_14-32-05, or a path."""
    if choice and choice != "latest":
        for p in (Path(models) / choice, Path(choice)):
            if p.exists():
                return p
        raise FileNotFoundError(f"no model {choice!r} in {models} (see `charliegpt models`)")
    runs = list_runs(models, name, backend)
    # No fallback to an untimestamped models/charlie: it predates the current prompt format and acts like a plain assistant.
    return runs[-1] if runs else None


def mark_done(run: Path) -> None:
    (run / DONE).write_text(datetime.now().isoformat())


def link_latest(run: Path) -> Path:
    """models/<name>-latest → the run, for Docker Compose (which can't scan folders)."""
    link = run.parent / (re.sub(r"-\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d$", "", run.name) + "-latest")
    link.unlink(missing_ok=True)
    link.symlink_to(run.name)  # relative, so it still resolves inside a container mount
    return link
