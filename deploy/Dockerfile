# Bot image (Discord / Instagram). Inference runs in the vLLM containers, so this needs no GPU.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra rag --extra discord --extra instagram

COPY configs ./configs
ENTRYPOINT ["uv", "run", "--no-sync", "charliegpt", "-c", "configs/docker.yaml"]
