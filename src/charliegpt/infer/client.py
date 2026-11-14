"""OpenAI-compatible client for vLLM (3090) or mlx_lm.server (Mac), with a fallback server."""

from __future__ import annotations

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from charliegpt.config import InferenceCfg


class LLMClient:
    def __init__(self, cfg: InferenceCfg):
        self.cfg = cfg
        self.servers: list[tuple[AsyncOpenAI, str | None]] = [
            (AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_seconds, max_retries=1), None)
        ]
        if cfg.fallback_url:
            fb = AsyncOpenAI(base_url=cfg.fallback_url, api_key=cfg.api_key, timeout=cfg.timeout_seconds, max_retries=0)
            self.servers.append((fb, cfg.fallback_model))

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        err: Exception | None = None
        for client, model_override in self.servers:
            try:
                resp = await client.chat.completions.create(
                    model=model_override or model or self.cfg.persona_model,
                    messages=messages,
                    temperature=self.cfg.temperature if temperature is None else temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=max_tokens or self.cfg.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return resp.choices[0].message.content or ""
            except (APIConnectionError, APITimeoutError) as e:
                err = e
        assert err is not None
        raise err

    async def persona(self, messages: list[dict]) -> str:
        return await self.chat(messages, model=self.cfg.persona_model)

    async def helper(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 300) -> str:
        return await self.chat(messages, model=self.cfg.helper_model, temperature=temperature, max_tokens=max_tokens)
