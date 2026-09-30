"""LLM abstraction layer — pluggable Anthropic / OpenAI backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from persona.config import settings


@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    model: str


class LLMClient:
    """Thin wrapper that dispatches to the configured LLM provider."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if self.provider == "anthropic":
            return await self._anthropic(system_prompt, user_message, max_tokens, temperature)
        if self.provider == "openai":
            return await self._openai(system_prompt, user_message, max_tokens, temperature)
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    async def _anthropic(
        self, system_prompt: str, user_message: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        resp = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text = resp.content[0].text if resp.content else ""
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        return LLMResponse(text=text, tokens_used=tokens, model=resp.model)

    async def _openai(
        self, system_prompt: str, user_message: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        import openai

        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return LLMResponse(text=text, tokens_used=tokens, model=resp.model or self.model)
