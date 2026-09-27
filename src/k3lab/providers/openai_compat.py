from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class ProviderResponse:
    assistant_message: dict[str, Any]
    usage: dict[str, Any]


class OpenAICompatibleProvider:
    """Thin adapter for OpenAI-compatible Chat Completions endpoints."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **extra: Any,
    ) -> ProviderResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            **extra,
        )
        message = response.choices[0].message.model_dump(exclude_none=True)
        usage = (
            response.usage.model_dump(exclude_none=True)
            if response.usage is not None
            else {}
        )
        return ProviderResponse(assistant_message=message, usage=usage)
