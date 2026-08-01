"""Normalized per-message token usage for the Console cost ticker.

Buckets are DISJOINT (spec: 2026-08-01-console-cost-ticker-design.md):
Anthropic's input_tokens already excludes cached tokens; OpenAI's
prompt_tokens includes them, so the adapters subtract. Dollars are never
stored — pricing is applied at read time by the pricing catalog.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _as_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    uncached_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    provider: str = ""
    model: str = ""
    partial: bool = False

    @property
    def total_tokens(self) -> int:
        return self.uncached_input + self.cache_read + self.cache_write + self.output

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | None) -> "ProviderUsage | None":
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        return cls(
            uncached_input=_as_count(data.get("uncached_input")),
            cache_read=_as_count(data.get("cache_read")),
            cache_write=_as_count(data.get("cache_write")),
            output=_as_count(data.get("output")),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            partial=bool(data.get("partial")),
        )

    @classmethod
    def from_provider_payload(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        provider: str,
        model: str,
        partial: bool = False,
    ) -> "ProviderUsage | None":
        if not isinstance(payload, Mapping):
            return None
        common = {"provider": provider, "model": model, "partial": partial}
        # OpenAI Responses API: has input_tokens like Anthropic — the
        # input_tokens_details key disambiguates, so check it FIRST.
        if isinstance(payload.get("input_tokens_details"), Mapping):
            total_input = _as_count(payload.get("input_tokens"))
            cached = _as_count(payload["input_tokens_details"].get("cached_tokens"))
            return cls(
                uncached_input=max(total_input - cached, 0),
                cache_read=cached,
                output=_as_count(payload.get("output_tokens")),
                **common,
            )
        # OpenAI chat-completions shape: prompt_tokens INCLUDES cached tokens.
        if "prompt_tokens" in payload:
            total_input = _as_count(payload.get("prompt_tokens"))
            details = payload.get("prompt_tokens_details")
            cached = (
                _as_count(details.get("cached_tokens"))
                if isinstance(details, Mapping)
                else 0
            )
            return cls(
                uncached_input=max(total_input - cached, 0),
                cache_read=cached,
                output=_as_count(payload.get("completion_tokens")),
                **common,
            )
        # Anthropic-native shape: buckets are already disjoint.
        if "input_tokens" in payload:
            return cls(
                uncached_input=_as_count(payload.get("input_tokens")),
                cache_read=_as_count(payload.get("cache_read_input_tokens")),
                cache_write=_as_count(payload.get("cache_creation_input_tokens")),
                output=_as_count(payload.get("output_tokens")),
                **common,
            )
        return None
