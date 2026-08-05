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
    """Normalized, disjoint token-usage buckets for one Console turn.

    Every provider reports token usage differently (some fold cached tokens
    into the input total, some report them separately, some omit cache
    entirely). ``from_provider_payload`` normalizes each provider's raw
    shape into these four buckets so downstream code -- pricing, display,
    persistence -- never has to know which provider produced them.

    Disjoint-bucket contract:
        ``uncached_input``, ``cache_read``, ``cache_write``, and ``output``
        never double-count the same tokens. Anthropic's native
        ``input_tokens`` already excludes cached tokens, so it is stored
        as-is; OpenAI's ``prompt_tokens``/``input_tokens`` INCLUDES cached
        tokens, so the adapter subtracts the cached count before storing
        ``uncached_input``. Summing the four buckets (``total_tokens``)
        therefore always yields a correct, non-inflated total.

    Never-store-dollars contract:
        Instances only ever hold token counts and provider/model
        identifiers -- never a computed price. Cost is derived at READ time
        by looking up ``provider``/``model`` in the pricing catalog against
        these buckets, so a later price-list correction re-prices every
        historical record instead of requiring a backfill.

    Captured-if-present contract:
        ``provider``, ``model``, and ``partial`` are best-effort metadata,
        not guaranteed on every instance. ``plus()`` (used to fold multiple
        provider calls from one agent turn into a single record) keeps
        whichever operand actually supplied a non-empty ``provider``/
        ``model`` rather than requiring both legs to agree, and treats
        ``partial`` as sticky: any incomplete leg marks the whole merged
        record partial, even if a later leg completed normally.

    Attributes:
        uncached_input: Input tokens NOT served from a prompt cache.
        cache_read: Input tokens served from an existing prompt cache.
        cache_write: Input tokens newly written to a prompt cache.
        output: Generated (completion) tokens.
        provider: Provider identifier the usage was captured against, or
            ``""`` when unknown.
        model: Model identifier the usage was captured against, or ``""``
            when unknown.
        partial: True when this record reflects an incomplete call (e.g. a
            stream stopped mid-generation) rather than a final usage report.
    """

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

    def plus(self, other: "ProviderUsage") -> "ProviderUsage":
        """Return the bucket-wise sum of two provider CALLS in one turn.

        An agent turn makes N provider calls; each is normalized on its own
        (raw payloads must never be key-merged across calls -- one call's
        stale ``cached_tokens`` beside another's ``prompt_tokens`` fabricates
        a cache read) and the disjoint buckets are summed here. ``partial``
        is sticky: any incomplete leg makes the whole turn's record partial.

        Args:
            other: The next provider call's normalized usage to fold into
                this one. Order does not matter for the token buckets (sum
                is commutative); for ``provider``/``model`` this instance's
                value wins when non-empty, otherwise ``other``'s does.

        Returns:
            A new ``ProviderUsage`` whose four token buckets are the
            element-wise sum of ``self`` and ``other``, whose ``provider``/
            ``model`` are the first non-empty value between the two (this
            instance preferred), and whose ``partial`` is True if either
            operand is partial.
        """
        return ProviderUsage(
            uncached_input=self.uncached_input + other.uncached_input,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            output=self.output + other.output,
            provider=self.provider or other.provider,
            model=self.model or other.model,
            partial=self.partial or other.partial,
        )

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
