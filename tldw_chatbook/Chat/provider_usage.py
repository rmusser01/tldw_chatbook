"""Normalized per-message token usage for the Console cost ticker.

Buckets are DISJOINT (spec: 2026-08-01-console-cost-ticker-design.md):
Anthropic's input_tokens already excludes cached tokens; OpenAI's
prompt_tokens includes them, so the adapters subtract. Dollars are never
stored — pricing is applied at read time by the pricing catalog.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _as_count(value: Any) -> int:
    # OverflowError: `int(float("inf"))` raises it, and `json.loads` accepts
    # the `Infinity` literal -- so a provider payload or a corrupt stored row
    # can reach here with a non-finite count. Degrading to 0 keeps the same
    # contract as every other unusable value.
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(count, 0)


def as_seconds(value: Any) -> float:
    """Coerce a wire/stored value to a finite, non-negative duration.

    The single sanitizer for every seconds-valued field: `ProviderUsage`'s
    own JSON restore and the realtime wiring that captures a transcription
    duration both route through it, so "what counts as a usable duration"
    has one definition.

    `max(value, 0.0)` alone is not enough. Every comparison with NaN is
    False, so `max(nan, 0.0)` returns NaN, and `max(inf, 0.0)` returns inf
    -- both would land in `transcription_seconds`, survive `plus()`, and be
    written to the database as JSON.

    Args:
        value: Any raw value claiming to be a number of seconds.

    Returns:
        The duration as a float, clamped to >= 0.0, with anything
        unparseable, negative or non-finite (NaN, +/-inf) reported as 0.0.
    """
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(seconds):
        return 0.0
    return max(seconds, 0.0)


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

    Audio metadata (task-2363), NOT part of the disjoint-bucket contract:
        ``audio_input``/``audio_output`` are the AUDIO-token portion of
        ``uncached_input``+``cache_read`` and ``output`` respectively (a
        SUBSET, live-confirmed on realtime `response.done`'s
        ``input_token_details``/``output_token_details`` -- see
        `LLM_Calls/realtime/openai_session.py`'s ground-truth header),
        never summed into ``total_tokens`` separately -- doing so would
        double-count. ``transcription_seconds`` is a different unit
        entirely (input-audio transcription duration, from a SEPARATE wire
        event -- `conversation.item.input_audio_transcription.completed`'s
        own ``usage: {"type": "duration", "seconds": N}``, independent of
        `response.done`'s token usage). Realtime is billed per audio
        MINUTE, not per audio token, so none of these three fields feed
        `LLM_Calls/pricing_catalog.py`'s cost math today -- captured for a
        future cost-chip task, deliberately inert for billing until then.

    Attributes:
        uncached_input: Input tokens NOT served from a prompt cache.
        cache_read: Input tokens served from an existing prompt cache.
        cache_write: Input tokens newly written to a prompt cache.
        output: Generated (completion) tokens.
        audio_input: Of ``uncached_input``+``cache_read``, how many were
            audio tokens (realtime only; 0 for every other provider/shape).
        audio_output: Of ``output``, how many were audio tokens (realtime
            only; 0 for every other provider/shape).
        transcription_seconds: Duration, in seconds, of input audio the
            provider transcribed for this turn (realtime only; 0 for every
            other provider/shape). Not a token count.
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
    audio_input: int = 0
    audio_output: int = 0
    transcription_seconds: float = 0.0
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
            A new ``ProviderUsage`` whose token buckets (including the
            audio subset counts and transcription duration) are the
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
            audio_input=self.audio_input + other.audio_input,
            audio_output=self.audio_output + other.audio_output,
            transcription_seconds=(
                self.transcription_seconds + other.transcription_seconds
            ),
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
            audio_input=_as_count(data.get("audio_input")),
            audio_output=_as_count(data.get("audio_output")),
            transcription_seconds=as_seconds(data.get("transcription_seconds")),
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
        #
        # The Realtime API spells the same block `input_token_details`
        # (SINGULAR "token"), live-confirmed on `response.done`. Without
        # this alias its payload fell through to the Anthropic-native
        # branch below, which reads no details at all — so every cached
        # token in a realtime session was priced as uncached input, and a
        # realtime session is nearly all cached input by construction (the
        # whole conversation is resent as context on every turn).
        input_details = payload.get("input_tokens_details")
        if not isinstance(input_details, Mapping):
            input_details = payload.get("input_token_details")
        if isinstance(input_details, Mapping):
            total_input = _as_count(payload.get("input_tokens"))
            cached = _as_count(input_details.get("cached_tokens"))
            # Realtime-only (task-2363, live-confirmed on `response.done`
            # -- see openai_session.py's ground-truth header): both
            # `input_token_details` and `output_token_details` split into
            # `text_tokens`/`audio_tokens`. Absent on every other shape
            # sharing this branch (the Responses API rarely carries audio),
            # in which case `_as_count` defaults both to 0.
            output_details = payload.get("output_token_details")
            audio_output = (
                _as_count(output_details.get("audio_tokens"))
                if isinstance(output_details, Mapping)
                else 0
            )
            return cls(
                uncached_input=max(total_input - cached, 0),
                cache_read=cached,
                output=_as_count(payload.get("output_tokens")),
                audio_input=_as_count(input_details.get("audio_tokens")),
                audio_output=audio_output,
                **common,
            )
        # OpenAI chat-completions shape: prompt_tokens INCLUDES cached tokens.
        # `cache_creation_tokens` is our own extension (TASK-18607): the
        # Console gateway's normalization preserves Anthropic's write bucket
        # under this key -- also folded into prompt_tokens -- so the budget
        # can price writes at their real rate through the normalized path.
        if "prompt_tokens" in payload:
            total_input = _as_count(payload.get("prompt_tokens"))
            details = payload.get("prompt_tokens_details")
            cached = (
                _as_count(details.get("cached_tokens"))
                if isinstance(details, Mapping)
                else 0
            )
            cache_write = (
                _as_count(details.get("cache_creation_tokens"))
                if isinstance(details, Mapping)
                else 0
            )
            return cls(
                uncached_input=max(total_input - cached - cache_write, 0),
                cache_read=cached,
                cache_write=cache_write,
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
