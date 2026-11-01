"""Process-wide session usage accumulation (issue #365).

Complements ``provider_usage`` (shape normalization) and ``usage_recorder``
(scoped research estimates): this module accumulates token usage for the
whole app session so the quit-time summary
(Docs/superpowers/specs/2026-09-22-quit-time-session-summary-design.md)
can display a total without recomputing anything during shutdown.

Rules (spec, "Tap Points and the Boundary Rule"):
- record exactly once, where a provider response's usage is parsed;
- never raise — accounting must not be able to break a provider call;
- exact provider usage always wins over a char-based estimate.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

from .provider_usage import ProviderUsage
from .usage_recorder import estimate_tokens

__all__ = [
    "SessionUsageSnapshot",
    "SessionUsageLedger",
    "reset_for_tests",
    "session_usage",
]


@dataclass(frozen=True)
class SessionUsageSnapshot:
    """Immutable point-in-time view of the session accumulator."""

    exact_tokens: int = 0
    estimated_tokens: int = 0
    calls: int = 0
    embeddings_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.exact_tokens + self.estimated_tokens


class SessionUsageLedger:
    """Thread-safe, O(1)-memory session accumulator. Never raises."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._exact = 0
        self._estimated = 0
        self._calls = 0
        self._embeddings = 0

    def record_exact(self, usage: Optional[ProviderUsage]) -> None:
        """Add a provider-reported (exact) usage record."""
        try:
            if usage is None:
                return
            total = usage.total_tokens
            if total <= 0:
                return
            with self._lock:
                self._exact += total
                self._calls += 1
        except Exception:  # noqa: BLE001 - accounting must never break a call
            pass

    def record_estimate(
        self, prompt_text: Optional[str], completion_text: Optional[str]
    ) -> None:
        """Add a char-based (~4 chars/token) estimate record.

        Both texts absent means nothing to estimate -- a no-op, not a
        phantom two-token record from ``estimate_tokens("")``.
        """
        if prompt_text is None and completion_text is None:
            return
        try:
            total = estimate_tokens(prompt_text or "") + estimate_tokens(
                completion_text or ""
            )
            with self._lock:
                self._estimated += total
                self._calls += 1
        except Exception:  # noqa: BLE001
            pass

    def record_embeddings(self, usage_payload: Any) -> None:
        """Add an embedding-call usage payload (kept out of the LLM total).

        Embedding tokens are real spend but swamp the "how much did I
        chat" signal, so they accrue to their own bucket and the summary
        shows them as a separate line. Never raises.
        """
        try:
            usage = ProviderUsage.from_provider_payload(
                usage_payload, provider="openai", model="embeddings"
            )
            if usage is None or usage.total_tokens <= 0:
                return
            with self._lock:
                self._embeddings += usage.total_tokens
        except Exception:  # noqa: BLE001
            pass

    def record_provider_payload(
        self,
        usage_payload: Any,
        *,
        provider: str = "",
        model: str = "",
        fallback_texts: tuple[Optional[str], Optional[str]] = (None, None),
    ) -> None:
        """Record a raw provider usage payload; estimate when it carries none.

        ``ProviderUsage.from_provider_payload`` never raises and returns
        ``None`` for unrecognized/malformed shapes (fabricating no zeros).
        """
        try:
            usage = ProviderUsage.from_provider_payload(
                usage_payload, provider=provider or "unknown", model=model
            )
            if usage is not None and usage.total_tokens > 0:
                self.record_exact(usage)
                return
            prompt_text, completion_text = fallback_texts
            if prompt_text is None and completion_text is None:
                return
            self.record_estimate(prompt_text, completion_text)
        except Exception:  # noqa: BLE001
            pass

    def snapshot(self) -> SessionUsageSnapshot:
        with self._lock:
            return SessionUsageSnapshot(
                exact_tokens=self._exact,
                estimated_tokens=self._estimated,
                calls=self._calls,
                embeddings_tokens=self._embeddings,
            )


_LEDGER = SessionUsageLedger()


def session_usage() -> SessionUsageLedger:
    """The process-wide ledger singleton."""
    return _LEDGER


def record_stream_terminal_usage(stream: Any) -> None:
    """Record a hosted stream's terminal-turn usage (issue #365).

    Shared exhaustion tap for the hosted-engine stream shims
    (`LegacyLineStream`, `MoonshotStream`, `ZAIStream`): their terminal
    turn's usage is final only at natural exhaustion, so a consumer Stop
    (close without exhausting) never reaches here and undercounts by
    policy. Never raises; duck-typed on ``stream.terminal_turn.usage``.
    """
    try:
        usage = stream.terminal_turn.usage
        if usage:
            _LEDGER.record_provider_payload(usage)
    except Exception:  # noqa: BLE001 - accounting must never break a stream
        pass


def reset_for_tests() -> None:
    """Swap in a fresh singleton; call from test fixtures only."""
    global _LEDGER
    _LEDGER = SessionUsageLedger()
