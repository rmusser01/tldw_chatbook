"""Context-scoped LLM token usage recording (task-16329).

The research budget ledger needs token counts to enforce ``max_tokens``,
but ``chat_api_call`` returns text only and provider handlers discard usage.
This module provides the plumbing seam: a recorder activated around
LLM-bearing calls (``usage_scope``), which ``chat_api_call`` feeds with
prompt + completion token ESTIMATES (~4 chars/token) for non-streaming
string responses whenever one is active -- zero overhead otherwise. When
providers expose real usage, callers can ``record_usage`` exact counts
through the same recorder and the estimates stop being the source.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

__all__ = [
    "UsageTokenRecorder",
    "active_recorder",
    "estimate_tokens",
    "usage_scope",
]

# ~4 characters per token is the standard rough ratio for English text;
# documented as an ESTIMATE -- the ledger snapshot labels token counts
# accordingly until real provider usage flows through record_usage().
_CHARS_PER_TOKEN = 4

_active_recorder: contextvars.ContextVar[Optional["UsageTokenRecorder"]] = (
    contextvars.ContextVar("active_usage_recorder", default=None)
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token), floored at 1 for non-empty
    text; empty text estimates 1 so a real exchange never counts as zero."""
    if not text:
        return 1
    return max(1, len(text) // _CHARS_PER_TOKEN)


class UsageTokenRecorder:
    """Accumulates prompt/completion token counts within a usage scope."""

    def __init__(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._exact_tokens = 0

    def record_usage(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        """Record exact, provider-reported counts.

        Args:
            prompt_tokens: Exact prompt tokens reported by the provider;
                ``None`` when the provider did not report that side (the
                missing side stays uncounted rather than zero-filled, so
                partial reports never masquerade as exact totals).
            completion_tokens: Exact completion tokens reported by the
                provider; ``None`` when unreported.
        """
        # A PARTIAL report (one side unreported) is not exact: the settled
        # total is incomplete, so it must not flip the ledger's
        # tokens_estimated flag (task-16814 follow-up).
        both_reported = prompt_tokens is not None and completion_tokens is not None
        if prompt_tokens is not None:
            self._prompt_tokens += max(0, int(prompt_tokens))
        if completion_tokens is not None:
            self._completion_tokens += max(0, int(completion_tokens))
        if both_reported:
            self._exact_tokens += max(0, int(prompt_tokens)) + max(
                0, int(completion_tokens)
            )

    def record_exchange(self, *, prompt_text: str, completion_text: str) -> None:
        """Record one non-streaming exchange by estimating both sides."""
        self.record_usage(
            prompt_tokens=estimate_tokens(prompt_text or ""),
            completion_tokens=estimate_tokens(completion_text or ""),
        )

    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    def completion_tokens(self) -> int:
        return self._completion_tokens

    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    def exact_tokens(self) -> int:
        """Settled tokens that came from provider-reported usage (not
        estimates) -- drives the ledger's tokens_estimated flag.

        Returns:
            The token count settled through ``record_usage`` so far.
        """
        return self._exact_tokens


def active_recorder() -> Optional[UsageTokenRecorder]:
    """The recorder for the current context, or None when no usage scope is
    active (the common case -- nothing records)."""
    return _active_recorder.get()


@contextmanager
def usage_scope() -> Iterator[UsageTokenRecorder]:
    """Activate a recorder for the enclosed (possibly async) work; scoped to
    the current context so concurrent runs keep separate ledgers."""
    recorder = UsageTokenRecorder()
    token = _active_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _active_recorder.reset(token)
