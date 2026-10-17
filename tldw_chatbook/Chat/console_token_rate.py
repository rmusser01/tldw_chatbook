# tldw_chatbook/Chat/console_token_rate.py
# Description: Tokens-per-second tracker and chip copy for the Console status strip.
#
"""Tokens-per-second measurement for the Console (TASK-35100).

While a response streams, the Console status strip shows a live smoothed
rate; when the response settles, the chip resolves to the overall figure
(provider-reported output tokens divided by the content window, so
time-to-first-token is excluded -- the "how fast does this endpoint
generate" number, not the "how long did the round trip take" one).

Design notes:

- The tracker is fed by ``ChatScreen``'s 0.2s transcript poll, the same
  tick that drives the transcript and the run chip. There are no per-chunk
  UI events in the Console (chunks land in the store; the screen pulls),
  so a poll-fed tracker is the one shape that covers every streaming path
  -- plain chat sends, agent tool-loop rounds, prefill, regeneration --
  without touching the controller.
- Live counts come from a token estimate over the accumulating message
  body (the same estimator vocabulary the live usage label's "~N tok" and
  the cost tracker already speak); the settled figure upgrades to the
  provider's ``ProviderUsage.output`` count when it attaches, falling back
  to the estimate for providers that report nothing.
- The EMA uses half-life semantics: ``half_life`` is the time over which a
  steady rate carries half the new weight, so the readout tracks real
  changes in about a second without per-tick jitter. Because the tracker
  is fed on every tick (streaming or not), a stalled endpoint's rate
  decays toward zero through the same math as any other tick -- no
  special stall path exists.
- The clock starts at the first observation of non-empty content, and a
  terminal observation for a message the tracker never saw streaming is
  ignored (hidden) -- a transcript first seen after its response finished
  must not manufacture an elapsed window measured from its own first
  sight.

Import discipline: this module is pure -- no Textual imports, no store,
no Console runtime. The token counter is injected so tests (and a future
caller with a different estimator) never depend on tokenizer availability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from tldw_chatbook.Chat.cost_display import format_token_count

__all__ = [
    "LIVE_FRAGMENT_FLOOR",
    "TokenRateReading",
    "TokenRateTracker",
    "format_rate_chip_live_fragment",
    "format_rate_chip_settled",
]

#: Message statuses that mean "this turn may still produce content". Any
#: other status settles the tracker. Mirrors the transcript's
#: ``_IN_FLIGHT_MESSAGE_STATUSES`` by value, not by import: the transcript
#: widget is a UI module and this file's import discipline excludes it.
_IN_FLIGHT_STATUSES = frozenset({"pending", "streaming"})


@dataclass(frozen=True)
class TokenRateReading:
    """One poll tick's measurement, already resolved to a display state.

    Attributes:
        kind: ``"hidden"`` (nothing to show), ``"live"`` (streaming rate),
            or ``"settled"`` (the finished response's overall figure).
        rate: Tokens per second, or ``None`` when unknown -- a live reading
            without one is not visible (nothing measured yet); a settled
            reading without one still shows its token total.
        total_tokens: The settled response's token total (provider count
            when known, estimate otherwise). ``None`` while live.
        elapsed_seconds: The settled response's content window
            (last-seen-streaming minus first-content), for the tooltip.
    """

    kind: str
    rate: float | None = None
    total_tokens: int | None = None
    elapsed_seconds: float | None = None


class TokenRateTracker:
    """Per-response tok/s state fed once per poll tick.

    One tracker tracks one response at a time, keyed by assistant message
    id: a new id resets the window (the previous turn's settled figure
    deliberately does not survive it), and ``None`` hides.

    Args:
        count_fn: Token estimator over the accumulated message body.
            Defaults to nothing (every reading hidden) -- callers inject
            ``estimate_tokens``-style callables; tests inject exact ones.
        half_life: EMA half-life in seconds; see the module docstring.
    """

    def __init__(
        self,
        *,
        count_fn: Callable[[str], int] | None = None,
        half_life: float = 1.0,
    ) -> None:
        self._count_fn = count_fn
        self.half_life = max(float(half_life), 1e-6)
        self._reset_window()

    def _reset_window(self) -> None:
        self._message_id: str | None = None
        self._first_content_at: float | None = None
        self._last_tokens: int | None = None
        self._last_at: float | None = None
        self._ema: float | None = None
        self._settled: TokenRateReading | None = None

    def reset(self) -> None:
        """Forget everything -- the next observation starts a fresh window."""
        self._reset_window()

    def observe(
        self,
        message_id: str | None,
        *,
        status: str,
        content: str,
        now: float,
        output_tokens: int | None = None,
    ) -> TokenRateReading:
        """Fold one poll tick in and return the reading to display.

        Args:
            message_id: The viewed session's newest assistant message id,
                or ``None`` when there is none.
            status: That message's store status (``"pending"``/
                ``"streaming"`` while in flight; anything else settles).
            content: The message's current body text.
            now: ``time.monotonic()`` reading for this tick.
            output_tokens: The provider's reported output total when the
                message carries usage; upgrades a settled estimate.
        """
        if message_id is None:
            self._reset_window()
            return TokenRateReading(kind="hidden")
        if message_id != self._message_id:
            self._reset_window()
            self._message_id = message_id
        if status in _IN_FLIGHT_STATUSES:
            if self._settled is not None:
                # A settled message re-observed as in flight is a new turn
                # on a recycled id (regenerate-in-place); start over.
                self._reset_window()
                self._message_id = message_id
        elif self._first_content_at is None:
            # Terminal with no streaming baseline: the tracker never watched
            # this response generate, so it must not invent an elapsed
            # window -- and it must forget the id, or a later usage attach
            # for this same message would settle from nothing.
            self._reset_window()
            return TokenRateReading(kind="hidden")
        else:
            return self._settle(content, now=now, output_tokens=output_tokens)
        tokens = self._count_fn(content) if self._count_fn is not None else 0
        if tokens <= 0:
            # Pre-content window (prompt processing): not generation time.
            return TokenRateReading(kind="hidden")
        if self._first_content_at is None or self._last_at is None:
            self._first_content_at = now
            self._last_tokens = tokens
            self._last_at = now
            return TokenRateReading(kind="hidden")
        dt = now - self._last_at
        if dt <= 0:
            # Timer overlap: hold the current estimate, keep the baseline
            # so the next well-ordered tick measures the full delta.
            return self._live_reading()
        instantaneous = max(0, tokens - (self._last_tokens or 0)) / dt
        alpha = 1.0 - 0.5 ** (dt / self.half_life)
        self._ema = (
            instantaneous
            if self._ema is None
            else self._ema + alpha * (instantaneous - self._ema)
        )
        self._last_tokens = tokens
        self._last_at = now
        return self._live_reading()

    def _live_reading(self) -> TokenRateReading:
        rate = self._ema
        if rate is not None and not math.isfinite(rate):
            rate = None
        return TokenRateReading(kind="live", rate=rate)

    def _settle(
        self, content: str, *, now: float, output_tokens: int | None
    ) -> TokenRateReading:
        """Resolve (or upgrade) the settled figure for the current message.

        The elapsed window latches at the FIRST settle: a later observation
        (the usage attach arriving after the terminal paint) upgrades the
        token total and its rate, but the window stays what the stream
        actually measured -- ``now`` by then is whenever the next poll ran,
        not when generation ended.
        """
        if self._settled is not None:
            settled = self._settled
            if output_tokens and output_tokens != settled.total_tokens:
                elapsed = settled.elapsed_seconds
                settled = TokenRateReading(
                    kind="settled",
                    total_tokens=output_tokens,
                    elapsed_seconds=elapsed,
                    rate=(output_tokens / elapsed if elapsed and elapsed > 0 else None),
                )
                self._settled = settled
            return settled
        elapsed = None
        if self._first_content_at is not None:
            elapsed = max(0.0, now - self._first_content_at)
        # A zero/falsy provider count means "not reported", not "no output"
        # -- fall back to the estimate the stream measured.
        total = output_tokens or None
        if total is None:
            total = self._count_fn(content) if self._count_fn is not None else 0
        rate = total / elapsed if elapsed and elapsed > 0 and total else None
        self._settled = TokenRateReading(
            kind="settled",
            rate=rate,
            total_tokens=total or None,
            elapsed_seconds=elapsed,
        )
        return self._settled


#: Live fragments below this rate are suppressed: the EMA decays toward
#: zero (never reaching it) whenever tokens stop flowing -- tool rounds,
#: approval waits -- and a "0.0 tok/s" fragment on the run chip would
#: mislabel a paused phase as crawling streaming (TASK-35160 review).
LIVE_FRAGMENT_FLOOR = 0.5


def format_rate_chip_live_fragment(reading: TokenRateReading) -> str:
    """The live rate as a fragment for the RUN chip's existing copy.

    Redesign (2026-10-16): the live rate no longer owns a strip slot --
    it rides ``"Run: Streaming response · 42.3 tok/s"``, so streaming
    never widens the transient cluster beyond the run chip's own
    variable-width contract and idle stays byte-identical to pre-feature.
    Review follow-up: fragments below :data:`LIVE_FRAGMENT_FLOOR` are
    suppressed -- during tool rounds and approval waits the EMA decays
    toward zero but never reaches it, and a "0.0 tok/s" badge would
    mislabel the paused phase.

    Args:
        reading: The tracker's reading for this tick.

    Returns:
        ``"42.3 tok/s"`` for a live reading at or above the floor;
        ``""`` otherwise (including settled readings -- a finished
        response's rate belongs to :func:`format_rate_chip_settled`,
        never the run chip).
    """
    if (
        reading.kind == "live"
        and reading.rate is not None
        and reading.rate >= LIVE_FRAGMENT_FLOOR
    ):
        return f"{reading.rate:.1f} tok/s"
    return ""


def format_rate_chip_settled(
    reading: TokenRateReading,
) -> tuple[bool, str, str]:
    """The settled response's ``(visible, label, tooltip)`` badge contract.

    Replaces the run chip's slot after it hides at terminal, so the two
    transient readouts are never on the row together. The label stays
    compact (``"80.0 tok/s avg"``); the detail a receipt would want --
    token total, measured window -- lives in the tooltip, where width is
    free.

    Args:
        reading: The tracker's reading for this tick.

    Returns:
        ``(False, "", "")`` unless the reading is settled with a rate or a
        token total; the ``"avg"`` marker distinguishes the overall figure
        from the run chip's live rate.
    """
    if reading.kind != "settled":
        return (False, "", "")
    label_segments = []
    if reading.rate is not None:
        label_segments.append(f"{reading.rate:.1f} tok/s avg")
    elif reading.total_tokens:
        label_segments.append(f"{format_token_count(reading.total_tokens)} tok")
    if not label_segments:
        return (False, "", "")
    tooltip_segments = []
    if reading.total_tokens:
        tooltip_segments.append(f"{format_token_count(reading.total_tokens)} tok")
    if reading.elapsed_seconds:
        tooltip_segments.append(f"{reading.elapsed_seconds:.1f}s")
    if reading.rate is None:
        tooltip_segments.append("rate unknown (no measurable window)")
    return (True, " ".join(label_segments), " · ".join(tooltip_segments))
