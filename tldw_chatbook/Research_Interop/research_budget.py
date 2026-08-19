"""Research budget ledger (task-16323, extends ADR-068's engine contract).

Reserve-and-settle budget enforcement for local research runs, porting two
reference designs: mole's ledger discipline (reserve before execution,
settle after, overshoot recorded not hidden) and tldw_server's
``limits.py`` semantics (limits parsed from ``limits_json``, missing keys
mean unlimited, violations raise structured
``research_limit_exceeded:<key>`` errors).

Axes:
- ``max_searches`` — query fan-out units; the engine reserves the fan-out
  cap BEFORE phase 1 spends and settles the actual query count after.
- ``max_fetched_docs`` — raw result documents processed; ``allot_docs``
  caps a batch at the remaining budget before it is handed onward.
- ``max_runtime_seconds`` — wall-clock deadline checked between phases.
- ``max_tokens`` — token reservations. Infrastructure only for now: the
  deep-search pipeline's LLM calls do not report usage, so the engine
  reserves at pipeline-call boundaries and records what it can honestly
  measure; per-LLM-call enforcement waits on usage plumbing.

All counters are non-negative; settling more than reserved (measured
overshoot) records the difference instead of erroring or clamping.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

__all__ = ["BudgetLedger", "ResearchLimitExceeded"]


class ResearchLimitExceeded(Exception):
    """Structured budget violation (server ``limits.py`` error contract)."""

    def __init__(self, limit_key: str, message: str) -> None:
        super().__init__(f"research_limit_exceeded:{limit_key}: {message}")
        self.limit_key = limit_key


def _limit_value(limits: Mapping[str, Any], key: str) -> float | None:
    """Coerce one limit to a non-negative finite number; anything else means
    unlimited (None) — mirrors the server's missing-key-is-unlimited rule
    and stays defensive against hand-edited ``limits_json``. An EXPLICIT 0
    is a valid zero budget (immediately exhausted), not "unlimited"."""
    raw = limits.get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0 or value != value:  # negative or NaN -> unlimited
        return None
    return value


class BudgetLedger:
    """Mutable reserve/settle ledger for one research run."""

    def __init__(self, limits: Mapping[str, Any] | None = None) -> None:
        limits = dict(limits or {})
        self.max_searches = _limit_value(limits, "max_searches")
        self.max_fetched_docs = _limit_value(limits, "max_fetched_docs")
        self.max_runtime_seconds = _limit_value(limits, "max_runtime_seconds")
        self.max_tokens = _limit_value(limits, "max_tokens")
        self.searches_used = 0
        self.searches_reserved = 0
        self.searches_overshoot = 0
        self.docs_used = 0
        self.tokens_reserved = 0
        self.tokens_settled = 0
        self.tokens_settled_exact = 0
        self._start_monotonic = time.monotonic()

    @classmethod
    def from_limits(cls, limits: Mapping[str, Any] | None) -> "BudgetLedger":
        return cls(limits)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any] | None,
        limits: Mapping[str, Any] | None,
    ) -> "BudgetLedger":
        """Rebuild a ledger that has already spent part of its budget.

        task-18060: ``execute_run`` used to rebuild from limits on every entry
        while writing ``budget_ledger.json`` and never reading it back, so a
        resumed run was granted its whole budget again. Resume is routine once
        runs survive an app exit, which turns that into a leak.

        Reservations are deliberately NOT restored. A reservation belongs to an
        in-flight call that died with its executor; carrying it forward would
        hold budget nothing is going to spend.

        Args:
            snapshot: A prior ``snapshot()`` payload, or None for a run that
                has never executed.
            limits: The run's CURRENT effective limits (any approved
                plan-review patch already merged over the run's stored
                limits); these win over the snapshot's own ``limits`` when
                provided, since a snapshot written before a plan-review
                patch would otherwise resurrect the stale, pre-patch values
                (same bug class as the Qodo PR 1766 fix a few lines above
                this one in ``local_research_engine.py``, for
                ``max_iterations``). Falls back to the snapshot's limits
                only when this argument is empty/None.

        Returns:
            A ledger whose used counters continue from the snapshot.
        """
        snapshot = snapshot or {}
        ledger = cls(dict(limits or snapshot.get("limits") or {}))
        ledger.searches_used = int(snapshot.get("searches_used") or 0)
        ledger.searches_overshoot = int(snapshot.get("searches_overshoot") or 0)
        ledger.docs_used = int(snapshot.get("docs_used") or 0)
        ledger.tokens_settled = int(snapshot.get("tokens_settled") or 0)
        # A snapshot records whether any settled usage was estimated rather than
        # provider-reported, not the exact-token count itself; restoring the
        # settled total as fully exact would erase that distinction, so an
        # estimated snapshot restores as estimated.
        if snapshot.get("tokens_estimated"):
            ledger.tokens_settled_exact = 0
        else:
            ledger.tokens_settled_exact = ledger.tokens_settled
        return ledger

    # -- searches ---------------------------------------------------------

    def remaining_searches(self) -> int | None:
        """Remaining spendable searches, net of outstanding reservations
        (reserved but not yet settled) — over-reserving before settlement
        must fail exactly like overspending after it."""
        if self.max_searches is None:
            return None
        outstanding = max(0, self.searches_reserved - self.searches_used)
        return max(0, int(self.max_searches) - self.searches_used - outstanding)

    def reserve_searches(self, count: int) -> None:
        remaining = self.remaining_searches()
        if remaining is not None and count > remaining:
            raise ResearchLimitExceeded(
                "max_searches",
                f"reserving {count} search(es) would exceed the remaining budget of {remaining}",
            )
        self.searches_reserved += max(0, int(count))

    def release_searches(self, count: int) -> None:
        """Return unused reservations (task-16814): when the pipeline stops
        its fan-out early (e.g. the phase-1 deadline), reserved-but-never-
        executed searches must not keep counting against the budget."""
        self.searches_reserved -= min(max(0, int(count)), self.searches_reserved)

    def settle_searches(self, count: int) -> None:
        count = max(0, int(count))
        # Overshoot is measured against RESERVATIONS (mole semantics:
        # what actually ran vs what was claimed), never an error at
        # settlement.
        self.searches_overshoot = max(
            0, (self.searches_used + count) - self.searches_reserved
        )
        self.searches_used += count

    # -- fetched docs -----------------------------------------------------

    def remaining_docs(self) -> int | None:
        if self.max_fetched_docs is None:
            return None
        return max(0, int(self.max_fetched_docs) - self.docs_used)

    def allot_docs(self, count: int) -> int:
        """Cap a batch of fetched docs at the remaining budget (the cap
        happens BEFORE the docs are processed onward); a zero remaining
        budget is an exhaustion error, not a silent empty batch -- unless
        the batch itself is empty, which consumes no budget (task-16814:
        ``allot_docs(0)`` must return 0 even on an exhausted budget)."""
        count = int(count)
        if count <= 0:
            return 0
        remaining = self.remaining_docs()
        if remaining is not None:
            if remaining == 0:
                raise ResearchLimitExceeded(
                    "max_fetched_docs", "fetched-doc budget is exhausted"
                )
            return min(count, remaining)
        return count

    def settle_docs(self, count: int) -> None:
        self.docs_used += max(0, int(count))

    # -- tokens -----------------------------------------------------------

    def reserve_tokens(self, count: int) -> None:
        if self.max_tokens is None:
            self.tokens_reserved += max(0, int(count))
            return
        if self.tokens_reserved + int(count) > int(self.max_tokens):
            raise ResearchLimitExceeded(
                "max_tokens",
                f"reserving {count} token(s) would exceed the budget of "
                f"{int(self.max_tokens)} (already reserved {self.tokens_reserved})",
            )
        self.tokens_reserved += max(0, int(count))

    def settle_tokens(self, count: int, *, exact: bool = False) -> None:
        """Record settled token usage; ``exact`` marks whether the counts
        came from provider-reported usage rather than estimates
        (task-16814: the snapshot flag must reflect reality)."""
        count = max(0, int(count))
        self.tokens_settled += count
        if exact:
            self.tokens_settled_exact += count

    def check_tokens(self) -> None:
        """Raise when settled token usage has reached the budget (checked
        between LLM-bearing units of work; enforcement is post-settlement
        because estimates arrive after calls complete)."""
        if self.max_tokens is None:
            return
        if self.tokens_settled >= int(self.max_tokens):
            raise ResearchLimitExceeded(
                "max_tokens",
                f"token budget exhausted ({self.tokens_settled} settled of "
                f"{int(self.max_tokens)})",
            )

    # -- runtime ----------------------------------------------------------

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_monotonic

    def check_runtime(self) -> None:
        if self.max_runtime_seconds is None:
            return
        elapsed = self.elapsed_seconds()
        if elapsed > self.max_runtime_seconds:
            raise ResearchLimitExceeded(
                "max_runtime_seconds",
                f"run elapsed {elapsed:.1f}s exceeds the {self.max_runtime_seconds:.1f}s budget",
            )

    # -- persistence ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "limits": {
                "max_searches": self.max_searches,
                "max_fetched_docs": self.max_fetched_docs,
                "max_runtime_seconds": self.max_runtime_seconds,
                "max_tokens": self.max_tokens,
            },
            "searches_used": self.searches_used,
            "searches_overshoot": self.searches_overshoot,
            "docs_used": self.docs_used,
            "tokens_reserved": self.tokens_reserved,
            "tokens_settled": self.tokens_settled,
            # True when any settled usage was estimated rather than
            # provider-reported (task-16814).
            "tokens_estimated": self.tokens_settled_exact < self.tokens_settled,
            "runtime_elapsed_s": round(self.elapsed_seconds(), 3),
        }
