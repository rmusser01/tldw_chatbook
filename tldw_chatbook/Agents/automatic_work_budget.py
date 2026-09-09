"""Typed, body-free automatic-work policy and ledger projections (ADR-134/135)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RuntimeRecoveryResult:
    """Body-free projection captured by the sole runtime startup transaction."""

    owner_id: str
    recovered_chains: tuple[str, ...]
    goals: tuple[tuple[str, int, str], ...]


class AutomaticWorkRefused(RuntimeError):
    """Admission refused with a stable reason, never a model or tool body."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)

    @property
    def termination_reason(self):
        from .agent_models import RunTerminationReason

        if self.reason == "cancelled":
            return RunTerminationReason.CANCELLED
        if self.reason in {
            "generation_budget",
            "child_launch_budget",
            "model_call_budget",
            "tokens_budget",
            "output_tokens_budget",
            "wall_budget",
            "iteration_wall_budget",
        }:
            return RunTerminationReason.AUTOMATIC_LIMIT
        if self.reason in {
            "history_unavailable",
            "review_required",
            "clock_unknown",
            "clock_reversed",
        }:
            return RunTerminationReason.UNKNOWN_EFFECT
        return RunTerminationReason.AUTHORITY_CHANGED


@dataclass(frozen=True)
class AutomaticWorkLimits:
    generations: int = 3
    child_launches: int = 6
    model_calls: int = 32
    budget_tokens: int = 500_000
    output_tokens: int = 8192
    wall_seconds: int = 900

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError("limits must be nonnegative SQLite integers")

    def resources(self) -> dict[str, int]:
        """Return finite admission ceilings for the four charged resources."""
        return {
            "generation": self.generations,
            "child_launch": self.child_launches,
            "model_call": self.model_calls,
            "tokens": self.budget_tokens,
        }

    @classmethod
    def from_settings(cls, attempt_kind: str = "fleet_wake") -> AutomaticWorkLimits:
        """Snapshot finite agent limits with environment precedence and defaults."""
        from .run_log import _setting

        if attempt_kind not in {"fleet_wake", "goal_iteration"}:
            raise ValueError("unknown automatic attempt kind")
        defaults = cls(child_launches=0) if attempt_kind == "goal_iteration" else cls()
        prefix = "max_goal" if attempt_kind == "goal_iteration" else "max_autowake"
        values = {}
        for field in fields(defaults):
            default = getattr(defaults, field.name)
            value = _setting(f"{prefix}_{field.name}", default)
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    value = default
            values[field.name] = (
                value if type(value) is int and 0 <= value < 2**63 else default
            )
        return cls(**values)


@dataclass(frozen=True)
class AutomaticWorkReservation:
    id: str
    chain_id: str
    owner_id: str
    kind: str
    amount: int
    state: str
    actual_amount: int | None


@dataclass(frozen=True)
class AutomaticWorkSnapshot:
    chain_id: str
    conversation_id: str
    limits: AutomaticWorkLimits
    status: str
    pause_reason: str | None
    started_at: float | None
    deadline_at: float | None
    used: Mapping[str, int]
    reserved: Mapping[str, int]
    available: Mapping[str, int]
    uncertain: bool


@dataclass(frozen=True)
class AutomaticWakeAttempt:
    id: str
    chain_id: str
    conversation_id: str
    session_id: str
    owner_id: str
    state: str
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class GoalAttempt:
    """An iteration link proves an empty source batch is goal authority."""

    id: str
    goal_id: str
    ordinal: int
    chain_id: str
    conversation_id: str
    session_id: str
    owner_id: str
    state: str
