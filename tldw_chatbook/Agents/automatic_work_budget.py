"""Typed, body-free automatic-work policy and ledger projections (ADR-134/135)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields


class AutomaticWorkRefused(RuntimeError):
    """Admission refused with a stable reason, never a model or tool body."""


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
    def from_settings(cls) -> AutomaticWorkLimits:
        """Snapshot finite agent limits with environment precedence and defaults."""
        from .run_log import _setting

        defaults = cls()
        values = {}
        for field in fields(defaults):
            default = getattr(defaults, field.name)
            value = _setting(f"max_autowake_{field.name}", default)
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
