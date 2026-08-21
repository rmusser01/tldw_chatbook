"""Typed state for workspace-owned Change Review consent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChangeReviewState(str, Enum):
    """Availability or consent state for Change Review."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ChangeReviewCapability:
    """Global Change Review capability state."""

    state: ChangeReviewState


MISSING_CHANGE_REVIEW_REVISION = "missing"


@dataclass(frozen=True, slots=True)
class ChangeReviewConsent:
    """One durable per-workspace consent observation."""

    state: ChangeReviewState
    revision: str = ""


class ChangeReviewStateConflict(RuntimeError):
    """Raised when a consent compare-and-set observation is stale."""
