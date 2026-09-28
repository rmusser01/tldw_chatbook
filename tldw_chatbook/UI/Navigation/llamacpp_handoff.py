"""Sanitized, exact-generation llama.cpp destination intents (ADR-114)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from ...LLM_Management.llamacpp_connection import LlamaCppConnectionTarget


@dataclass(frozen=True, slots=True)
class _LlamaCppIntent:
    api_url: str
    model_id: str
    generation: int
    runtime_owner: Literal["lab_process", "external_server"]

    def __post_init__(self) -> None:
        from ...LLM_Management.llamacpp_connection import LlamaCppConnectionTarget

        LlamaCppConnectionTarget(
            self.api_url, self.model_id, self.runtime_owner, self.generation
        )

    @classmethod
    def from_target(cls, target: LlamaCppConnectionTarget) -> Self:
        from ...LLM_Management.llamacpp_connection import LlamaCppConnectionTarget

        if type(target) is not LlamaCppConnectionTarget:
            raise TypeError("An exact verified llama.cpp target is required.")
        return cls(
            target.base_url,
            target.model_id,
            target.verification_generation,
            target.runtime_owner,
        )


@dataclass(frozen=True, slots=True)
class LlamaCppConsoleIntent(_LlamaCppIntent):
    """Apply the verified endpoint to the active Console session only."""


@dataclass(frozen=True, slots=True)
class LlamaCppDefaultIntent(_LlamaCppIntent):
    """Stage a verified target in Settings without persisting it."""


def owner_has_current_intent(owner: object, intent: object) -> bool:
    """Recheck readiness, generation and exact local claim before consumption."""
    from ...LLM_Management.llamacpp_connection import (
        LlamaCppConnectionOwner,
        LlamaCppConnectionTarget,
    )

    if type(owner) is not LlamaCppConnectionOwner or type(intent) not in {
        LlamaCppConsoleIntent,
        LlamaCppDefaultIntent,
    }:
        return False
    target = LlamaCppConnectionTarget(
        intent.api_url, intent.model_id, intent.runtime_owner, intent.generation
    )
    return owner.is_current(target)
