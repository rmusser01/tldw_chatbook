"""Secret-free, generation-fenced vLLM cross-screen intents."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from ...Chat.provider_endpoint_contract import resolve_provider_endpoint

if TYPE_CHECKING:
    from ..LLM_Management.vllm_setup import VllmConnectionTarget


_WINDOWS_ROOT = re.compile(r"^[A-Za-z]:[/\\]")


def _is_admissible_model_id(value: object) -> bool:
    if type(value) is not str or not 1 <= len(value) <= 120:
        return False
    if value != " ".join(value.split()) or not value.isprintable():
        return False
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value
    ):
        return False
    lowered = value.casefold()
    if (
        lowered.startswith("file:")
        or value.startswith(("/", "./", "../", "~/", "\\\\", "//"))
        or _WINDOWS_ROOT.match(value)
        or "\\" in value
        or lowered.endswith(".gguf")
    ):
        return False
    return all(segment not in {".", ".."} for segment in value.split("/"))


def _validate_intent_fields(
    api_url: object, model_id: object, generation: object
) -> None:
    if type(api_url) is not str:
        raise TypeError("vLLM handoff URL must be exact text")
    resolution = resolve_provider_endpoint("vllm", api_url)
    if resolution.errors or resolution.persisted_endpoint != api_url:
        raise ValueError("vLLM handoff URL must be a canonical provider endpoint")
    if not _is_admissible_model_id(model_id):
        raise ValueError("vLLM handoff model identifier is invalid")
    if type(generation) is not int or generation < 1:
        raise ValueError("vLLM handoff generation must be a positive exact integer")


def _intent_fields_from_target(
    target: VllmConnectionTarget,
) -> tuple[str, str, int]:
    from ..LLM_Management.vllm_setup import VllmConnectionTarget

    if type(target) is not VllmConnectionTarget:
        raise TypeError("vLLM handoff requires an exact connection target")
    if target.provider_key != "vllm":
        raise ValueError("vLLM handoff target provider is invalid")
    return target.api_url, target.model_id, target.generation


@dataclass(frozen=True, slots=True)
class VllmConsoleIntent:
    """Apply one verified target to the active Console session only."""

    api_url: str
    model_id: str
    generation: int

    def __post_init__(self) -> None:
        _validate_intent_fields(self.api_url, self.model_id, self.generation)

    @classmethod
    def from_target(cls, target: VllmConnectionTarget) -> VllmConsoleIntent:
        """Detach only the non-secret fields Console is allowed to consume.

        Args:
            target: Exact connection target returned by the readiness owner.

        Returns:
            Immutable Console intent for the target's generation.

        Raises:
            TypeError: The target is not an exact connection-target instance.
            ValueError: The target contains invalid provider or intent fields.
        """

        return cls(*_intent_fields_from_target(target))


@dataclass(frozen=True, slots=True)
class VllmDefaultIntent:
    """Prefill one verified target in Settings without saving it."""

    api_url: str
    model_id: str
    generation: int

    def __post_init__(self) -> None:
        _validate_intent_fields(self.api_url, self.model_id, self.generation)

    @classmethod
    def from_target(cls, target: VllmConnectionTarget) -> VllmDefaultIntent:
        """Detach only the non-secret fields Settings is allowed to stage.

        Args:
            target: Exact connection target returned by the readiness owner.

        Returns:
            Immutable Settings intent for the target's generation.

        Raises:
            TypeError: The target is not an exact connection-target instance.
            ValueError: The target contains invalid provider or intent fields.
        """

        return cls(*_intent_fields_from_target(target))


VllmHandoffIntent: TypeAlias = VllmConsoleIntent | VllmDefaultIntent


def owner_has_current_intent(
    owner: object,
    intent: VllmHandoffIntent,
) -> bool:
    """Return whether ``intent`` still names the owner's exact ready target.

    Args:
        owner: Readiness owner exposing the current snapshot.
        intent: Detached handoff to compare with the owner's current target.

    Returns:
        Whether exact types, readiness, generation, model and endpoint match.
    """

    snapshot_method = getattr(owner, "snapshot", None)
    if not callable(snapshot_method):
        return False
    if type(intent) not in {VllmConsoleIntent, VllmDefaultIntent}:
        return False
    from ..LLM_Management.vllm_setup import (
        VllmConnectionTarget,
        VllmReadinessState,
    )

    snapshot = snapshot_method()
    target = snapshot.target
    token = snapshot.current_token
    return bool(
        snapshot.state is VllmReadinessState.READY
        and type(target) is VllmConnectionTarget
        and token is not None
        and token.generation == intent.generation
        and target.generation == intent.generation
        and target.provider_key == "vllm"
        and target.api_url == intent.api_url
        and target.model_id == intent.model_id
    )


__all__ = [
    "VllmConsoleIntent",
    "VllmDefaultIntent",
    "VllmHandoffIntent",
    "owner_has_current_intent",
]
