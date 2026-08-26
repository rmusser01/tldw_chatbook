"""Shared, non-sensitive UI actions for TTS profile dependency recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from tldw_chatbook.TTS.profile_service import TTSProfileDependencyProjection

TTSProfileDependencyOperation: TypeAlias = Literal[
    "open_audio_cpp_settings",
    "open_speech_lab_apply",
    "generate_new_profile",
]


@dataclass(frozen=True, slots=True)
class TTSProfileDependencyActionProjection:
    """One visible action whose label and execution cannot diverge."""

    operation: TTSProfileDependencyOperation
    role: Literal["blocker", "advisory"]
    label: str
    tooltip: str


_ACTIONS = {
    "open_audio_cpp_settings": TTSProfileDependencyActionProjection(
        operation="open_audio_cpp_settings",
        role="blocker",
        label="Configure model",
        tooltip="Open global audio.cpp settings to configure the required model.",
    ),
    "open_speech_lab_apply": TTSProfileDependencyActionProjection(
        operation="open_speech_lab_apply",
        role="blocker",
        label="Apply settings",
        tooltip="Open this exact voice in Speech Lab to apply the saved model settings.",
    ),
    "generate_new_profile": TTSProfileDependencyActionProjection(
        operation="generate_new_profile",
        role="advisory",
        label="Preview & save new",
        tooltip=(
            "Preview or generate this exact voice, save it as a new profile, "
            "then reassign or remove the legacy profile."
        ),
    ),
}


def dependency_recovery_actions(
    dependency: TTSProfileDependencyProjection,
) -> tuple[TTSProfileDependencyActionProjection, ...]:
    """Return blocker first and independent advisory second."""

    if type(dependency) is not TTSProfileDependencyProjection:
        raise TypeError("dependency must be a TTS profile dependency projection")
    operations = tuple(
        operation
        for operation in (dependency.action, dependency.advisory_action)
        if operation != "none"
    )
    return tuple(_ACTIONS[operation] for operation in operations)


__all__ = [
    "TTSProfileDependencyActionProjection",
    "TTSProfileDependencyOperation",
    "dependency_recovery_actions",
]
