"""Local speech dependency status, as displayable copy.

Deliberately pure, mirroring :mod:`lab_server_status`: it reads the shared
``DEPENDENCIES_AVAILABLE`` map and returns strings, so the Speech mode's
status chip is testable without mounting ``STTSWindow`` -- which is a
5,900-line widget that builds a TTS playground on compose.

The logic lived as four private methods on ``STTSWindow`` and was rendered
inside its sidebar. Speech's sidebar moved into the Lab frame's rail, so the
status had to move with it; extracting rather than reaching across let the
screen compose the chip *before* the deferred body exists, which is the whole
reason the body mount is deferred.
"""

from __future__ import annotations

from tldw_chatbook.Utils.optional_deps import (
    DEPENDENCIES_AVAILABLE,
    check_stt_deps,
    check_tts_deps,
)
from tldw_chatbook.UI.destination_recovery import optional_dependency_recovery_state

#: Stable selector the recovery state reports against, unchanged from when
#: this rendered inside the sidebar.
SPEECH_CAPABILITY_SELECTOR = "speech-capability-status"


def speech_dependencies_available() -> bool:
    """Report whether both local TTS and local STT are importable.

    Returns:
        True only when both dependency groups are present.
    """
    return bool(DEPENDENCIES_AVAILABLE.get("tts_processing", False)) and bool(
        DEPENDENCIES_AVAILABLE.get("stt_processing", False)
    )


def speech_dependency_recovery_state():
    """Build the recovery state describing what is missing and how to fix it.

    Returns:
        The shared optional-dependency recovery state, naming only the
        dependency groups that are actually absent.
    """
    missing_dependencies: list[str] = []
    if not DEPENDENCIES_AVAILABLE.get("tts_processing", False):
        missing_dependencies.append("local_tts")
    if not DEPENDENCIES_AVAILABLE.get("stt_processing", False):
        missing_dependencies.extend(
            ("transcription_faster_whisper", "speech_recording")
        )

    return optional_dependency_recovery_state(
        unavailable_what="Local speech providers",
        missing_dependencies=tuple(missing_dependencies),
        install_target=(
            'pip install "tldw_chatbook'
            '[local_tts,transcription_faster_whisper,speech_recording]"'
        ),
        stable_selector=SPEECH_CAPABILITY_SELECTOR,
        recovery_action="Settings > Speech",
    )


def speech_capability_text() -> str:
    """Return the one-line capability status for the Speech status chip.

    Re-checks the dependency probes first, so a user who installs the extras
    and returns to the screen sees the change rather than a cached "missing".

    Returns:
        ``"Local speech: ready"``, or the recovery state's visible copy.
    """
    check_tts_deps()
    check_stt_deps()

    if speech_dependencies_available():
        return "Local speech: ready"
    return "Local speech: dependencies missing"


def speech_capability_detail() -> str:
    """Return the full recovery taxonomy for the inspector.

    Headline / Why / Next / Recovery / Owner, including the exact pip
    command. This is ~14 rendered lines, which is why it lives in the
    inspector rather than the rail: rendered inline it buried the six view
    rows it was meant to sit beside, and rendered as a tooltip it would put
    the fix behind a pointer in a keyboard-first app.

    Returns:
        A confirmation line when both groups are present, otherwise the
        recovery state's full visible copy.
    """
    if speech_dependencies_available():
        return "Local TTS and STT dependencies are available."
    return speech_dependency_recovery_state().visible_copy


def speech_capability_tooltip() -> str:
    """Return install guidance for the capability chip's tooltip.

    Returns:
        A confirmation when both groups are present, otherwise the recovery
        state's disabled tooltip naming the install target.
    """
    if speech_dependencies_available():
        return "Local TTS and STT dependencies are available."
    return speech_dependency_recovery_state().disabled_tooltip
