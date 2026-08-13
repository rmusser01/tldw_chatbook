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

from importlib.util import find_spec

from tldw_chatbook.UI.destination_recovery import optional_dependency_recovery_state
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechLocalDependencyAvailability,
)
from tldw_chatbook.Utils.optional_deps import (
    DEPENDENCIES_AVAILABLE,
    check_stt_deps,
    check_tts_deps,
)

#: Stable selector the recovery state reports against, unchanged from when
#: this rendered inside the sidebar.
SPEECH_CAPABILITY_SELECTOR = "speech-capability-status"

_LOCAL_CAPABILITIES = (
    ("Local transcription", "stt", "transcription_faster_whisper"),
    ("Local Kokoro", "kokoro", "local_tts"),
    ("Local Chatterbox", "chatterbox", "chatterbox"),
    ("Local Higgs", "higgs", "higgs_tts"),
)


def speech_dependencies_available() -> bool:
    """Report whether every independently presented local capability is ready.

    Returns:
        True only when all four local capabilities are present.
    """
    dependencies = speech_local_dependency_availability()
    return all(
        getattr(dependencies, attribute)
        for _label, attribute, _extra in _LOCAL_CAPABILITIES
    )


def speech_local_dependency_availability(
    *,
    refresh: bool = False,
) -> SpeechLocalDependencyAvailability:
    """Return independent local dependency facts without provider inference.

    Args:
        refresh: Re-run non-importing local module-presence probes.

    Returns:
        A four-capability snapshot. External provider readiness is deliberately
        absent because local imports cannot prove it.
    """

    if refresh:
        return SpeechLocalDependencyAvailability(
            stt=any(
                _speech_dependency_installed(module_name)
                for module_name in (
                    "nemo_toolkit",
                    "faster_whisper",
                    "lightning_whisper_mlx",
                    "parakeet_mlx",
                )
            ),
            kokoro=_speech_dependency_installed("kokoro_onnx"),
            chatterbox=_speech_dependency_installed("chatterbox"),
            higgs=_speech_dependency_installed("boson_multimodal"),
        )
    return SpeechLocalDependencyAvailability(
        stt=bool(DEPENDENCIES_AVAILABLE.get("stt_processing", False)),
        kokoro=bool(DEPENDENCIES_AVAILABLE.get("kokoro_onnx", False)),
        chatterbox=bool(DEPENDENCIES_AVAILABLE.get("chatterbox", False)),
        higgs=bool(DEPENDENCIES_AVAILABLE.get("higgs_tts", False)),
    )


def _speech_dependency_installed(module_name: str) -> bool:
    """Check module presence without importing or initializing a runtime."""

    try:
        return find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def speech_dependency_recovery_state():
    """Build the recovery state describing what is missing and how to fix it.

    Returns:
        The shared optional-dependency recovery state, naming only the
        dependency groups that are actually absent.
    """
    dependencies = speech_local_dependency_availability()
    missing_capabilities = [
        (label, extra)
        for label, attribute, extra in _LOCAL_CAPABILITIES
        if not getattr(dependencies, attribute)
    ]
    missing_dependencies = [extra for _label, extra in missing_capabilities]
    extras = ",".join(missing_dependencies)

    return optional_dependency_recovery_state(
        unavailable_what=", ".join(label for label, _extra in missing_capabilities),
        missing_dependencies=tuple(missing_dependencies),
        install_target=f'pip install "tldw_chatbook[{extras}]"',
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

    dependencies = speech_local_dependency_availability()
    ready_count = sum(
        getattr(dependencies, attribute)
        for _label, attribute, _extra in _LOCAL_CAPABILITIES
    )
    return (
        "OpenAI-compatible speech: available when configured; "
        f"local capabilities: {ready_count}/{len(_LOCAL_CAPABILITIES)} ready"
    )


def _speech_capability_lines() -> tuple[str, ...]:
    """Return one exact status and recovery extra per local capability."""

    dependencies = speech_local_dependency_availability()
    lines = ["OpenAI-compatible speech: available when configured"]
    for label, attribute, extra in _LOCAL_CAPABILITIES:
        if getattr(dependencies, attribute):
            lines.append(f"{label}: ready")
        else:
            lines.append(
                f'{label}: missing - pip install "tldw_chatbook[{extra}]"'
            )
    return tuple(lines)


def speech_capability_detail() -> str:
    """Return independent remote and local capability facts for the inspector.

    The inspector keeps all exact install commands visible without crowding
    the rail's one-line summary or hiding recovery behind pointer interaction.

    Returns:
        One remote availability line followed by all four local capabilities.
    """
    return "\n".join(_speech_capability_lines())


def speech_capability_tooltip() -> str:
    """Return install guidance for the capability chip's tooltip.

    Returns:
        The same capability-specific guidance in a compact single line.
    """
    return " ".join(_speech_capability_lines())
