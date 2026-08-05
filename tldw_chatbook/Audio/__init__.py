# Audio module for speech recording and live dictation
"""
Audio recording and dictation functionality for tldw_chatbook.
Provides cross-platform audio capture and real-time transcription.

Final whole-branch review (streaming-pcm-sink, C1): this package used to
eagerly `from .recording_service import AudioRecordingService,
AudioRecordingError` at module scope. `recording_service.py` imports
`sounddevice` at ITS module scope, and `sounddevice`'s own `_initialize()`
calls `Pa_Initialize()` at IMPORT TIME -- raising `PortAudioError` (not
`ImportError`) when PortAudio cannot initialize (headless container, no
ALSA, CoreAudio unavailable, audio server down). Because
`Event_Handlers/TTS_Events/tts_events.py` imports
`tldw_chatbook.Audio.streaming_sink` at ITS OWN module scope (which, being
a submodule of this package, always imports this `__init__.py` first per
ordinary Python package semantics), that uncaught `PortAudioError` used to
fail the import of `tldw_chatbook.app` itself, for any user with the
`speech_recording` extra installed on a machine where PortAudio can't
init. Failure mode: "voice features degrade" -> "the application does not
start". `AudioRecordingService`/`AudioRecordingError` are now lazy exports
too, via the same `__getattr__` pattern already used below for the
dictation stack -- so importing this package, or any submodule under it,
no longer transitively imports `recording_service`/`sounddevice` at all.
Only an explicit `from tldw_chatbook.Audio import AudioRecordingService`
(or importing `.recording_service` directly, as every real production
call site already does) triggers it, same as before for the dictation
exports.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "AudioRecordingService",
    "AudioRecordingError",
    "LiveDictationService",
    "DictationResult",
    "DictationState",
]

#: Maps each lazy export to the submodule (relative to this package) that
#: defines it. `__getattr__` below imports that submodule -- and only that
#: submodule -- the first time the name is actually requested.
_LAZY_EXPORTS = {
    "AudioRecordingService": ".recording_service",
    "AudioRecordingError": ".recording_service",
    "LiveDictationService": ".dictation_service",
    "DictationResult": ".dictation_service",
    "DictationState": ".dictation_service",
}


def __getattr__(name: str) -> Any:
    """Load a lazy export's owning submodule only when explicitly requested.

    Args:
        name: Package attribute requested by the caller.

    Returns:
        The lazily imported attribute.

    Raises:
        AttributeError: The requested name is not a supported lazy export.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
