"""`Audio/__init__.py`'s lazy-import safety (final whole-branch review, C1).

`tts_events.py`'s module-scope `from tldw_chatbook.Audio.streaming_sink
import ...` (streaming-pcm-sink Task 4) pulls `tldw_chatbook/Audio/
__init__.py`, which -- before this fix -- eagerly did `from
.recording_service import AudioRecordingService, AudioRecordingError`.
`recording_service.py` imports `sounddevice` at module scope, and
`sounddevice`'s own `_initialize()` calls `Pa_Initialize()` AT IMPORT TIME,
raising `PortAudioError` (NOT `ImportError`) when PortAudio cannot
initialize (headless container, no ALSA, CoreAudio unavailable, audio
server down). `recording_service.py`'s probe only caught `ImportError`, so
that `PortAudioError` propagated all the way up, uncaught -- failing the
ENTIRE APP's import for any user with the `speech_recording` extra
installed on a machine where PortAudio can't init. Failure-mode change:
"voice features degrade" -> "the application does not start."

Fixed by extending `Audio/__init__.py`'s existing `__getattr__` lazy-export
pattern (previously used only for the dictation stack) to also cover
`AudioRecordingService`/`AudioRecordingError`, so importing the `Audio`
package (or any submodule under it, including `streaming_sink`) no longer
transitively imports `recording_service`/`sounddevice` at all -- only an
explicit `from tldw_chatbook.Audio import AudioRecordingService` (or
`.recording_service` directly, as every real production call site already
does) does. Belt-and-braces: `recording_service.py`'s three backend probes
(pyaudio/sounddevice/webrtcvad) are widened from `except ImportError` to
`except Exception`, so even a caller that DOES eventually trigger the
import is protected against a non-`ImportError` init failure.

Both probes below run in a FRESH subprocess: the main test process has
already imported sounddevice/`Audio` submodules many times over by the
point any of these tests run (this repo's own suite exercises real
dictation/streaming-sink code elsewhere), making an in-process
`sys.modules` check meaningless either way.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

_PORTAUDIO_FAILURE_FINDER = textwrap.dedent("""
    import sys
    import importlib.abc
    import importlib.machinery


    class _PortAudioError(Exception):
        pass


    class _FailingLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise _PortAudioError(
                "Error initializing PortAudio [PaErrorCode -9986]"
            )


    class _FailingSounddeviceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "sounddevice":
                return importlib.machinery.ModuleSpec(name, _FailingLoader())
            return None


    sys.meta_path.insert(0, _FailingSounddeviceFinder())
""")


def _run_probe(script: str) -> subprocess.CompletedProcess:
    """Run `script` in a fresh interpreter using the SAME venv as pytest.

    `sys.executable` inside the pytest process is already this repo's
    `.venv/bin/python`, so the subprocess sees the identical dependency
    set -- no environment drift between the probe and the real suite.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_app_import_survives_a_portaudio_init_failure():
    """C1 pin: the reviewer's exact reproduction, in the suite. A
    `PortAudioError` raised from `import sounddevice` (simulating PortAudio
    failing to initialize) must not fail `import tldw_chatbook.app` -- it
    must degrade to "voice features unavailable", never "the application
    does not start".
    """
    script = _PORTAUDIO_FAILURE_FINDER + textwrap.dedent("""
        import tldw_chatbook.app  # noqa: F401
        print("RESULT: APP IMPORT SUCCEEDED")
    """)
    result = _run_probe(script)
    assert "RESULT: APP IMPORT SUCCEEDED" in result.stdout, (
        f"app import failed under an injected PortAudioError "
        f"(exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_streaming_sink_import_alone_pulls_no_audio_backend():
    """Import-graph pin: `Audio/streaming_sink.py` (Task 1) is deliberately
    designed to never import `sounddevice` at module scope, probing
    availability via `find_spec` instead -- but before this fix, merely
    importing ANY submodule under the `Audio` package (including this one)
    transitively pulled `recording_service`, and thus `sounddevice`/
    `pyaudio`/`webrtcvad`, via the package's own eager `__init__.py`.
    Confirms the fix: a fresh process that imports ONLY
    `tldw_chatbook.Audio.streaming_sink` never loads any of the three real
    audio backend packages.
    """
    script = textwrap.dedent("""
        import sys
        import tldw_chatbook.Audio.streaming_sink  # noqa: F401

        pulled = sorted(
            name for name in ("sounddevice", "pyaudio", "webrtcvad")
            if name in sys.modules
        )
        print(f"RESULT: PULLED={pulled!r}")
    """)
    result = _run_probe(script)
    assert "RESULT: PULLED=[]" in result.stdout, (
        f"importing streaming_sink alone pulled in real audio backend "
        f"modules (exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_audio_recording_service_still_resolves_via_the_package():
    """Existing call sites (`from tldw_chatbook.Audio import
    AudioRecordingService`, e.g. `Tests/Audio/test_audio_integration.py`,
    `test_property_based.py`) must still resolve to the real class,
    unchanged, once it is lazily exported instead of eagerly imported.
    """
    from tldw_chatbook.Audio import AudioRecordingService, AudioRecordingError
    from tldw_chatbook.Audio.recording_service import (
        AudioRecordingService as DirectAudioRecordingService,
        AudioRecordingError as DirectAudioRecordingError,
    )

    assert AudioRecordingService is DirectAudioRecordingService
    assert AudioRecordingError is DirectAudioRecordingError
