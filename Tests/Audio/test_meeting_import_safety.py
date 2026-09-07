"""`meeting_owner`'s import graph must not need numpy (final review, C1).

`app.py` imports `build_meeting_session_owner` at MODULE SCOPE, and
`meeting_owner.py` used to do a module-scope `from .meeting_capture import
MeetingCapture`. `meeting_capture.py` opens with a bare `import numpy as
np` -- the only unguarded numpy import in the package -- and numpy ships
only in optional extras. The result was not "meetings are unavailable" but
"the application does not start" for every install without an extra that
happens to pull numpy:

    File "tldw_chatbook/app.py", line 546, in <module>
      from .Audio.meeting_owner import build_meeting_session_owner
    File "tldw_chatbook/Audio/meeting_owner.py", line 20, in <module>
      from .meeting_capture import MeetingCapture
    File "tldw_chatbook/Audio/meeting_capture.py", line 18, in <module>
      import numpy as np
  ImportError: No module named numpy

Fixed by moving that import inside `MeetingSessionOwner.start()` (the
module has `from __future__ import annotations`, so the type hints on
`_default_dictation_factory` and `dictation_factory` keep working) --
nothing pays for numpy until a meeting actually starts, and a numpy-less
install reaches the screen and reports the missing recorder in
`PrepareResult.capture_error` instead of crashing at boot.

Both probes run in a FRESH subprocess: this suite imports numpy for real
many times over (see `Tests/Audio/test_meeting_capture.py`), so an
in-process `sys.modules` check would be meaningless. Modelled on
`Tests/Audio/test_audio_init_lazy_import_safety.py`.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit

_BLOCK_NUMPY = textwrap.dedent("""
    import sys
    import importlib.abc


    class _NoNumpyFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "numpy" or name.startswith("numpy."):
                raise ImportError("No module named numpy")
            return None


    sys.meta_path.insert(0, _NoNumpyFinder())
""")


def _run_probe(script: str) -> subprocess.CompletedProcess:
    """Run `script` in a fresh interpreter using the SAME venv as pytest."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_app_import_survives_a_missing_numpy():
    """The exact reproduction, in the suite: no numpy must degrade to
    "meetings cannot record", never "the application does not start"."""
    script = _BLOCK_NUMPY + textwrap.dedent("""
        import tldw_chatbook.app  # noqa: F401
        print("RESULT: APP IMPORT SUCCEEDED")
    """)
    result = _run_probe(script)
    assert "RESULT: APP IMPORT SUCCEEDED" in result.stdout, (
        f"app import failed with numpy blocked (exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_meeting_owner_imports_without_numpy_and_leaves_the_mixer_unloaded():
    """The owner module itself must import numpy-free, and must not have
    pulled `meeting_capture` (the numpy-dependent mixer) along with it."""
    script = _BLOCK_NUMPY + textwrap.dedent("""
        import sys
        import tldw_chatbook.Audio.meeting_owner  # noqa: F401

        pulled = "tldw_chatbook.Audio.meeting_capture" in sys.modules
        print(f"RESULT: OWNER IMPORTED, capture_pulled={pulled}")
    """)
    result = _run_probe(script)
    assert "RESULT: OWNER IMPORTED, capture_pulled=False" in result.stdout, (
        f"meeting_owner did not import cleanly without numpy "
        f"(exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_app_import_pulls_in_no_diarizer_module():
    """Boot must never import the diarizer backends, or torch (spec §3.4,
    §7): `build_diarizer()` (`meeting_owner.py`) imports `SpeechBrainDiarizer`
    from `diarizer_local` LAZILY, only when a meeting actually starts with
    `live_diarization` on, and `diarizer_local` itself only spawns
    `diarizer_worker.py` as a SEPARATE subprocess (never imports it) --
    `torch`/`speechbrain` therefore only ever load in that child process, not
    in the TUI. Run with numpy blocked too, matching the two probes above:
    app import must survive with neither numpy nor the diarizer/torch stack
    present.
    """
    script = _BLOCK_NUMPY + textwrap.dedent("""
        import sys
        import tldw_chatbook.app  # noqa: F401

        watched = (
            "tldw_chatbook.Audio.diarizer_local",
            "tldw_chatbook.Audio.diarizer_worker",
            "torch",
        )
        pulled = sorted(name for name in watched if name in sys.modules)
        print(f"RESULT: PULLED={pulled}")
    """)
    result = _run_probe(script)
    assert "RESULT: PULLED=[]" in result.stdout, (
        f"app import pulled in a diarizer module or torch at boot "
        f"(exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
