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

#: The same trick for torch, so the ONNX probe below has a real negative
#: control on a machine where torch simply is not installed (this one): a
#: regression that imports it fails the probe by RAISING, not only by showing
#: up in `sys.modules` on some other machine.
_BLOCK_TORCH = textwrap.dedent("""
    import sys
    import importlib.abc


    class _NoTorchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("torch", "torchaudio", "speechbrain"):
                raise ImportError(f"No module named {name}")
            return None


    sys.meta_path.insert(0, _NoTorchFinder())
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
    pulled `meeting_capture` (the numpy-dependent mixer) or `Audio.voiceprint`
    (TASK-31826, spec §3.1/§7) along with it. `meeting_owner.py` only ever
    reaches `.voiceprint` from inside `_voiceprint_store()` (called from
    `enroll_from_mic`/`accept_learning`/`_load_voiceprint`, all lazy) --
    never at module scope -- because that module pulls the keyring backend,
    and boot must not."""
    script = _BLOCK_NUMPY + textwrap.dedent("""
        import sys
        import tldw_chatbook.Audio.meeting_owner  # noqa: F401

        capture_pulled = "tldw_chatbook.Audio.meeting_capture" in sys.modules
        voiceprint_pulled = "tldw_chatbook.Audio.voiceprint" in sys.modules
        engine_pulled = sorted(n for n in ("sherpa_onnx", "numpy") if n in sys.modules)
        print(f"RESULT: OWNER IMPORTED, capture_pulled={capture_pulled}, voiceprint_pulled={voiceprint_pulled}, engine_pulled={engine_pulled}")
    """)
    result = _run_probe(script)
    assert "RESULT: OWNER IMPORTED, capture_pulled=False, voiceprint_pulled=False, engine_pulled=[]" in result.stdout, (
        f"meeting_owner did not import cleanly without numpy, or pulled in "
        f"the mixer/voiceprint module at boot (exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_meetings_screen_imports_pull_in_no_voiceprint_module():
    """The Meetings screen module itself must not import `Audio.voiceprint`
    at module scope (TASK-31826): the screen only reaches it from inside
    `_build_store()`, called off the UI thread by the prepare worker, never
    while the module is being imported. Run in a fresh subprocess (import
    order across the suite would otherwise make this meaningless) with
    numpy blocked too, matching the probes above -- the screen module chain
    reaches `meeting_owner`, which must survive the same way."""
    script = _BLOCK_NUMPY + textwrap.dedent("""
        import sys
        import tldw_chatbook.UI.Screens.meetings_screen  # noqa: F401

        pulled = "tldw_chatbook.Audio.voiceprint" in sys.modules
        engine_pulled = sorted(n for n in ("sherpa_onnx", "numpy") if n in sys.modules)
        print(f"RESULT: SCREEN IMPORTED, voiceprint_pulled={pulled}, engine_pulled={engine_pulled}")
    """)
    result = _run_probe(script)
    assert "RESULT: SCREEN IMPORTED, voiceprint_pulled=False, engine_pulled=[]" in result.stdout, (
        f"importing meetings_screen pulled in Audio.voiceprint at module "
        f"scope (exit={result.returncode}):\n"
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
            "tldw_chatbook.Audio.diarizer_engine_onnx",
            "torch",
            "sherpa_onnx",
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


def test_loading_the_onnx_engine_never_pulls_in_torch(tmp_path):
    """The whole point of the ONNX engine (spec §2/§7): a base install with no
    torch must be able to run live speaker labels.

    The other probes on this page prove the APP process stays torch-free; this
    one proves the WORKER process does too, which is where torch would
    otherwise load. A fresh subprocess with a fake `sherpa_onnx` runs the real
    `diarizer_engine_onnx.load()` -- the same call `diarizer_worker.main()`
    makes for `--engine onnx` -- against empty model files with hash
    verification off, and reports what got imported. numpy is expected (the
    engine's own dependency); torch, torchaudio and speechbrain are not.
    """
    script = _BLOCK_TORCH + textwrap.dedent(f"""
        import sys
        import types
        from pathlib import Path

        class _Extractor:
            def __init__(self, config): self.config = config

        sys.modules["sherpa_onnx"] = types.SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=lambda **kw: kw,
            SpeakerEmbeddingExtractor=_Extractor,
        )

        from tldw_chatbook.Audio import diarizer_engine_onnx as eng
        from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

        models = Path({str(tmp_path)!r})
        (models / eng.SEGMENTATION.file_name).write_bytes(b"")
        (models / eng.EMBEDDERS[eng.DEFAULT_EMBEDDER].file_name).write_bytes(b"")

        loaded = eng.load(OnlineClusterer(), 8, models_dir_override=models, verify_hashes=False)
        torch_like = sorted(
            n for n in ("torch", "torchaudio", "speechbrain") if n in sys.modules
        )
        print(f"RESULT: LOADED={{loaded.model_id.startswith('sherpa-onnx/')}}, TORCH={{torch_like}}")
    """)
    result = _run_probe(script)
    assert "RESULT: LOADED=True, TORCH=[]" in result.stdout, (
        f"loading the ONNX engine pulled in torch (exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
