"""`_transcribe_buffer_with_parakeet_mlx` must hand parakeet-mlx a file path.

The installed `parakeet_mlx` package's `transcribe()` takes a file path --
internally it does `Path(path)` then `load_audio(...)` -- it does not accept
a numpy array. Before this test's fix, `_transcribe_buffer_with_parakeet_mlx`
called `self._parakeet_mlx_model.transcribe(audio_array)` with a numpy
ndarray directly, which raises on the real package: "argument should be a
str or an os.PathLike object where __fspath__ returns a str, not 'ndarray'"
(reproduced live -- this is the dictation stop/tail path for parakeet-mlx,
so this failure ended every parakeet-mlx dictation capture).

Kept as its own file, separate from
`Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py`, and with the
`_NUMPY_INT16_CAST_WARMUP` call below: under pytest specifically (never
reproduced with a bare `python -c`), the *first* `numpy` int16 dtype-cast in
a process that has already imported `transcription_service` intermittently
raised `ImportError: cannot load module more than once per process` out of
`numpy._core.multiarray` -- observed on this machine while several other
concurrent pytest runs (other worktree sessions) were competing for the same
shared `.venv`. Exercising that exact cast once at collection time, before
any fixture touches `transcription_service`, reproduced-fixed it across
repeated runs under the same concurrent load; the assertions below are
otherwise a normal fake-loader unit test.
"""

from __future__ import annotations

import importlib
import os
import stat
import struct
import sys
import wave
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# See module docstring: forces the int16-cast code path this test's fix
# exercises to resolve once, up front, rather than for the first time inside
# a fixture/monkeypatched `transcription_service` call.
np.clip(np.round(np.zeros(1, dtype=np.float32)), -32768, 32767).astype(np.int16)

SERVICE_MODULE = "tldw_chatbook.Local_Ingestion.transcription_service"


@pytest.fixture(scope="module")
def service_module():
    """Import fresh, faking out the optional mlx/parakeet_mlx packages.

    Module-scoped, and installs the fake stub packages with a plain
    `sys.modules[...] = ...` assignment rather than `patch.dict`/
    `monkeypatch.setitem` -- both of those revert `sys.modules` to its exact
    pre-entry snapshot on exit, which deletes every module newly added
    during that window, not just the ones explicitly listed. Verified
    directly: importing a stdlib module inside
    `patch.dict(sys.modules, {...})` and checking membership immediately
    after the block exits shows it gone. `importlib.import_module
    (SERVICE_MODULE)` used to run *inside* that reverting block, so the
    revert took `SERVICE_MODULE` down with it -- and everything it
    transitively imports at module level (`tldw_chatbook.config`,
    `...DB.ChaChaNotes_DB`, `...Metrics.metrics_logger`, torch,
    ctranslate2, ...).

    This file previously had exactly one test using this fixture, so the
    corruption was invisible: nothing ever asked twice. Adding a second and
    third test (Finding 3, PR #1171 review) surfaced it two different ways
    depending on the exact fix tried, both confirming the same root cause:
    * function-scoped + reverting: the second invocation's
      `SERVICE_MODULE in sys.modules` fast path missed (wiped by the first
      invocation's own revert), forcing a real second `import torch` in
      -process, which crashed with `RuntimeError: function
      '_has_torch_function' already has a docstring` (torch registers
      docstrings on process-global C functions at import time; not
      idempotent against a second "logical" import).
    * module-scoped + still reverting: `SERVICE_MODULE` itself was cached
      correctly (module scope only runs this body once), but
      `tldw_chatbook.config` and friends -- newly imported for the first
      time in this process as a side effect of importing `SERVICE_MODULE`,
      via a route this fixture's revert did not know it needed to protect
      -- were still wiped, and a LATER, unrelated autouse fixture
      (`isolate_test_environment` in `Tests/conftest.py`) re-importing
      `tldw_chatbook.config` fresh crashed with `ValueError: Level 'METRIC'
      already exists` (loguru's custom level registration in
      `Metrics/metrics_logger.py` is not idempotent either).

    Not reverting at all -- this fixture's own `if SERVICE_MODULE in
    sys.modules` fast path already assumes `sys.modules` correctly and
    permanently caches the real import -- fixes both: nothing this fixture
    imports is ever removed again, so every module it pulls in for the
    first time (transitively or not) stays exactly as correctly cached as
    a normal, unpatched `import` would leave it. The two fake stub entries
    left behind are harmless: nothing else in the suite tries to use the
    real `lightning_whisper_mlx`/`parakeet_mlx` packages for real work.
    """
    if SERVICE_MODULE in sys.modules:
        return sys.modules[SERVICE_MODULE]
    lightning_module = ModuleType("lightning_whisper_mlx")
    lightning_module.LightningWhisperMLX = object()
    parakeet_module = ModuleType("parakeet_mlx")
    parakeet_module.from_pretrained = object()
    sys.modules["lightning_whisper_mlx"] = lightning_module
    sys.modules["parakeet_mlx"] = parakeet_module
    return importlib.import_module(SERVICE_MODULE)


def _service(service_module, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        service_module, "get_cli_setting", lambda _key, default=None: default
    )
    return service_module._LegacyTranscriptionBackend()


def _install_fake_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    mlx_module = ModuleType("mlx")
    core_module = ModuleType("mlx.core")
    core_module.float32 = object()
    core_module.float16 = object()
    core_module.bfloat16 = object()
    mlx_module.core = core_module
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", core_module)


class _FakeResult:
    text = "the exact pcm made it through"


def test_parakeet_buffer_transcription_writes_a_wav_and_passes_the_path(
    service_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the fix: write a temp 16 kHz mono 16-bit WAV, pass its *path*.

    The fake model's `transcribe()` stands in for the real package's
    path-only API: it rejects anything that is not a `str`/`PathLike`
    (reproducing the ndarray failure pre-fix) and reads the file back with
    `wave` to assert the WAV is well-formed 16 kHz/mono/16-bit and contains
    the *exact* input PCM samples, not a lossy re-encoding. Also asserts the
    temp file is cleaned up afterward -- the stop/tail path cannot leak a
    WAV per segment over a long dictation session.
    """
    # Varied samples (both extremes, negative, positive, zero) so a lossy
    # float32 round-trip through the array-preparation code above this call
    # would be caught, not just coincidentally survive an all-zero buffer.
    samples = (-32768, -1000, -1, 0, 1, 1234, 32767) * 20
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    recorded_paths: list[str] = []

    class _FakeParakeetModel:
        def transcribe(self, path):
            assert isinstance(path, (str, os.PathLike)), (
                f"parakeet_mlx.transcribe() requires a path, got {type(path)!r}"
            )
            path = str(path)
            recorded_paths.append(path)
            with wave.open(path, "rb") as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == 2
                assert wav_file.getframerate() == 16000
                written = wav_file.readframes(wav_file.getnframes())
            assert written == pcm
            return _FakeResult()

    service = _service(service_module, monkeypatch)
    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr(service_module, "PARAKEET_MLX_AVAILABLE", True)
    monkeypatch.setattr(
        service_module,
        "_ensure_parakeet_mlx_import",
        lambda: (lambda model, dtype=None: _FakeParakeetModel()),
    )

    result = service._transcribe_buffer_with_parakeet_mlx(
        pcm,
        sample_rate=16000,
        channels=1,
        sample_width=2,
        model=None,
        language=None,
    )

    assert result["text"] == "the exact pcm made it through"
    assert result["provider"] == "parakeet-mlx"
    assert len(recorded_paths) == 1
    # Cleaned up: the "must work" stop/tail path cannot leak a temp WAV per
    # segment over a long dictation session.
    assert not os.path.exists(recorded_paths[0])


# --------------------------------------------------------------------------
# Review Finding 3, PR #1171: the temp WAV must be identifiable, private
# while it exists, and cleaned up even when transcription itself fails.
# --------------------------------------------------------------------------


def test_parakeet_buffer_temp_file_has_an_identifying_prefix_and_owner_only_perms(
    service_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`prefix="parakeet_mlx_"` and best-effort `0o600` before any audio is written.

    A crash between file creation and the `finally` cleanup below leaves raw
    microphone audio on disk; the prefix makes that leftover recognizable in
    a temp-dir listing (a future stale-file sweep is out of scope for this
    fix, but needs a name to look for), and owner-only permissions keep that
    audio from being group/world-readable for however long it survives.
    Both must be true *before* cleanup runs, so this reads them from inside
    the fake model's `transcribe()` -- the only point with a live path.
    """
    pcm = struct.pack("<4h", 0, 1, -1, 32767)
    recorded_paths: list[str] = []
    captured_modes: list[int] = []

    class _FakeParakeetModel:
        def transcribe(self, path):
            path = str(path)
            recorded_paths.append(path)
            if os.name == "posix":
                captured_modes.append(stat.S_IMODE(os.stat(path).st_mode))
            return _FakeResult()

    service = _service(service_module, monkeypatch)
    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr(service_module, "PARAKEET_MLX_AVAILABLE", True)
    monkeypatch.setattr(
        service_module,
        "_ensure_parakeet_mlx_import",
        lambda: (lambda model, dtype=None: _FakeParakeetModel()),
    )

    service._transcribe_buffer_with_parakeet_mlx(
        pcm,
        sample_rate=16000,
        channels=1,
        sample_width=2,
        model=None,
        language=None,
    )

    assert len(recorded_paths) == 1
    basename = os.path.basename(recorded_paths[0])
    assert basename.startswith("parakeet_mlx_"), (
        f"expected an identifying prefix on the temp WAV, got {basename!r}"
    )
    if os.name == "posix":
        assert captured_modes[0] == 0o600, (
            f"expected owner-only permissions (0o600) on the temp WAV "
            f"while it existed, got {oct(captured_modes[0])}"
        )
    # Still cleaned up on the success path.
    assert not os.path.exists(recorded_paths[0])


def test_parakeet_buffer_temp_file_is_removed_even_when_transcribe_raises(
    service_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `finally` cleanup path must run when `transcribe()` itself fails.

    Before this fix the file was created, closed, and reopened by path to
    write the audio -- a crash anywhere in that window (or afterward, in
    `transcribe()`) still relied on the same `finally`/`os.unlink`, but this
    pins that the rewritten write-through-the-same-handle path preserved
    that guarantee rather than accidentally dropping it.
    """
    pcm = struct.pack("<4h", 0, 1, -1, 32767)
    recorded_paths: list[str] = []

    class _FakeParakeetModel:
        def transcribe(self, path):
            recorded_paths.append(str(path))
            raise RuntimeError("simulated transcription failure")

    service = _service(service_module, monkeypatch)
    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr(service_module, "PARAKEET_MLX_AVAILABLE", True)
    monkeypatch.setattr(
        service_module,
        "_ensure_parakeet_mlx_import",
        lambda: (lambda model, dtype=None: _FakeParakeetModel()),
    )

    with pytest.raises(service_module.TranscriptionError):
        service._transcribe_buffer_with_parakeet_mlx(
            pcm,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            model=None,
            language=None,
        )

    assert len(recorded_paths) == 1
    assert not os.path.exists(recorded_paths[0]), (
        "the temp WAV survived a raise from transcribe() -- the finally "
        "cleanup did not run (or ran against the wrong path)"
    )
