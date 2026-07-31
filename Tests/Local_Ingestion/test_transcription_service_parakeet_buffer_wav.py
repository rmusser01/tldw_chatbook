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
import struct
import sys
import wave
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# See module docstring: forces the int16-cast code path this test's fix
# exercises to resolve once, up front, rather than for the first time inside
# a fixture/monkeypatched `transcription_service` call.
np.clip(np.round(np.zeros(1, dtype=np.float32)), -32768, 32767).astype(np.int16)

SERVICE_MODULE = "tldw_chatbook.Local_Ingestion.transcription_service"


@pytest.fixture()
def service_module():
    """Import fresh, faking out the optional mlx/parakeet_mlx packages."""
    if SERVICE_MODULE in sys.modules:
        return sys.modules[SERVICE_MODULE]
    lightning_module = ModuleType("lightning_whisper_mlx")
    lightning_module.LightningWhisperMLX = object()
    parakeet_module = ModuleType("parakeet_mlx")
    parakeet_module.from_pretrained = object()
    with patch.dict(
        sys.modules,
        {
            "lightning_whisper_mlx": lightning_module,
            "parakeet_mlx": parakeet_module,
        },
    ):
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
