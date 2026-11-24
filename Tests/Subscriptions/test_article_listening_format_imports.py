"""Article storage can validate WAV bytes without starting audio services."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.unit


def test_article_audio_format_helpers_run_without_runtime_dependencies():
    script = dedent(r"""
        import ctypes
        import io
        import sys
        import wave
        from importlib.abc import MetaPathFinder

        blocked = (
            "numpy", "sounddevice", "textual", "torch",
            "tldw_chatbook.app", "tldw_chatbook.Backup_Recovery",
            "tldw_chatbook.TTS.TTS_Generation",
            "tldw_chatbook.TTS.adapter_bootstrap",
            "tldw_chatbook.TTS.audio_cpp_supervisor",
            "tldw_chatbook.TTS.audio_output_ownership",
            "tldw_chatbook.TTS.controlled_wav_playback",
        )

        class BlockRuntimeImports(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                for name in blocked:
                    if fullname == name or fullname.startswith(name + "."):
                        raise AssertionError(f"unexpected import: {fullname}")
                return None

        def deny_native_load(*args, **kwargs):
            raise AssertionError("unexpected Windows DLL load")

        if hasattr(ctypes, "WinDLL"):
            ctypes.WinDLL = deny_native_load
        sys.meta_path.insert(0, BlockRuntimeImports())

        from tldw_chatbook.TTS.adapter_types import TTSOperationError
        from tldw_chatbook.TTS.audio_limits import check_buffered_audio_size
        from tldw_chatbook.TTS.controlled_wav_profile import (
            ControlledWavError,
            validate_controlled_wav,
        )

        stream = io.BytesIO()
        with wave.open(stream, "wb") as writer:
            writer.setparams((2, 2, 48_000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\x01\x00\x01\x00" + bytes(12_000 * 4))
        validate_controlled_wav(stream, 12_001)

        damaged = io.BytesIO(stream.getvalue()[:-1] + b"\x01")
        try:
            validate_controlled_wav(damaged, 12_001)
        except ControlledWavError as error:
            assert str(error) == "invalid_retained_wave"
        else:
            raise AssertionError("nonzero padding was accepted")

        limit = 64 * 1024 * 1024
        check_buffered_audio_size(limit)
        try:
            check_buffered_audio_size(limit + 1)
        except TTSOperationError as error:
            assert error.code == "request_invalid"
        else:
            raise AssertionError("oversized audio was accepted")
        """)
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
