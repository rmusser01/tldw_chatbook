"""One-shot Console microphone dictation service contracts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import Mock

import pytest

from tldw_chatbook.Audio.console_dictation import (
    CONSOLE_DICTATION_MAX_BYTES,
    CONSOLE_DICTATION_MAX_SECONDS,
    ConsoleDictationError,
    ConsoleDictationSession,
)


REQUIRED_MODEL_FILES = (
    "config.json",
    "vocab.txt",
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
)


def test_console_dictation_import_keeps_legacy_transcription_stack_lazy():
    repo_root = Path(__file__).resolve().parents[2]
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import types

                sys.modules["parakeet_mlx"] = types.ModuleType("parakeet_mlx")
                import tldw_chatbook.Audio.console_dictation  # noqa: F401

                assert (
                    "tldw_chatbook.Local_Ingestion.transcription_service"
                    not in sys.modules
                )
                """
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr


def _model_dir(tmp_path):
    model_dir = tmp_path / "parakeet-v2-int8"
    model_dir.mkdir()
    for filename in REQUIRED_MODEL_FILES:
        (model_dir / filename).touch()
    return model_dir


class FakeRecorder:
    def __init__(self, *, start_result=True, audio=b"\0\0" * 160, **kwargs):
        self.start_result = start_result
        self.audio = audio
        self.kwargs = kwargs
        self.start_recording = Mock(return_value=start_result)
        self.stop_recording = Mock(return_value=audio)


def test_session_records_once_and_transcribes_explicit_english_v2_int8(tmp_path):
    model_dir = _model_dir(tmp_path)
    recorders = []

    def recorder_factory(**kwargs):
        recorder = FakeRecorder(**kwargs)
        recorders.append(recorder)
        return recorder

    transcriber = Mock()
    transcriber.transcribe_buffer.return_value = {"text": "  hello Console  "}
    on_buffer_limit = Mock()
    session = ConsoleDictationSession(
        model_dir=model_dir,
        recorder_factory=recorder_factory,
        transcription_service=transcriber,
    )

    session.start(on_buffer_limit=on_buffer_limit)
    text = session.stop_and_transcribe()

    assert CONSOLE_DICTATION_MAX_SECONDS == 60.0
    assert recorders[0].kwargs == {
        "sample_rate": 16_000,
        "channels": 1,
        "use_vad": False,
        "max_buffer_bytes": CONSOLE_DICTATION_MAX_BYTES,
        "on_buffer_limit": on_buffer_limit,
    }
    transcriber.transcribe_buffer.assert_called_once_with(
        b"\0\0" * 160,
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        provider="parakeet-onnx",
        model="nemo-parakeet-tdt-0.6b-v2",
        language="en",
        model_dir=str(model_dir),
    )
    assert text == "hello Console"


def test_session_accepts_only_verified_default_library_install(tmp_path):
    installed_dir = _model_dir(tmp_path)
    verify = Mock(return_value=True)
    recorder = FakeRecorder()
    session = ConsoleDictationSession(
        installed_model_dir=installed_dir,
        verify_installed_bundle=verify,
        recorder_factory=lambda **kwargs: recorder,
        transcription_service=Mock(),
    )

    session.start()

    verify.assert_called_once_with(installed_dir)
    assert session.model_dir == installed_dir


def test_session_reports_missing_model_without_starting_microphone(tmp_path):
    recorder_factory = Mock()
    missing_dir = tmp_path / "missing"
    session = ConsoleDictationSession(
        installed_model_dir=missing_dir,
        verify_installed_bundle=lambda path: False,
        recorder_factory=recorder_factory,
        transcription_service=Mock(),
    )

    with pytest.raises(ConsoleDictationError, match="Parakeet v2 model"):
        session.start()

    recorder_factory.assert_not_called()


def test_session_reports_microphone_start_failure(tmp_path):
    session = ConsoleDictationSession(
        model_dir=_model_dir(tmp_path),
        recorder_factory=lambda **kwargs: FakeRecorder(start_result=False, **kwargs),
        transcription_service=Mock(),
    )

    with pytest.raises(ConsoleDictationError, match="microphone"):
        session.start()


def test_session_reports_empty_audio_without_transcribing(tmp_path):
    transcriber = Mock()
    session = ConsoleDictationSession(
        model_dir=_model_dir(tmp_path),
        recorder_factory=lambda **kwargs: FakeRecorder(audio=b"", **kwargs),
        transcription_service=transcriber,
    )
    session.start()

    with pytest.raises(ConsoleDictationError, match="No audio"):
        session.stop_and_transcribe()

    transcriber.transcribe_buffer.assert_not_called()


def test_session_discard_stops_capture_without_transcribing(tmp_path):
    recorder = FakeRecorder()
    transcriber = Mock()
    session = ConsoleDictationSession(
        model_dir=_model_dir(tmp_path),
        recorder_factory=lambda **kwargs: recorder,
        transcription_service=transcriber,
    )
    session.start()

    session.discard()

    recorder.stop_recording.assert_called_once_with()
    transcriber.transcribe_buffer.assert_not_called()
