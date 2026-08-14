from __future__ import annotations

import io
import wave
from pathlib import Path

from tldw_chatbook.TTS import sample_audio_validation


def _complete_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)
    return output.getvalue()


def test_compressed_audio_uses_central_optional_dependency_loader(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def unavailable(module_name: str, feature_name: str):
        calls.append((module_name, feature_name))
        return None

    monkeypatch.setattr(sample_audio_validation, "get_safe_import", unavailable)

    assert (
        sample_audio_validation.compressed_audio_has_decodable_frame(
            b"not-a-frame",
            "mp3",
        )
        is False
    )
    assert calls == [("av", "av")]


def test_playable_audio_file_passes_shared_path_validation_before_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = tmp_path / "sample.wav"
    sample.write_bytes(_complete_wav())
    calls: list[tuple[Path, Path, bool, bool]] = []

    def reject_path(
        user_path,
        base_directory,
        *,
        redact_paths: bool,
        allow_hidden: bool,
    ):
        calls.append(
            (Path(user_path), Path(base_directory), redact_paths, allow_hidden)
        )
        raise ValueError("rejected by shared validator")

    monkeypatch.setattr(sample_audio_validation, "validate_path", reject_path)

    assert (
        sample_audio_validation.validate_playable_audio_file(
            sample,
            "wav",
            "audio/wav",
            {},
        )
        is None
    )
    assert calls == [(sample, sample.parent, True, True)]


def test_shared_path_validation_does_not_weaken_symlink_rejection(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.wav"
    target.write_bytes(_complete_wav())
    sample = tmp_path / "sample.wav"
    sample.symlink_to(target)

    assert (
        sample_audio_validation.validate_playable_audio_file(
            sample,
            "wav",
            "audio/wav",
            {},
        )
        is None
    )
