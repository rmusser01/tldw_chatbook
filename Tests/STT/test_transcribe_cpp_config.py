from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.STT import transcribe_cpp_config


def test_configure_model_path_admits_then_writes_only_dedicated_key(
    tmp_path, monkeypatch
):
    selected = tmp_path / "private-model.gguf"
    admitted = selected.absolute()
    calls: list[object] = []

    monkeypatch.setattr(
        transcribe_cpp_config,
        "validate_local_gguf",
        lambda path: calls.append(("validate", path)) or SimpleNamespace(path=admitted),
    )
    monkeypatch.setattr(
        transcribe_cpp_config,
        "save_settings_to_cli_config",
        lambda values: calls.append(("save", values)) or True,
    )

    transcribe_cpp_config.configure_model_path(selected)

    assert calls == [
        ("validate", selected),
        (
            "save",
            {"transcription.transcribe_cpp": {"model_path": str(admitted)}},
        ),
    ]


def test_configure_model_path_does_not_persist_failed_admission(tmp_path, monkeypatch):
    selected = tmp_path / "private-model.gguf"

    def save(_values):
        pytest.fail("invalid GGUF must not be persisted")

    monkeypatch.setattr(
        transcribe_cpp_config,
        "validate_local_gguf",
        lambda _path: (_ for _ in ()).throw(ValueError("invalid model")),
    )
    monkeypatch.setattr(transcribe_cpp_config, "save_settings_to_cli_config", save)

    with pytest.raises(ValueError, match="invalid model"):
        transcribe_cpp_config.configure_model_path(selected)


def test_configure_model_path_uses_path_safe_persistence_failure(tmp_path, monkeypatch):
    selected = tmp_path / "private-model.gguf"
    monkeypatch.setattr(
        transcribe_cpp_config,
        "validate_local_gguf",
        lambda _path: SimpleNamespace(path=selected.absolute()),
    )
    monkeypatch.setattr(
        transcribe_cpp_config, "save_settings_to_cli_config", lambda _values: False
    )

    with pytest.raises(transcribe_cpp_config.TranscribeCppConfigError) as exc_info:
        transcribe_cpp_config.configure_model_path(selected)

    assert str(selected) not in str(exc_info.value)
    assert str(selected) not in repr(exc_info.value)


def test_gguf_filter_accepts_only_gguf_suffix():
    assert transcribe_cpp_config.is_gguf_file(Path("model.GGUF"))
    assert not transcribe_cpp_config.is_gguf_file(Path("model.bin"))
