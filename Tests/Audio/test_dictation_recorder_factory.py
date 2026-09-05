"""Task 3: the lazy dictation service accepts an injected recorder factory."""
from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _stub_settings(monkeypatch) -> None:
    from tldw_chatbook.Audio import dictation_service_lazy

    values = {
        "dictation.buffer_duration_ms": 10,
        "dictation.privacy.save_history": False,
        "dictation.privacy.encrypt_history": True,
        "dictation.privacy.local_only": False,
        "dictation.privacy.auto_clear_buffer": True,
    }

    def _get(section: str, key: Any = None, default: Any = None) -> Any:
        if key is not None and not isinstance(key, str):
            key, default = None, key
        path = section if key is None else f"{section}.{key}"
        return values.get(path, default)

    monkeypatch.setattr(dictation_service_lazy, "get_cli_setting", _get)


class _Recorder:
    sample_rate = 16000
    channels = 1


def test_recorder_factory_is_used_and_receives_recorder_kwargs(monkeypatch):
    _stub_settings(monkeypatch)
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    recorder = _Recorder()
    seen: dict[str, Any] = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return recorder

    service = LazyLiveDictationService(
        transcription_provider="faster-whisper",
        enable_commands=False,
        recorder_factory=factory,
    )

    assert service.audio_service is recorder
    assert seen["use_vad"] is True
    assert "chunk_size" in seen and "vad_preroll_ms" in seen
    assert service._audio_service is recorder  # cached, not rebuilt


def test_default_factory_is_the_real_recorder_class(monkeypatch):
    _stub_settings(monkeypatch)
    from tldw_chatbook.Audio import dictation_service_lazy
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService(enable_commands=False)
    assert service._recorder_factory is None
