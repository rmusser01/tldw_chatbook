from __future__ import annotations

import pytest

from tldw_chatbook.TTS.TTS_Backends import BackendRegistry


def test_legacy_registry_is_closed_to_new_providers() -> None:
    BackendRegistry._reset_for_tests()
    BackendRegistry.ensure_builtins()

    with pytest.raises(RuntimeError, match="sealed legacy registry"):
        BackendRegistry.register("new_provider_*", object)  # type: ignore[arg-type]


def test_legacy_registry_reset_is_deterministic() -> None:
    BackendRegistry._reset_for_tests()
    first = tuple(BackendRegistry.ensure_builtins())
    second = tuple(BackendRegistry.ensure_builtins())

    assert first == second
    assert len(first) == len(set(first))
    assert set(first) <= {
        "openai_official_*",
        "local_kokoro_*",
        "elevenlabs_*",
        "local_chatterbox_*",
        "alltalk_*",
        "local_higgs_*",
    }
