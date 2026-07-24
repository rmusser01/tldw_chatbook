from __future__ import annotations

import importlib
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import pytest

from tldw_chatbook.TTS import TTS_Backends
from tldw_chatbook.TTS.TTS_Backends import BackendRegistry, TTSBackendManager
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend


EXPECTED_LEGACY_IDS = {
    "openai_official_*",
    "local_kokoro_*",
    "elevenlabs_*",
    "local_chatterbox_*",
    "alltalk_*",
    "local_higgs_*",
}


@pytest.fixture(autouse=True)
def reset_legacy_registry_state() -> Iterator[None]:
    BackendRegistry._registry.clear()
    BackendRegistry._builtins_loaded = False
    yield
    BackendRegistry._registry.clear()
    BackendRegistry._builtins_loaded = False


def test_legacy_registry_is_closed_to_new_providers() -> None:
    BackendRegistry.ensure_builtins()

    with pytest.raises(RuntimeError, match="sealed legacy registry"):
        BackendRegistry.register("new_provider_*", object)  # type: ignore[arg-type]


def test_legacy_registry_has_exact_routes_and_manager_lookup() -> None:
    manager = TTSBackendManager({})
    first = tuple(BackendRegistry.ensure_builtins())
    second = tuple(BackendRegistry.ensure_builtins())

    assert first == second
    assert len(first) == len(set(first))
    assert set(first) == EXPECTED_LEGACY_IDS
    assert set(manager.list_available_backends()) == EXPECTED_LEGACY_IDS
    assert BackendRegistry.get("openai_official_tts-1") is OpenAITTSBackend


def test_concurrent_manager_construction_loads_builtins_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls = 0
    original_load = BackendRegistry._load_builtin_classes

    def counted_load(cls: type[BackendRegistry]) -> None:
        nonlocal load_calls
        load_calls += 1
        time.sleep(0.01)
        original_load()

    monkeypatch.setattr(
        BackendRegistry,
        "_load_builtin_classes",
        classmethod(counted_load),
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        managers = list(pool.map(lambda _: TTSBackendManager({}), range(16)))

    assert len(managers) == 16
    assert load_calls == 1
    assert set(BackendRegistry.list_backends()) == EXPECTED_LEGACY_IDS


def test_legacy_registry_exposes_no_test_only_reset_hook() -> None:
    assert not hasattr(BackendRegistry, "_reset_for_tests")


def test_missing_builtin_class_does_not_abort_remaining_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_module = "tldw_chatbook.TTS.backends.openai"
    original_import_module = importlib.import_module
    warnings: list[tuple[str, tuple[object, ...]]] = []

    def import_module(module_name: str) -> ModuleType:
        if module_name == missing_module:
            return ModuleType(module_name)
        return original_import_module(module_name)

    monkeypatch.setattr(TTS_Backends.importlib, "import_module", import_module)
    monkeypatch.setattr(
        TTS_Backends.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    loaded = set(BackendRegistry.ensure_builtins())

    assert loaded == EXPECTED_LEGACY_IDS - {"openai_official_*"}
    assert (
        "Legacy TTS backend is unavailable: {}",
        ("openai_official_*",),
    ) in warnings


def test_builtin_import_attribute_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_module = "tldw_chatbook.TTS.backends.openai"
    original_import_module = importlib.import_module

    def import_module(module_name: str) -> ModuleType:
        if module_name == missing_module:
            raise AttributeError("backend import bug")
        return original_import_module(module_name)

    monkeypatch.setattr(TTS_Backends.importlib, "import_module", import_module)

    with pytest.raises(AttributeError, match="backend import bug"):
        BackendRegistry.ensure_builtins()
