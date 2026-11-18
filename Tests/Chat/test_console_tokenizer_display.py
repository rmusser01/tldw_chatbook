"""Tokenizer preparation qualification must not invoke custom callbacks."""

import asyncio
import json

import pytest

from tldw_chatbook.UI.Console_Modules import tokenizer_display as display
from tldw_chatbook.Utils import custom_tokenizers as custom
from tldw_chatbook.Utils import token_counter as counter

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_cold_manager_discovering_heuristic_mapping_returns_to_direct_path(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    directory = tmp_path / ".config" / "tldw_cli" / "tokenizers"
    directory.mkdir(parents=True)
    (directory / "mappings.json").write_text(json.dumps({"other-model": "fixture"}))
    monkeypatch.setattr(custom, "_tokenizer_manager", None)
    request = display.capture_tokenizer_preparation("new-model", "OpenAI")
    assert request is not None
    assert await asyncio.to_thread(request) is False
    assert not display.preparation_current(request)
    assert display.capture_tokenizer_preparation("new-model", "OpenAI") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("file_exists", [True, False])
async def test_original_exact_custom_mapping_is_prepared_or_stays_unavailable(
    monkeypatch, tmp_path, file_exists
):
    from tokenizers import Tokenizer, models, pre_tokenizers

    manager = await asyncio.to_thread(
        custom.CustomTokenizerManager, str(tmp_path / "tokenizers")
    )
    monkeypatch.setattr(custom, "_tokenizer_manager", manager)
    if file_exists:
        tokenizer = Tokenizer(
            models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(tmp_path / "tokenizers" / "fixture.json"))
    await asyncio.to_thread(manager.add_mapping, "fixture-model", "fixture")
    request = display.capture_tokenizer_preparation("fixture-model", "OpenAI")
    assert request is not None
    assert await asyncio.to_thread(request) is file_exists
    assert display.preparation_current(request)
    if file_exists:
        assert manager.count_tokens("hello hello", "fixture-model", "openai") == 2
        assert "fixture" in manager._tokenizers


def plain_manager(monkeypatch, *, mappings=None):
    manager = object.__new__(custom.CustomTokenizerManager)
    manager.tokenizers_dir = "unused-tokenizer-fixture"
    manager._model_mappings = mappings or {}
    manager._tokenizers = {}
    monkeypatch.setattr(custom, "_tokenizer_manager", manager)
    return manager


def test_character_tier_preparation_needs_no_backend(monkeypatch):
    monkeypatch.setattr(counter, "CUSTOM_TOKENIZERS_AVAILABLE", False)
    monkeypatch.setattr(counter, "TIKTOKEN_AVAILABLE", False)
    preparation = display.capture_tokenizer_preparation("model", "OpenAI")
    assert preparation is not None
    assert preparation.provider == "OpenAI"
    assert display.preparation_current(preparation)
    assert preparation() is True
    assert display.preparation_current(preparation)


def test_rejects_custom_counter_without_calling_it(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("custom counter invoked during qualification")

    monkeypatch.setattr(counter, "estimate_tokens", forbidden)
    assert display.capture_tokenizer_preparation("model", "provider") is None


def test_custom_manager_is_not_inspected_or_called(monkeypatch):
    class Foreign:
        def __bool__(self):
            raise AssertionError("custom manager truthiness invoked")

    monkeypatch.setattr(counter, "CUSTOM_TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(custom, "TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(custom, "_tokenizer_manager", Foreign())
    assert display.capture_tokenizer_preparation("model", "provider") is None


def test_mapping_change_invalidates_captured_preparation(monkeypatch):
    monkeypatch.setattr(counter, "CUSTOM_TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(custom, "TOKENIZERS_AVAILABLE", True)
    manager = plain_manager(monkeypatch, mappings={"model": "first"})
    preparation = display.capture_tokenizer_preparation("model", "provider")
    assert preparation is not None
    manager._model_mappings["model"] = "second"
    assert not display.preparation_current(preparation)
    assert preparation() is False


def test_instance_loader_override_declines_without_calling(monkeypatch):
    monkeypatch.setattr(counter, "CUSTOM_TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(custom, "TOKENIZERS_AVAILABLE", True)
    manager = plain_manager(monkeypatch)
    manager.load_tokenizer = lambda name: pytest.fail("custom loader invoked")
    assert display.capture_tokenizer_preparation("model", "provider") is None


def test_no_mappings_and_no_tiktoken_finishes_without_native_io(monkeypatch):
    monkeypatch.setattr(counter, "CUSTOM_TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(custom, "TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(counter, "TIKTOKEN_AVAILABLE", False)
    manager = plain_manager(monkeypatch)
    preparation = display.capture_tokenizer_preparation("model", "provider")
    assert preparation is not None and preparation() is True
    assert display.preparation_current(preparation)
    manager._model_mappings["model"] = "new"
    assert not display.preparation_current(preparation)


def test_manager_replacement_invalidates_ready_preparation(monkeypatch):
    monkeypatch.setattr(counter, "CUSTOM_TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(custom, "TOKENIZERS_AVAILABLE", True)
    monkeypatch.setattr(counter, "TIKTOKEN_AVAILABLE", False)
    plain_manager(monkeypatch)
    preparation = display.capture_tokenizer_preparation("model", "provider")
    assert preparation is not None and preparation() is True
    plain_manager(monkeypatch)
    assert not display.preparation_current(preparation)


def test_changed_original_function_body_invalidates(monkeypatch):
    preparation = display.capture_tokenizer_preparation("model", "provider")
    assert preparation is not None

    def replacement(text, model="gpt-3.5-turbo", provider=""):
        return 1

    monkeypatch.setattr(counter.estimate_tokens, "__code__", replacement.__code__)
    assert not display.preparation_current(preparation)


def test_tiktoken_model_mapping_change_invalidates_preparation(monkeypatch):
    monkeypatch.setattr(counter, "CUSTOM_TOKENIZERS_AVAILABLE", False)
    preparation = display.capture_tokenizer_preparation("model", "provider")
    assert preparation is not None
    monkeypatch.setitem(counter.TIKTOKEN_MODEL_ENCODINGS, "model", "changed-encoding")
    assert not display.preparation_current(preparation)
    assert preparation() is False


def test_foreign_tiktoken_mapping_declines_without_calling(monkeypatch):
    class Foreign(dict):
        def items(self):
            pytest.fail("custom mapping inspected")

    monkeypatch.setattr(counter, "TIKTOKEN_MODEL_ENCODINGS", Foreign())
    assert display.capture_tokenizer_preparation("model", "provider") is None
