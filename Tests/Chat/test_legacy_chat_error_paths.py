"""TASK-38602: legacy chat error paths must raise, not reply with errors.

``Chat.Chat_Functions.chat`` previously stringified every failure into
``f"An error occurred in the chat function: {str(e)}"`` and returned it AS
the assistant reply -- callers could not distinguish failure from content,
and raw exception text flowed into chat content. These tests pin the new
contract: Chat*Error family failures re-raise unchanged and every other
exception propagates as-is.

``Chat.document_generator.DocumentGenerator`` previously dispatched through
a hand-maintained ``provider_functions`` dict that rejected every provider
registered after it was written (moonshot, zai, qwencloud, hosted presets)
with ``ChatAPIError("Unknown provider: ...")``. It now routes through
``chat_api_call`` so every registered provider name dispatches.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat import Chat_Functions as cf_module
from tldw_chatbook.Chat import document_generator as dg_module
from tldw_chatbook.Chat.Chat_Deps import ChatAPIError, ChatProviderError
from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS, chat
from tldw_chatbook.Chat.document_generator import DocumentGenerator
from tldw_chatbook.Event_Handlers import worker_events

pytestmark = [pytest.mark.unit]

_ERROR_REPLY_PREFIX = "An error occurred in the chat function"


def _chat_kwargs(**overrides):
    """Minimal non-streaming kwargs that reach the chat_api_call seam."""
    kwargs = dict(
        message="hello",
        history=[],
        media_content=None,
        selected_parts=[],
        api_endpoint="openai",
        api_key="test-key",
        custom_prompt=None,
        temperature=0.7,
        streaming=False,
    )
    kwargs.update(overrides)
    return kwargs


# --- (a) chat() raises instead of returning error-text replies ---------------


@pytest.mark.parametrize(
    ("raised", "expected_type"),
    [
        (ChatAPIError("provider exploded"), ChatAPIError),
        (ChatProviderError("upstream 502"), ChatProviderError),
    ],
)
def test_chat_re_raises_chat_error_family_from_handler(
    monkeypatch, raised, expected_type
):
    """A handler raising a Chat*Error surfaces as that exception, never as
    an error-text reply string (previously the terminal except returned
    ``f"An error occurred in the chat function: ..."``)."""
    sentinel = raised

    def failing_handler(**kwargs):
        raise sentinel

    monkeypatch.setitem(API_CALL_HANDLERS, "openai", failing_handler)
    with pytest.raises(expected_type) as excinfo:
        chat(**_chat_kwargs())

    assert isinstance(excinfo.value, expected_type)
    assert not str(excinfo.value).startswith(_ERROR_REPLY_PREFIX)


def test_chat_propagates_non_chat_exceptions(monkeypatch):
    """Non-Chat exceptions also propagate (binding resolution for the
    ambiguity over wrap-vs-propagate): chat() never converts any failure
    into a reply string."""
    def failing_call(**kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(cf_module, "chat_api_call", failing_call)
    with pytest.raises(ValueError, match="boom"):
        chat(**_chat_kwargs())


def test_chat_wrapper_surfaces_chat_errors(monkeypatch):
    """worker_events.chat_wrapper_function (the retained non-Console
    adapter and chat()'s only in-repo caller) propagates a raised
    ChatAPIError to its caller instead of crashing or swallowing it."""
    sentinel = ChatAPIError("provider down")

    def raising_core(**kwargs):
        raise sentinel

    monkeypatch.setattr(worker_events, "core_chat_function", raising_core)
    with pytest.raises(ChatAPIError) as excinfo:
        worker_events.chat_wrapper_function(
            None, message="hi", streaming=False, strip_thinking_tags=True
        )
    assert excinfo.value is sentinel


def test_chat_wrapper_success_path_unchanged(monkeypatch):
    expected = object()
    monkeypatch.setattr(worker_events, "core_chat_function", lambda **kwargs: expected)
    assert (
        worker_events.chat_wrapper_function(None, message="hi", streaming=False)
        is expected
    )


# --- (b) DocumentGenerator dispatches through chat_api_call ------------------

_PROVIDERS_ADDED_AFTER_THE_STALE_DICT = ("moonshot", "zai", "qwencloud")


@pytest.fixture
def scratch_config(tmp_path, monkeypatch):
    """Point DocumentGenerator's config reads at a throwaway empty config.

    Uses the sanctioned ``select_config_source`` helper (fresh config module
    bound to the selected file) rather than raw env-var switching, which the
    guarded loader rejects with ``RecoveryRequired``.
    """
    from Tests.Backup_Recovery.config_test_support import select_config_source

    config_file = tmp_path / "config.toml"
    config_file.write_text("", encoding="utf-8")
    select_config_source(monkeypatch, config_file, vars(dg_module))
    yield


@pytest.mark.parametrize("provider", _PROVIDERS_ADDED_AFTER_THE_STALE_DICT)
def test_document_generator_dispatches_modern_providers(
    scratch_config, monkeypatch, provider
):
    """Providers registered after the stale dict (which raised
    ``ChatAPIError("Unknown provider: moonshot")``) now dispatch through
    chat_api_call and succeed."""
    captured = []

    def fake_dispatch(**kwargs):
        captured.append(kwargs)
        return "generated content"

    monkeypatch.setattr(dg_module, "chat_api_call", fake_dispatch)
    generator = DocumentGenerator(":memory:", "test-client")

    result = generator.generate_timeline(
        conversation_id="nonexistent",
        provider=provider,
        model="some-model",
        api_key="k",
    )

    assert result == "generated content"
    assert captured, "chat_api_call dispatch fake was never invoked"
    assert captured[0]["api_endpoint"] == provider
    messages = captured[0]["messages_payload"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert captured[0]["streaming"] is False
    assert captured[0]["model"] == "some-model"
    # The stale provider dict is gone.
    assert not hasattr(generator, "provider_functions")


def test_document_generator_unknown_provider_surfaces_dispatcher_error(
    scratch_config,
):
    """An unregistered provider name now comes from the dispatcher's
    vocabulary (ValueError listing valid endpoints), not a
    ``ChatAPIError("Unknown provider: ...")`` from the local dict."""
    # No stub: the real dispatcher must reject the name.
    generator = DocumentGenerator(":memory:", "test-client")
    with pytest.raises(ValueError, match="Unsupported API endpoint"):
        generator.generate_timeline(
            conversation_id="nonexistent",
            provider="not-a-real-provider",
            model="m",
            api_key="k",
        )


def test_document_generator_preserves_chat_error_type(
    scratch_config, monkeypatch
):
    """A Chat*Error from the dispatcher propagates unchanged (no
    re-wrapping that would erase the classified error type)."""
    sentinel = ChatProviderError("upstream 502")

    def failing_dispatch(**kwargs):
        raise sentinel

    monkeypatch.setattr(dg_module, "chat_api_call", failing_dispatch)
    generator = DocumentGenerator(":memory:", "test-client")
    with pytest.raises(ChatProviderError) as excinfo:
        generator.generate_briefing(
            conversation_id="nonexistent",
            provider="openai",
            model="gpt-4",
            api_key="k",
        )
    assert excinfo.value is sentinel
