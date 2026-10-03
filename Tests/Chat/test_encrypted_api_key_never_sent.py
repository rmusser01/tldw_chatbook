"""An ``enc:`` config value is never sent as a credential (TASK-34100.4).

new-protect-summary-03: a locked session (encryption on, no password) or a
value that did not decrypt keeps its ``enc:`` ciphertext in the loaded config.
Provider handlers whose fallback read ``api_key`` raw from config, and
``chat_api_call`` itself, would hand that ciphertext to the provider as a
bearer token (and fail with a 401). These tests drive each handler with only
an encrypted key configured and require the missing-key refusal instead --
before any request is built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatAuthenticationError, ChatConfigurationError

CIPHERTEXT = "enc:AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
MESSAGES = [{"role": "user", "content": "hello"}]
REFUSALS = (ChatConfigurationError, ChatAuthenticationError)


def _snapshot(provider: str):
    return lambda: SimpleNamespace(
        values={"api_settings": {provider: {"api_key": CIPHERTEXT}}}
    )


@pytest.mark.parametrize(
    ("module_name", "function_name", "provider"),
    [
        ("tldw_chatbook.LLM_Calls.deepseek", "chat_with_deepseek", "deepseek"),
        ("tldw_chatbook.LLM_Calls.groq", "chat_with_groq", "groq"),
        ("tldw_chatbook.LLM_Calls.mistral", "chat_with_mistral", "mistral"),
        ("tldw_chatbook.LLM_Calls.openrouter", "chat_with_openrouter", "openrouter"),
        ("tldw_chatbook.LLM_Calls.LLM_API_Calls", "chat_with_cohere", "cohere"),
    ],
)
def test_snapshot_backed_handlers_refuse_ciphertext(
    monkeypatch, module_name, function_name, provider
):
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "get_runtime_config_snapshot", _snapshot(provider))

    with pytest.raises(REFUSALS):
        getattr(module, function_name)(MESSAGES, api_key=None, model="m")


def test_openai_refuses_ciphertext_from_either_config_table(monkeypatch):
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    monkeypatch.setattr(
        LLM_API_Calls._provider_recovery, "recovered_settings", lambda: None
    )
    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {
            "openai_api": {"api_key": CIPHERTEXT},
            "api_settings": {"openai": {"api_key": CIPHERTEXT}},
        },
    )

    with pytest.raises(REFUSALS):
        LLM_API_Calls.chat_with_openai.__wrapped__(MESSAGES, api_key=None)


def test_anthropic_refuses_ciphertext(monkeypatch):
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {"anthropic_api": {"api_key": CIPHERTEXT}},
    )

    with pytest.raises(REFUSALS):
        LLM_API_Calls.chat_with_anthropic(MESSAGES, api_key=None, model="m")


def test_chat_api_call_drops_a_ciphertext_api_key(monkeypatch):
    from tldw_chatbook.Chat import Chat_Functions

    received = {}

    def capture(**kwargs):
        received.update(kwargs)
        return "ok"

    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, "openai", capture)

    assert (
        Chat_Functions.chat_api_call("openai", MESSAGES, api_key=CIPHERTEXT) == "ok"
    )
    assert received.get("api_key") is None
