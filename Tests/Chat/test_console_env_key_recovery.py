"""Console 'Set up provider' opens on a provider whose key is exported.

TASK-34100.6 (review new-coverage-power-03): with only ANTHROPIC_API_KEY
exported, the card's 'Set up provider' opened Settings ▸ Providers & Models
on the template OpenAI ('API key source: missing'), and nothing said an
Anthropic key was found. It now preselects the provider whose key is there.
(TASK-34100.2 owns the card's own 'Use <provider>' offer.)
"""

from __future__ import annotations

from tldw_chatbook.Chat.console_onboarding_state import env_keyed_provider_for_recovery

_KEY = "sk-ant-" + "a" * 40
_TEMPLATE = {
    "chat_defaults": {"provider": "openai"},
    "api_settings": {"openai": {"api_key": "<API_KEY_HERE>"}},
}


def test_an_exported_key_for_another_provider_is_preselected():
    assert (
        env_keyed_provider_for_recovery(_TEMPLATE, "openai", {"ANTHROPIC_API_KEY": _KEY})
        == "anthropic"
    )


def test_nothing_changes_when_the_current_provider_has_its_key():
    environ = {"ANTHROPIC_API_KEY": _KEY, "OPENAI_API_KEY": "sk-proj-" + "b" * 40}
    assert env_keyed_provider_for_recovery(_TEMPLATE, "openai", environ) is None


def test_placeholders_and_empty_values_are_not_keys():
    environ = {"ANTHROPIC_API_KEY": "<API_KEY_HERE>", "GROQ_API_KEY": "  "}
    assert env_keyed_provider_for_recovery(_TEMPLATE, "openai", environ) is None


def test_a_keyless_current_provider_is_left_alone():
    """A local provider never needed a key: nothing to preselect over it."""
    config = {"chat_defaults": {"provider": "llama_cpp"}, "api_settings": {}}
    assert (
        env_keyed_provider_for_recovery(config, "llama_cpp", {"ANTHROPIC_API_KEY": _KEY})
        is None
    )


def test_the_ui_thread_lookup_never_reads_a_subscription_credential_synchronously(
    monkeypatch,
):
    """Review round 1, F8: 'Set up provider' runs this on the UI thread. For an
    Anthropic subscription it read the Claude credential synchronously (with a
    macOS Keychain fallback); every other UI readiness call asks for the
    background read."""
    from tldw_chatbook.Chat import provider_readiness

    seen: list[dict] = []
    real = provider_readiness.get_provider_readiness

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(provider_readiness, "get_provider_readiness", recording)
    env_keyed_provider_for_recovery(_TEMPLATE, "openai", {"ANTHROPIC_API_KEY": _KEY})
    assert seen and all(kwargs.get("background_credentials") is True for kwargs in seen)
