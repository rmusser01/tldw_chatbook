"""TASK-33620.10: every Console provider streams by default.

Owner decision, 2026-10-27: ``streaming = true`` ships in ``[chat_defaults]``,
which outranks the per-provider ``[api_settings.*]`` scalars, so OpenAI,
Anthropic and every other provider stream out of the box. Before it, eleven
providers (OpenAI, Anthropic, Cohere, DeepSeek, Groq, Google, HuggingFace,
MistralAI, OpenRouter, Custom, Custom_2) resolved Off from the shipped
template and showed "Generating…" for the whole reply.

The shipped template is deep-merged under EVERY load (not only used to seed a
new file), so a profile written by an older template inherits the new key;
an explicit choice in the user's file -- ``streaming`` or the legacy
``enable_streaming`` -- keeps winning, and a provider- or model-scoped saved
default still turns streaming off for a server that refuses it.
"""

import tomllib

import pytest

from Tests.Backup_Recovery.config_test_support import select_config_source
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_session_settings import (
    build_default_console_session_settings,
)

SHIPPED_PROVIDERS = tuple(config_module.DEFAULT_CONFIG_FROM_TOML["providers"])

#: A profile file the way an older template wrote it: the cloud provider
#: blocks carry the old ``streaming = false`` and ``[chat_defaults]`` has no
#: streaming key at all.
OLDER_TEMPLATE_PROFILE = """\
[chat_defaults]
provider = "Anthropic"
model = "claude-haiku-4-5"
temperature = 0.6

[api_settings.anthropic]
model = "claude-haiku-4-5"
streaming = false

[api_settings.openai]
model = "gpt-4.1-mini"
streaming = false
"""


def _load_profile(tmp_path, monkeypatch, text: str) -> dict:
    config_path = tmp_path / "config.toml"
    config_path.write_text(text, encoding="utf-8")
    select_config_source(monkeypatch, str(config_path), globals())
    return config_module.load_settings(force_reload=True)


def test_shipped_chat_defaults_turn_streaming_on():
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert template["chat_defaults"]["streaming"] is True
    assert config_module.DEFAULT_CONFIG_FROM_TOML["chat_defaults"]["streaming"] is True


def test_the_provider_sweep_names_anthropic_and_openai():
    """The parametrized sweep below cannot silently drop the two AC providers."""
    assert {"Anthropic", "OpenAI"} <= set(SHIPPED_PROVIDERS)


@pytest.mark.parametrize("provider", SHIPPED_PROVIDERS)
def test_every_shipped_provider_resolves_console_streaming_on(provider):
    settings = build_default_console_session_settings(
        config_module.DEFAULT_CONFIG_FROM_TOML, provider
    )

    assert settings.streaming is True, provider


def test_a_profile_written_by_an_older_template_streams_after_the_upgrade(
    tmp_path, monkeypatch
):
    settings = _load_profile(tmp_path, monkeypatch, OLDER_TEMPLATE_PROFILE)

    assert settings["chat_defaults"]["streaming"] is True
    for provider in ("Anthropic", "OpenAI"):
        resolved = build_default_console_session_settings(settings, provider)
        assert resolved.streaming is True, provider


def test_an_explicit_chat_defaults_streaming_off_keeps_streaming_off(
    tmp_path, monkeypatch
):
    settings = _load_profile(
        tmp_path, monkeypatch, "[chat_defaults]\nstreaming = false\n"
    )

    for provider in ("Anthropic", "OpenAI", "Llama_cpp"):
        resolved = build_default_console_session_settings(settings, provider)
        assert resolved.streaming is False, provider


def test_a_legacy_enable_streaming_off_keeps_streaming_off_across_load_and_save(
    tmp_path, monkeypatch
):
    """The shipped ``streaming = true`` must never shadow the legacy key.

    ``chat_defaults.enable_streaming`` is read only when ``streaming`` is
    absent from the user's file (Settings says so, TASK-33007.7). Merged
    naively, the template's ``streaming`` would be present in every loaded
    config and silently turn this user's streaming back on.
    """
    settings = _load_profile(
        tmp_path, monkeypatch, "[chat_defaults]\nenable_streaming = false\n"
    )

    assert build_default_console_session_settings(settings, "OpenAI").streaming is False
    assert config_module.get_chat_defaults_streaming(default=True) is False

    # A later save re-publishes the merged view from the raw file it wrote;
    # the atomic snapshot merges the raw file under the write lock. Both are
    # separate merges of the shipped defaults and must agree.
    assert config_module.save_setting_to_cli_config("chat_defaults", "temperature", 0.4)
    published = config_module.load_cli_config_and_ensure_existence()
    assert (
        build_default_console_session_settings(published, "OpenAI").streaming is False
    )
    snapshot = config_module.get_atomic_config_snapshot()
    assert (
        build_default_console_session_settings(snapshot.values, "OpenAI").streaming
        is False
    )


def test_a_provider_saved_default_with_streaming_off_outranks_the_new_default(
    tmp_path, monkeypatch
):
    """The non-streaming fallback for a server that refuses streaming.

    Console's "Save as default" for a provider writes
    ``[console.provider_defaults.<provider>]``; it outranks ``[chat_defaults]``.
    """
    settings = _load_profile(
        tmp_path,
        monkeypatch,
        "[console.provider_defaults.custom]\nstreaming = false\n",
    )

    assert build_default_console_session_settings(settings, "Custom").streaming is False
    assert build_default_console_session_settings(settings, "OpenAI").streaming is True


def test_a_model_saved_default_with_streaming_off_outranks_the_new_default(
    tmp_path, monkeypatch
):
    settings = _load_profile(
        tmp_path,
        monkeypatch,
        '[api_settings.openai.model_defaults."gpt-4.1-mini"]\nstreaming = false\n',
    )

    assert (
        build_default_console_session_settings(
            settings, "OpenAI", "gpt-4.1-mini"
        ).streaming
        is False
    )
    assert (
        build_default_console_session_settings(
            settings, "OpenAI", "gpt-5-mini"
        ).streaming
        is True
    )
