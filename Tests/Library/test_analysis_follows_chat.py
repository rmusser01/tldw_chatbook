"""TASK-34100.2 AC#8 (coverage-17): document analysis follows the chat model.

Setup writes only ``[chat_defaults]``, while Library's analyze-after-import,
the Media analysis panel and the Reader's Generate read ``[analysis_defaults]``
-- which every profile carries from the shipped template (OpenAI / gpt-4o).
A user who connected Anthropic was told analysis was blocked because OpenAI
had no key. While ``[analysis_defaults]`` still equals the shipped template,
analysis now uses the chat provider and model; a value the user set is never
overwritten (nothing is written at all), and a not-ready hint says why a
provider is involved instead of asking for its key.
"""

from __future__ import annotations

import copy
from types import MethodType

from Tests.private_profile import private_profile_test
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
from tldw_chatbook.Library.ingest_analysis import (
    effective_analysis_defaults,
    resolve_ingest_analysis_provider,
)

ANTHROPIC_KEY = "sk-ant-test-analysis-follows-chat"


def _profile(chat_provider="anthropic", chat_model="claude-sonnet-5", **analysis):
    config = copy.deepcopy(
        {
            "analysis_defaults": DEFAULT_CONFIG_FROM_TOML["analysis_defaults"],
            "chat_defaults": {"provider": chat_provider, "model": chat_model},
            "api_settings": {
                "anthropic": {"api_key": ANTHROPIC_KEY},
                "openai": {"api_key_env_var": "OPENAI_API_KEY"},
            },
        }
    )
    config["analysis_defaults"].update(analysis)
    return config


def test_a_template_analysis_default_follows_the_chat_provider_and_model() -> None:
    config = _profile()

    defaults, follows_chat = effective_analysis_defaults(config)
    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert follows_chat is True
    assert (defaults["provider"], defaults["model"]) == ("anthropic", "claude-sonnet-5")
    assert resolution.ready is True
    assert resolution.dispatch_name == "anthropic"
    assert resolution.model == "claude-sonnet-5"
    assert resolution.api_key == ANTHROPIC_KEY
    # Nothing was written: the profile still carries the template values.
    assert config["analysis_defaults"]["provider"] == "OpenAI"


def test_an_analysis_default_the_user_set_is_kept() -> None:
    config = _profile(provider="Anthropic", model="claude-haiku-4-5")

    defaults, follows_chat = effective_analysis_defaults(config)

    assert follows_chat is False
    assert (defaults["provider"], defaults["model"]) == ("Anthropic", "claude-haiku-4-5")


def test_changing_only_the_model_still_counts_as_a_user_choice() -> None:
    _defaults, follows_chat = effective_analysis_defaults(_profile(model="gpt-4.1"))

    assert follows_chat is False


def test_the_not_ready_hint_says_why_openai_is_involved_not_to_get_a_key() -> None:
    # The user chose OpenAI for analysis on purpose; OpenAI has no key.
    config = _profile(provider="OpenAI", model="gpt-4.1")

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is False
    assert "analysis uses its own default" in resolution.hint
    assert "OpenAI · gpt-4.1" in resolution.hint
    assert "[analysis_defaults]" in resolution.hint
    assert "OPENAI_API_KEY" not in resolution.hint
    assert "api_key under" not in resolution.hint


def test_a_following_analysis_names_the_chat_model_when_chat_is_not_ready() -> None:
    config = _profile(chat_provider="openai", chat_model="gpt-5.6-terra")

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is False
    assert "your chat model" in resolution.hint
    assert "OpenAI · gpt-5.6-terra" in resolution.hint


class _Field:
    def __init__(self) -> None:
        self.value = None
        self.text = ""
        self.options: list = []

    def set_options(self, options) -> None:
        self.options = list(options)


class _Panel:
    """Just enough of MediaViewerPanel for its two default-seeding methods."""

    def __init__(self) -> None:
        self.fields: dict[str, _Field] = {}

    def query_one(self, selector, _kind=None):
        return self.fields.setdefault(selector, _Field())


def test_the_media_analysis_panel_seeds_the_same_effective_defaults(monkeypatch) -> None:
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel

    monkeypatch.setattr(config_module, "load_settings", lambda: _profile())
    monkeypatch.setattr(
        config_module,
        "get_cli_providers_and_models",
        lambda: {
            "OpenAI": ["gpt-4o", "gpt-4.1"],
            "Anthropic": ["claude-haiku-4-5", "claude-sonnet-5"],
        },
    )
    panel = _Panel()
    panel.update_models_for_provider = MethodType(
        MediaViewerPanel.update_models_for_provider, panel
    )

    MediaViewerPanel.populate_providers(panel)

    assert panel.fields["#analysis-provider-select"].value == "Anthropic"
    # The chat model, not the provider's first listed model.
    assert panel.fields["#analysis-model-select"].value == "claude-sonnet-5"


# --- the app's config snapshot carries [analysis_defaults] (live finding) ---


def _load_settings_from(monkeypatch, body: str) -> dict:
    from pathlib import Path

    from tldw_chatbook import config as config_module

    config_path = Path(config_module.get_cli_config_path())
    config_path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    return config_module.load_settings(force_reload=True)


@private_profile_test
def test_the_loaded_settings_follow_the_chat_model(request, monkeypatch) -> None:
    """Live (g2-analysis/01): with Anthropic set up, Library's import panel
    said 'no analysis provider is configured', because load_settings()
    dropped [analysis_defaults] from the app's config snapshot -- every
    surface that reads app_config (Library import, the ingest job, the Media
    panel) saw no section at all."""
    from tldw_chatbook.Library.ingest_analysis import resolve_ingest_analysis_provider

    settings = _load_settings_from(
        monkeypatch,
        '[chat_defaults]\nprovider = "anthropic"\nmodel = "claude-sonnet-5"\n',
    )

    assert settings["analysis_defaults"]["provider"] == (
        DEFAULT_CONFIG_FROM_TOML["analysis_defaults"]["provider"]
    )
    resolution = resolve_ingest_analysis_provider(
        settings, environ={"ANTHROPIC_API_KEY": ANTHROPIC_KEY}
    )
    assert resolution.provider == "anthropic"
    assert resolution.model == "claude-sonnet-5"
    assert resolution.ready is True


@private_profile_test
def test_the_loaded_settings_keep_a_users_own_analysis_model(
    request, monkeypatch
) -> None:
    from tldw_chatbook.Library.ingest_analysis import resolve_ingest_analysis_provider

    settings = _load_settings_from(
        monkeypatch,
        '[chat_defaults]\nprovider = "anthropic"\nmodel = "claude-sonnet-5"\n'
        '[analysis_defaults]\nprovider = "OpenAI"\nmodel = "gpt-4.1"\n',
    )

    resolution = resolve_ingest_analysis_provider(settings, environ={})
    assert resolution.provider == "OpenAI"
    assert "its own default (OpenAI · gpt-4.1, set under [analysis_defaults])" in (
        resolution.hint
    )
