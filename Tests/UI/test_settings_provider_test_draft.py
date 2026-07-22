import os
from unittest.mock import patch
from types import SimpleNamespace

from tldw_chatbook.UI.Screens.settings_screen import (
    SettingsScreen,
    overlay_provider_draft_config,
)
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness


def _base_config():
    return {
        "api_settings": {
            "llama_cpp": {"api_url": "http://localhost:8080/completion", "api_key": "saved-key"},
            "openai": {"api_key": "other-saved"},
        }
    }


def test_overlay_endpoint_only_deep_copies_and_preserves_others():
    base = _base_config()
    merged = overlay_provider_draft_config(
        base,
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint="http://localhost:9099",
        draft_env_var=None,
        draft_api_key=None,
    )
    # draft endpoint overlaid
    assert merged["api_settings"]["llama_cpp"]["api_url"] == "http://localhost:9099"
    # saved key + other provider preserved
    assert merged["api_settings"]["llama_cpp"]["api_key"] == "saved-key"
    assert merged["api_settings"]["openai"]["api_key"] == "other-saved"
    # input not mutated
    assert base["api_settings"]["llama_cpp"]["api_url"] == "http://localhost:8080/completion"


def test_overlay_api_key_and_env_var():
    merged = overlay_provider_draft_config(
        _base_config(),
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var="MY_LLAMA_KEY",
        draft_api_key="draft-secret",
    )
    section = merged["api_settings"]["llama_cpp"]
    assert section["api_key"] == "draft-secret"
    assert section["api_key_env_var"] == "MY_LLAMA_KEY"


def test_overlay_api_key_clear_sets_empty():
    merged = overlay_provider_draft_config(
        _base_config(),
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var=None,
        draft_api_key="",
    )
    assert merged["api_settings"]["llama_cpp"]["api_key"] == ""


def test_overlay_creates_missing_section():
    merged = overlay_provider_draft_config(
        {"api_settings": {}},
        provider_save_key="newprov",
        endpoint_key="api_base_url",
        draft_endpoint="http://x:1/v1",
        draft_env_var=None,
        draft_api_key=None,
    )
    assert merged["api_settings"]["newprov"]["api_base_url"] == "http://x:1/v1"


def test_overlay_no_fields_is_a_faithful_copy():
    base = _base_config()
    merged = overlay_provider_draft_config(
        base,
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var=None,
        draft_api_key=None,
    )
    assert merged == base
    assert merged is not base


def _bare_settings_screen(app_config):
    screen = SettingsScreen.__new__(SettingsScreen)
    screen.app_instance = SimpleNamespace(app_config=app_config)
    return screen


def test_findings_show_draft_endpoint_tagged():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:9099"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, _summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp", "llama-3", readiness,
        draft_endpoint="http://localhost:9099", dirty={"endpoint"},
    )
    assert "http://localhost:9099 (draft)" in detail
    assert "8080" not in detail


def test_findings_relabel_draft_api_key_source_and_hide_value():
    app_config = {"api_settings": {"openai": {"api_key": "draft-secret"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})
    detail, summary, passed = screen._build_provider_readiness_findings(
        "OpenAI", "gpt-4o", readiness,
        draft_endpoint="", dirty={"api_key"},
    )
    assert "api_key_source=draft api_key (unsaved)" in detail
    assert "draft-secret" not in detail and "draft-secret" not in summary


def test_findings_tag_draft_env_var():
    app_config = {"api_settings": {"openai": {"api_key_env_var": "MY_KEY"}}}
    screen = _bare_settings_screen(app_config)
    with patch.dict(os.environ, {"MY_KEY": "envval"}, clear=False):
        readiness = get_provider_readiness("OpenAI", app_config)
        detail, _summary, _passed = screen._build_provider_readiness_findings(
            "OpenAI", "gpt-4o", readiness,
            draft_endpoint="", dirty={"credential_env_var"},
        )
    assert "(draft env var)" in detail


def test_findings_no_draft_has_no_tags():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, _summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp", "llama-3", readiness,
        draft_endpoint="http://localhost:8080", dirty=set(),
    )
    assert "(draft)" not in detail and "(unsaved)" not in detail
    assert "http://localhost:8080" in detail
