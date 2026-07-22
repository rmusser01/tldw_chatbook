import os
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from textual.widgets import Input, Static

from Tests.UI.test_destination_shells import (
    _active_destination_screen,
    _static_text,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_configuration_hub import (
    StyledSettingsDestinationHarness,
    _click_scrolled_settings_button,
    _open_settings_category,
    _wait_for_settings_text,
)
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


# --- Pilot tests: the clickable Test button path (AC#2/AC#3) + widget wiring ---
#
# These drive the real SettingsScreen through the harness
# Tests/UI/test_settings_configuration_hub.py uses (StyledSettingsDestinationHarness
# is required alongside _click_scrolled_settings_button -- every existing caller
# of that helper in the suite uses the styled harness so the detail-pane scroll
# geometry the click depends on is computed from real CSS).


def _provider_test_result_text(screen) -> str:
    return _static_text(screen.query_one("#settings-provider-test-result", Static))


@pytest.mark.asyncio
async def test_test_provider_button_click_runs_the_check():
    """AC#2: clicking #settings-test-provider (not the 't' hotkey) runs the test."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        # Sanity: the test has not run yet (mount-time default copy only).
        assert _provider_test_result_text(screen) == "Provider test has not run."

        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "Provider test")

        result_text = _provider_test_result_text(screen)
        assert "Provider test" in result_text
        assert result_text != "Provider test has not run."


@pytest.mark.asyncio
async def test_test_provider_button_runs_with_provider_input_focused():
    """AC#3: the button still runs the check with a provider Input focused.

    ``action_settings_test_category`` no-ops for the 't' hotkey while a text
    entry widget has focus (``_settings_text_entry_has_focus``); the button's
    handler (``handle_test_provider``) passes ``allow_text_entry_focus=True``
    explicitly, so it must run regardless of what has focus.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        model_input = screen.query_one("#settings-model-value", Input)
        model_input.focus()
        await pilot.pause()
        # Sanity: this is exactly the state that would make the 't' hotkey no-op.
        assert screen._settings_text_entry_has_focus() is True

        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "Provider test")

        assert "Provider test" in _provider_test_result_text(screen)


@pytest.mark.asyncio
async def test_test_provider_result_shows_draft_endpoint():
    """Wiring: a staged (unsaved) endpoint edit reaches the Test result.

    Exercises the widget-reading wrapper (``_provider_readiness_test_report``)
    that Task 2's unit tests (above) did not cover, by typing a draft endpoint
    into the real ``#settings-provider-endpoint-value`` input, firing its
    change handler (staging it dirty), then running the test via the button.
    The model is left unset so readiness never "passes" and no async endpoint
    probe worker starts -- keeping the assertion on the synchronously-set
    pre-probe detail line, which is where the draft tag is threaded through.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": ""}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://localhost:8080"},
        "openai": {"api_base_url": "https://api.openai.com/v1"},
    }
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://localhost:9099"
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))
        await pilot.pause()

        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "Provider test")

        detail = _provider_test_result_text(screen)
        assert "http://localhost:9099 (draft)" in detail
        assert "status=blocked" in detail  # no model set -> readiness never "passes"
