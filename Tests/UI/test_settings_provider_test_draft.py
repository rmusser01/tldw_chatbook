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
            "llama_cpp": {"api_url": "http://localhost:8080/completion", "api_key": "fake-saved-key-not-real"},
            "openai": {"api_key": "fake-other-key-not-real"},
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
    assert merged["api_settings"]["llama_cpp"]["api_key"] == "fake-saved-key-not-real"
    assert merged["api_settings"]["openai"]["api_key"] == "fake-other-key-not-real"
    # input not mutated
    assert base["api_settings"]["llama_cpp"]["api_url"] == "http://localhost:8080/completion"


def test_overlay_api_key_and_env_var():
    merged = overlay_provider_draft_config(
        _base_config(),
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var="MY_LLAMA_KEY",
        draft_api_key="fake-draft-key-not-real",
    )
    section = merged["api_settings"]["llama_cpp"]
    assert section["api_key"] == "fake-draft-key-not-real"
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
    app_config = {"api_settings": {"openai": {"api_key": "fake-draft-key-not-real"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})
    detail, summary, passed = screen._build_provider_readiness_findings(
        "OpenAI", "gpt-4o", readiness,
        draft_endpoint="", dirty={"api_key"},
    )
    assert "api_key_source=draft api_key (unsaved)" in detail
    assert "fake-draft-key-not-real" not in detail and "fake-draft-key-not-real" not in summary


def test_findings_tag_draft_env_var_and_never_leak_value():
    # A custom-named credential env var whose NAME does not match the secret
    # pattern -- its raw value must still never be printed (task-483 folded in),
    # only presence via the ``<redacted>`` marker, plus the draft tag.
    app_config = {"api_settings": {"openai": {"api_key_env_var": "MY_CUSTOM_CRED"}}}
    screen = _bare_settings_screen(app_config)
    with patch.dict(os.environ, {"MY_CUSTOM_CRED": "env-secret-XYZ"}, clear=False):
        readiness = get_provider_readiness("OpenAI", app_config)
        detail, summary, _passed = screen._build_provider_readiness_findings(
            "OpenAI", "gpt-4o", readiness,
            draft_endpoint="", dirty={"credential_env_var"},
        )
    assert "(draft env var)" in detail
    assert "MY_CUSTOM_CRED=<redacted>" in detail
    assert "env-secret-XYZ" not in detail and "env-secret-XYZ" not in summary


def test_mask_url_userinfo_masks_password_in_endpoint():
    from tldw_chatbook.UI.Screens.settings_screen import _mask_url_userinfo

    assert _mask_url_userinfo("http://user:s3cret@host:8080/v1") == "http://user:***@host:8080/v1"
    assert _mask_url_userinfo("http://:s3cret@host/v1") == "http://***@host/v1"
    # password-less / non-URL inputs are unchanged
    assert _mask_url_userinfo("http://localhost:9099") == "http://localhost:9099"
    assert _mask_url_userinfo("") == ""
    # username-only userinfo (no password) is left as-is
    assert _mask_url_userinfo("http://user@host/v1") == "http://user@host/v1"
    # malformed/out-of-range port must not raise (uses .port property otherwise)
    assert _mask_url_userinfo("http://u:p@host:99999/v1") == "http://u:***@host:99999/v1"
    assert _mask_url_userinfo("http://u:p@host:notaport/v1") == "http://u:***@host:notaport/v1"
    # IPv6 host keeps its brackets while the password is masked
    assert (
        _mask_url_userinfo("http://u:p@[::1]:8080/v1") == "http://u:***@[::1]:8080/v1"
    )


def test_findings_mask_endpoint_userinfo_password():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp", "llama-3", readiness,
        draft_endpoint="http://user:hunter2@host:9099/v1", dirty={"endpoint"},
    )
    assert "http://user:***@host:9099/v1 (draft)" in detail
    assert "hunter2" not in detail and "hunter2" not in summary


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


def test_findings_avoid_ready_claim_when_blocked_on_missing_model():
    """TASK-366: a config-ready provider with no default model must not read
    'is ready' next to 'status=blocked' — the detail leads with one verdict
    consistent with the final status line, and still explains the block."""
    app_config = {"api_settings": {"openai": {"api_key": "sk-test-fake"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})
    assert readiness.ready is True  # config-level readiness is fine...

    detail, summary, passed = screen._build_provider_readiness_findings(
        "OpenAI", "", readiness,
        draft_endpoint="", dirty=set(),
    )

    assert passed is False
    assert "status=blocked" in detail
    assert "is ready" not in detail  # no contradictory ready claim
    assert "model" in detail.lower()  # verdict still explains the block


def test_findings_keep_ready_verdict_when_passing():
    """TASK-366 guard: a genuine pass must still read 'ready' / status=ready."""
    app_config = {"api_settings": {"openai": {"api_key": "sk-test-fake"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})

    detail, summary, passed = screen._build_provider_readiness_findings(
        "OpenAI", "gpt-4o", readiness,
        draft_endpoint="", dirty=set(),
    )

    assert passed is True
    assert "status=ready" in detail
    assert "ready" in summary.lower()


def test_mark_provider_test_result_stale_invalidates_prior_verdict():
    """TASK-366: editing a provider input must invalidate a prior Test result so
    a stale 'ready'/'blocked' verdict cannot linger while the form has changed.
    No-op when nothing has run or it is already stale."""
    screen = _bare_settings_screen({})
    screen._provider_test_result = (
        "Provider test | llama.cpp is ready | model=llama-3 | status=ready"
    )

    screen._mark_provider_test_result_stale()
    assert "re-run" in screen._provider_test_result.lower()

    # Idempotent: a second edit does not re-flag or accumulate.
    stale = screen._provider_test_result
    screen._mark_provider_test_result_stale()
    assert screen._provider_test_result == stale

    # No-op on the never-run sentinel.
    screen._provider_test_result = SettingsScreen._PROVIDER_TEST_NOT_RUN_COPY
    screen._mark_provider_test_result_stale()
    assert screen._provider_test_result == SettingsScreen._PROVIDER_TEST_NOT_RUN_COPY


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
    """AC#3: a real mouse click on the button still runs the check, starting
    from an Input-focused state.

    This proves the button is a working non-hotkey path: even when a text
    entry widget starts out focused, clicking ``#settings-test-provider``
    runs the readiness check and reads the current widget values.

    Note: by the time ``Button.Pressed`` dispatches, Textual has already
    moved keyboard focus onto the Button itself, so this test does not by
    itself pin the ``allow_text_entry_focus=True`` bypass in
    ``handle_test_provider`` -- see
    ``test_t_hotkey_does_not_run_test_while_input_focused`` below, which
    pins the actual rationale (the 't' hotkey no-ops while an input has
    focus, which is why a clickable button is needed).
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
async def test_t_hotkey_does_not_run_test_while_input_focused():
    """AC#3 rationale: the 't' hotkey does not run the test while a text
    entry has focus -- this is why the clickable button is needed.

    Two things are pinned here:

    1. The observable behavior a real keypress produces: pressing 't' while
       the model Input is focused types "t" into the input rather than
       running the readiness check. (Textual's own Input widget consumes
       printable keys before the Screen's ``("t", "settings_test_category",
       ...)`` binding is even considered -- see
       ``Input.check_consume_key``/``Screen._binding_chain`` -- so this
       part alone would hold even if ``action_settings_test_category``'s
       internal guard were removed.)
    2. The actual guard: ``action_settings_test_category`` (the method the
       't' binding invokes, with no arguments -- i.e.
       ``allow_text_entry_focus=False``) is a no-op while
       ``_settings_text_entry_has_focus()`` is true. Calling it directly,
       the same way the binding dispatch would, is what makes this test
       fail if that guard is ever removed -- part 1 alone would not catch
       that regression, since Textual's own key consumption already
       prevents the keypress from reaching the binding either way.
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

        before = _provider_test_result_text(screen)
        assert before == "Provider test has not run."

        # 1. Real keypress: consumed by the focused Input, never reaches the
        # 't' binding at all.
        await pilot.press("t")
        await pilot.pause()
        assert _provider_test_result_text(screen) == before
        assert "t" in model_input.value

        # 2. Direct action-level check -- the same call Textual's binding
        # dispatch makes for the 't' hotkey (no arguments). This is the part
        # that actually exercises `_settings_text_entry_has_focus()`.
        screen.action_settings_test_category()
        await pilot.pause()
        assert _provider_test_result_text(screen) == before


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
