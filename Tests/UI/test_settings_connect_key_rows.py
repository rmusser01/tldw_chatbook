"""Settings ▸ Providers & Models ▸ Connect: the key and endpoint rows (TASK-33007.2).

Split out of test_settings_connect_rows.py so the UI Fast Lane's round-robin
shards can spread the Connect cases (each mounts the whole Settings screen,
~16 s a case on CI). The API key row's source words and Clear key, the Key
check row, an endpoint with no shipped default, and the card's result line.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import (
    _CLEAR_HINT,
    _CLEAR_KEY,
    _FAKE_KEY,
    _SAVED_KEY,
    _SIZE,
    _open_providers,
    _resting_help_counts,
    _squeezed,
    _text,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("setup", "word"),
    [("config", "saved in config"), ("env", "from env var"), ("none", "missing")],
)
async def test_api_key_row_says_where_the_key_comes_from(
    request, monkeypatch, setup, word
):
    """AC#4: masked, one row, the source in words, and Clear still offered."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if setup == "env":
        monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": _FAKE_KEY} if setup == "config" else {}
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        status = screen.query_one("#settings-provider-key-status", Static)
        row = screen.query_one("#settings-provider-api-key-row")

        assert api_key.password
        assert row.region.height == 1
        assert str(status.renderable) == word
        assert clear.parent is row
        assert clear.disabled is (setup != "config")
        assert _FAKE_KEY not in "\n".join(_region_rows(screen, row))
        # ADR-031 rule 4: the row names Clear's key only while it clears.
        assert (_CLEAR_HINT in _text(screen, "#settings-provider-api-key-help")) is (
            setup == "config"
        )

        env_row = screen.query_one("#settings-provider-env-var-row")
        endpoint_row = screen.query_one("#settings-provider-endpoint-row")
        assert env_row.region.height == endpoint_row.region.height == 1
        assert _text(screen, "#settings-provider-env-var-source") == (
            "set in shell" if setup == "env" else "not set"
        )
        assert _text(screen, "#settings-provider-endpoint-source") == "built-in"
        assert "safer" in _text(screen, "#settings-provider-credential-guidance")
        assert _text(screen, "#settings-provider-endpoint-help") == (
            "blank uses the provider default"
        )

        # Which key is sent when both exist: the row's help has no room for it
        # beside the two keys, so the Inspector's guide for the field says.
        api_key.focus()
        await pilot.pause()
        guide = " ".join(
            _text(screen, f"#settings-provider-field-guide-{index}")
            for index in range(4)
        )
        assert "Focused setting: API key" in guide
        assert "A saved key is used before the env var." in guide
        # ADR-031 rule 4 rests on this: while Clear is disabled its key does
        # nothing, so the row names the key only for a saved one. Clear is
        # live beside a saved key, so that case disables it by hand.
        clear.disabled = True
        await pilot.press(_CLEAR_KEY)
        await pilot.pause()
        assert not screen._settings_drafts
        assert str(status.renderable) == word


@pytest.mark.asyncio
@private_profile_test
async def test_key_check_row_ends_connect_and_its_detail_lives_in_the_inspector(
    request, monkeypatch
):
    """AC#6, AC#7, AC#8: the verdict and a t action on one row; the labelled
    rows go to the Inspector's Key block, so Default model does not move."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        card = screen.query_one("#settings-providers-models-card")
        verdict = screen.query_one("#settings-provider-readiness", Static)
        button = screen.query_one("#settings-test-provider", Button)
        result = screen.query_one("#settings-provider-test-result", Static)
        default_model = screen.query_one("#settings-default-model-title", Static)

        assert str(verdict.renderable) == "Not ready · no key"
        assert "(t)" in str(button.label)
        assert button.parent is verdict.parent
        assert verdict.parent.region.height == 1
        assert screen.query_one("#settings-impact-pane") in result.ancestors
        assert card not in result.ancestors
        assert "Provider readiness" not in [
            str(widget.renderable) for widget in card.query(".destination-section")
        ]
        assert _text(screen, "#settings-provider-source") == "new-chat default"
        assert _text(screen, "#settings-model-source") == "new-chat default"
        widgets = list(card.query("*"))
        assert widgets.index(verdict.parent) < widgets.index(default_model)

        default_model_y = default_model.virtual_region.y
        button.press()
        await pilot.pause(0.2)

        assert str(result.renderable).startswith("Readiness")
        assert str(verdict.renderable) == "Not ready · no key"
        assert default_model.virtual_region.y == default_model_y


@pytest.mark.asyncio
@private_profile_test
async def test_ctrl_l_on_the_api_key_field_clears_the_saved_key_until_revert_or_save(
    request, monkeypatch
):
    """Owner ruling 2026-10-04: Clear is a key on the API key field. It
    stages the removal as the button does: a confirmed r undoes it, s saves."""
    from Tests.UI.test_settings_configuration_hub import (
        _capture_provider_settings_mutations,
    )
    from Tests.UI.test_settings_provider_keyboard_journeys import _revert, _settle
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    category = SettingsCategoryId.PROVIDERS_MODELS
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        api_key.focus()
        await pilot.pause()
        await pilot.press(_CLEAR_KEY)
        await pilot.pause()

        assert host.focused is api_key
        assert _text(screen, "#settings-provider-key-status") == "cleared *"
        assert _text(screen, "#settings-provider-api-key-help") == (
            "s removes the saved key"
        )
        assert screen._settings_drafts[category].values["api_key"] == ""
        assert mutations == []
        assert app.app_config["api_settings"]["openai"]["api_key"] == _FAKE_KEY

        await _revert(host, pilot, discard=True)
        assert _text(screen, "#settings-provider-key-status") == "saved in config"
        assert not screen._category_has_unsaved_changes(category)
        assert mutations == []

        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        await pilot.press(_CLEAR_KEY, "escape", "s")
        await _settle(host, pilot)

        assert len(mutations) == 1
        assert "api_key" in mutations[0][1]["api_settings.openai"]
        assert "api_key" not in app.app_config["api_settings"]["openai"]


@pytest.mark.asyncio
@private_profile_test
async def test_an_emptied_api_key_field_keeps_the_saved_key_and_only_clear_removes_it(
    request, monkeypatch
):
    """Final review finding 5: typing a character and deleting it is no edit
    (Web Search's rule); it used to stage removal, so s deleted the key."""
    from Tests.UI.test_settings_configuration_hub import (
        _capture_provider_settings_mutations,
    )
    from Tests.UI.test_settings_provider_keyboard_journeys import _settle

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "edited *"

        await pilot.press("backspace")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "saved in config"
        assert not screen._settings_drafts
        assert not clear.disabled
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert mutations == []
        assert app.app_config["api_settings"]["openai"]["api_key"] == _FAKE_KEY

        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        await pilot.press(_CLEAR_KEY)
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "cleared *"

        # Checkpoint review: typing after Clear and deleting it again returns
        # to Clear's removal; the emptied-field rule must not undo a Clear.
        await pilot.press("x")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "edited *"
        await pilot.press("backspace")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "cleared *"


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("provider", ["azure", "cloudflare", "databricks"])
@pytest.mark.parametrize("where", ["config", "env"])
async def test_a_key_reads_saved_before_its_base_url_is_set(
    request, monkeypatch, provider, where
):
    """Final review finding 2 (AC#2, AC#4): readiness drops the key's source
    for these three until a base URL is set, but the key is there all the
    same -- the row says where, Clear clears a saved one, and the provider
    leads the list as Configured."""
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var

    env_var = default_api_key_env_var(provider)
    assert env_var
    monkeypatch.delenv(env_var, raising=False)
    if where == "env":
        monkeypatch.setenv(env_var, _FAKE_KEY)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {
        provider: dict(_SAVED_KEY) if where == "config" else {}
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)

        assert _text(screen, "#settings-provider-key-status") == (
            "saved in config" if where == "config" else "from env var"
        )
        assert clear.disabled is (where != "config")
        configured = next(
            group
            for group in screen._provider_picker_groups()
            if group.group_id == "configured"
        )
        assert [option.provider_id for option in configured.options] == [provider]
        assert _resting_help_counts(
            _text(screen, "#settings-provider-search-status"), 1
        )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "target"),
    [
        ("azure", "resource host"),
        ("cloudflare", "account URL"),
        ("databricks", "workspace host"),
    ],
)
async def test_an_endpoint_with_no_shipped_default_reads_required(
    request, monkeypatch, provider, target
):
    """Capture 01c (fix 1): with no base URL the Key check reads "Not ready",
    yet the Endpoint row said built-in and "blank uses the provider default".
    These three ship no default, so the row says the URL is required."""
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var

    monkeypatch.delenv(default_api_key_env_var(provider), raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {provider: dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        help_line = f"required: your {target}"

        assert endpoint.value == ""
        assert _text(screen, "#settings-provider-readiness").startswith("Not ready")
        assert _text(screen, "#settings-provider-endpoint-source") == "not set"
        assert _text(screen, "#settings-provider-endpoint-help") == help_line
        assert endpoint.placeholder == f"Enter your {target}"
        painted = _region_rows(
            screen, screen.query_one("#settings-provider-endpoint-row")
        )
        assert help_line in painted[0] and endpoint.placeholder in painted[0], painted


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "settings", "required"),
    [
        ("azure", dict(_SAVED_KEY), "your resource host"),
        ("llama_cpp", {}, "the server's base URL"),
        ("openai", dict(_SAVED_KEY), None),
    ],
)
async def test_the_focused_endpoint_guide_says_required_as_its_row_does(
    request, provider, settings, required
):
    """TASK-33007 follow-up to fix 1: the Endpoint row reads "required: ..."
    for providers that ship no URL, yet its Inspector guide still said the
    address applies "when set". It says required, in the row's words; a
    provider with a default URL keeps "when set"."""
    from tldw_chatbook.Chat.console_provider_support import MODEL_CONFIG_FIELDS

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {provider: settings}
    host = _SettingsCssHarness(app, "settings")
    address = MODEL_CONFIG_FIELDS["endpoint"].valid_range

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        screen.query_one("#settings-provider-endpoint-value", Input).focus()
        await pilot.pause()

        assert _text(screen, "#settings-provider-field-guide-0") == (
            "Focused setting: Endpoint"
        )
        guide = _squeezed(_text(screen, "#settings-provider-field-guide-3"))
        if required is None:
            assert guide == _squeezed(f"Validation: {address} when set")
            return
        assert _text(screen, "#settings-provider-endpoint-help") == (
            f"required: {required}"
        )
        assert guide == _squeezed(f"Validation: {address}; required: {required}")


@pytest.mark.asyncio
@private_profile_test
async def test_the_card_result_line_takes_no_row_until_there_is_a_result(
    request, monkeypatch
):
    """Checkpoint review (captures): every card opened with "… have not been
    saved this session." -- a row saying nothing happened, while the Inspector
    already says "Save (s) — no changes". The line now shows only once a save
    or revert has something to report."""
    from Tests.UI.test_settings_configuration_hub import (
        _capture_provider_settings_mutations,
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _capture_provider_settings_mutations(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        result = screen.query_one("#settings-provider-save-result", Static)
        assert result.display is False
        assert result.region.height == 0

        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        from tldw_chatbook.UI.Screens.settings_config_models import (
            SettingsCategoryId,
        )

        await pilot.press("x")
        await pilot.pause()
        # What "r" then "Discard changes" runs once confirmed.
        screen._revert_category(SettingsCategoryId.PROVIDERS_MODELS)
        await pilot.pause()
        await pilot.pause()
        assert result.display is True
        assert "reverted" in _text(screen, "#settings-provider-save-result")
