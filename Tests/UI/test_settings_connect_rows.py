"""Settings ▸ Providers & Models ▸ Connect, one row per fact (TASK-33007.2).

Mounted with the real application stylesheet at 211x44, the size the spec's
mockup (c) is drawn at, and driven by real keypresses.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, OptionList, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness, _region_rows

_SIZE = (211, 44)
_FAKE_KEY = "sk-proj-abcdefghijklmnop1234"


async def _open_providers(app, pilot):
    await _settle_settings(pilot)
    await _click_settings_category(pilot, "providers-models")
    await pilot.pause()
    return _active_destination_screen(app)


def _text(screen, selector: str) -> str:
    return str(screen.query_one(selector, Static).renderable)


@pytest.mark.asyncio
@private_profile_test
async def test_provider_is_one_row_one_tab_stop_and_filters_by_name_or_id(request):
    """AC#1, AC#3: one row, one Tab stop; typing filters by name or id; the
    list's rows paint whole; Enter chooses and the control names the choice."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "anthropic", "model": "claude-x"}
    app.app_config["api_settings"] = {"anthropic": {"api_key": _FAKE_KEY}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)
        api_key = screen.query_one("#settings-provider-api-key", Input)

        assert control.region.height == 1
        assert control.value == "Anthropic"
        assert not picker.display
        assert "configured: Anthropic ·" in _text(
            screen, "#settings-provider-search-status"
        )

        control.focus()
        await pilot.pause()
        await pilot.press(*"anthro")
        await pilot.pause()
        assert picker.display
        painted = "\n".join(_region_rows(screen, picker))
        assert "Anthropic" in painted, painted
        # A match in the Configured group counts as a catalog match.
        assert _text(screen, "#settings-provider-search-status") == (
            "1 found · Enter picks · Esc cancels"
        )

        # The open list is still not a Tab stop, and an unfinished filter is
        # dropped rather than kept as the provider's name.
        await pilot.press("tab")
        await pilot.pause(0.2)
        assert host.focused is api_key
        assert not picker.display
        assert control.value == "Anthropic"

        control.focus()
        await pilot.pause()
        await pilot.press(*"local_ll")
        await pilot.pause()
        listed = {
            getattr(picker.get_option_at_index(index), "provider_id", None)
            for index in range(picker.option_count)
        } - {None}
        assert listed == {"local_llamacpp", "local_llamafile", "local_llm"}
        highlighted = picker.highlighted
        await pilot.press("down")
        assert picker.highlighted not in (None, highlighted)

        await pilot.press("escape")
        await pilot.pause()
        assert not picker.display
        assert control.value == "Anthropic"
        assert host.focused is control

        await pilot.press(*"llamaf", "enter")
        await pilot.pause(0.2)
        assert screen._provider_setting_values_mapping()["provider"] == (
            "local_llamafile"
        )
        assert control.value == "Llamafile"
        assert not picker.display
        assert _text(screen, "#settings-provider-source") == "edited *"


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
