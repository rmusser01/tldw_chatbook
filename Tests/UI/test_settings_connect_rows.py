"""Settings ▸ Providers & Models ▸ Connect, one row per fact (TASK-33007.2).

Mounted with the real application stylesheet at 211x44, the size the spec's
mockup (c) is drawn at, and driven by real keypresses.

TASK-33007.9 adds what the Provider control shows -- and paints -- after a
choice or a Revert, and that its open list is drawn whole inside the card,
which is also checked at 235x52.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.input import Selection

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness

_SIZE = (211, 44)
_FAKE_KEY = "sk-proj-abcdefghijklmnop1234"
#: Owner ruling 2026-10-04: Clear is this key on the API key field.
_CLEAR_KEY = "ctrl+l"
_CLEAR_HINT = f"(t) test · ({_CLEAR_KEY}) clear"


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
        # dropped rather than kept as the provider's name. For Anthropic the
        # next stop is TASK-34201's Sign in with row, just above the API key.
        await pilot.press("tab")
        await pilot.pause(0.2)
        assert host.focused is screen.query_one("#settings-provider-auth-source")
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

        # ADR-031 / task-1560: one Esc closes the list AND releases the field,
        # so the footer's "Esc, s save category" chain holds with the list open.
        assert ("Esc, s", "save category") in screen._footer_shortcut_entries()
        await pilot.press("escape")
        await pilot.pause()
        assert not picker.display
        assert control.value == "Anthropic"
        assert host.focused is None
        assert ("s", "save category") in screen._footer_shortcut_entries()

        control.focus()
        await pilot.pause()
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


def _app_mouse(host, cls, x: int, y: int, button: int):
    """Post a mouse event through the App, the way the terminal driver does,
    so App synthesizes the Click at MouseUp (pilot.click forwards it directly
    and cannot show a press that outlives a focus change)."""
    host.post_message(cls(None, x, y, 0, 0, button, False, False, False, x, y))


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("hold", [0.0, 0.3, 0.8])
async def test_a_held_mouse_press_on_the_open_list_still_chooses(request, hold):
    """Review I1: a press moves focus to the scrolling pane; the list must stay
    open until the release, which is when the choice lands."""
    from textual import events

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "anthropic", "model": "claude-x"}
    app.app_config["api_settings"] = {"anthropic": {"api_key": _FAKE_KEY}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)
        control.focus()
        await pilot.pause()
        await pilot.press(*"olla")
        await pilot.pause()
        rows = _region_rows(screen, picker)
        row = next(i for i, text in enumerate(rows) if text.strip() == "Ollama")
        x = picker.region.x + rows[row].index("Ollama") + 1
        y = picker.region.y + row

        _app_mouse(host, events.MouseMove, x, y, 0)
        _app_mouse(host, events.MouseDown, x, y, 1)
        await pilot.pause(hold)
        _app_mouse(host, events.MouseUp, x, y, 1)
        # The Click -> OptionSelected -> Select.Changed chain is posted after
        # pause()'s idle wait begins; give it time under xdist load.
        for _ in range(100):
            await pilot.pause(0.05)
            if screen._provider_setting_values_mapping()["provider"] == "ollama":
                break

        assert screen._provider_setting_values_mapping()["provider"] == "ollama"
        assert control.value == "Ollama"
        assert not picker.display


# Parent AC#2. Owner ruling 2026-10-04: Clear left the Tab chain (it is a key
# on the API key field), so a saved key adds no stop. These three providers
# each carry one conditional Connect stop and are pinned at their exact
# chains: a sixth stop, or a lost one, turns them red.
_CONNECT_STOPS = (
    "settings-provider-api-key",
    "settings-provider-credential-env-var",
    "settings-provider-endpoint-value",
)
_ANTHROPIC_STOPS = ("settings-provider-auth-source", *_CONNECT_STOPS)
_OPENAI_STOPS = (*_CONNECT_STOPS, "settings-openai-reconnect-review")
_QWENCLOUD_STOPS = (*_CONNECT_STOPS, "settings-provider-api-mode")
_SAVED_KEY = {"api_key": _FAKE_KEY}


async def _tab_stops_to_model(host, pilot, screen) -> list[str | None]:
    """Tab from the Provider control until the Model field holds focus.

    Args:
        host: The app under test.
        pilot: Its pilot.
        screen: The Settings screen showing Providers & Models.

    Returns:
        The id focused after each press; the last is the Model field's.
    """
    # TASK-33007.3: the Model stop is the Default model picker's field (the
    # Input behind it is a hidden adapter).
    model = screen.query_one("#model-search-picker-input", Input)
    screen.query_one("#settings-provider-search", Input).focus()
    await pilot.pause()
    stops: list[str | None] = []
    while host.focused is not model and len(stops) < 10:
        await pilot.press("tab")
        await pilot.pause()
        stops.append(getattr(host.focused, "id", None))
    return stops


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "settings", "stops_before_model"),
    [
        pytest.param("anthropic", {}, _ANTHROPIC_STOPS, id="anthropic-no-key"),
        pytest.param("openai", {}, _OPENAI_STOPS, id="openai-no-key"),
        pytest.param("qwencloud", {}, _QWENCLOUD_STOPS, id="qwencloud-no-key"),
        pytest.param(
            "anthropic", _SAVED_KEY, _ANTHROPIC_STOPS, id="anthropic-key-saved"
        ),
        pytest.param("openai", _SAVED_KEY, _OPENAI_STOPS, id="openai-key-saved"),
        pytest.param(
            "qwencloud", _SAVED_KEY, _QWENCLOUD_STOPS, id="qwencloud-key-saved"
        ),
    ],
)
async def test_model_is_at_most_five_tab_presses_from_provider(
    request, monkeypatch, provider, settings, stops_before_model
):
    """Parent AC#2: neither Test (t) nor Clear is a Tab stop -- a key runs
    each -- so Model stays within five presses of the Provider control, with
    OpenAI's "Review restored OpenAI connection" (AC#9), QwenCloud's API mode
    Select or Anthropic's Sign in with Select (TASK-34201) in the chain."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {provider: dict(settings)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        test_button = screen.query_one("#settings-test-provider", Button)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)

        stops = await _tab_stops_to_model(host, pilot, screen)

        assert stops == [*stops_before_model, "model-search-picker-input"]
        assert len(stops) <= 5
        assert test_button.display and not test_button.disabled
        assert clear.display and clear.disabled == (not settings)


def _cloud_provider_ids() -> list[str]:
    """List the providers the Settings picker groups under Cloud.

    Returns:
        The readiness key of every catalog provider that needs an API key and
        is not a custom slot or legacy alias.
    """
    from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    groups = build_provider_picker_groups(settings_provider_catalog(), "", "")
    return [
        str(option.provider_id)
        for group in groups
        if group.group_id == "cloud"
        for option in group.options
    ]


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("keyed", [False, True], ids=["no-key", "key-saved"])
async def test_every_cloud_provider_reaches_model_within_five_tab_presses(
    request, keyed
):
    """Parent AC#2 over the whole catalog: every Cloud provider, with and
    without a saved key, chosen in turn on one mounted card."""
    providers = _cloud_provider_ids()
    assert len(providers) >= 40, providers
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": providers[0], "model": "m-1"}
    app.app_config["api_settings"] = {
        provider: dict(_SAVED_KEY) if keyed else {} for provider in providers
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        # The hidden Select is the adapter the Provider control chooses through.
        adapter = screen.query_one("#settings-provider-value", Select)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        over_budget: dict[str, list[str | None]] = {}
        clearable: list[str] = []
        for provider in providers:
            adapter.value = screen._provider_select_value_for_provider(provider)
            # Select.Changed and the staging it drives can outlast one pause
            # under xdist load.
            for _ in range(100):
                await pilot.pause(0.05)
                if screen._provider_setting_values_mapping()["provider"] == provider:
                    break
            assert screen._provider_setting_values_mapping()["provider"] == provider
            if not clear.disabled:
                clearable.append(provider)
            stops = await _tab_stops_to_model(host, pilot, screen)
            if (
                stops[-1] != "model-search-picker-input"
                or len(stops) > 5
                or clear.id in stops
            ):
                over_budget[provider] = stops

        assert not over_budget, over_budget
        # The saved-key walk means something only while Clear is live in it.
        assert bool(clearable) is keyed, clearable


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
async def test_clear_is_not_a_tab_stop_stays_clickable_and_the_row_names_its_key(
    request, monkeypatch
):
    """Owner ruling 2026-10-04: like Test (t), Clear is a visible action that
    Tab skips; the API key row's help names both keys in full at 211x44."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        row = screen.query_one("#settings-provider-api-key-row")

        assert clear.display and not clear.disabled and not clear.can_focus
        assert _text(screen, "#settings-provider-api-key-help") == (
            f"masked · {_CLEAR_HINT}"
        )
        assert row.region.height == 1
        assert _CLEAR_HINT in "\n".join(_region_rows(screen, row))

        api_key.focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert host.focused is screen.query_one("#settings-provider-credential-env-var")

        await pilot.click("#settings-provider-api-key-clear")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "cleared *"
        assert host.focused is not clear


@pytest.mark.asyncio
@private_profile_test
async def test_provider_help_names_a_provider_once_its_save_configures_it(
    request, monkeypatch
):
    """Review finding 7: the "configured: ..." help is rebuilt after a save."""
    from Tests.UI.test_settings_configuration_hub import (
        _capture_provider_settings_mutations,
    )

    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {"llama_cpp": {}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        assert "llama.cpp" not in _text(screen, "#settings-provider-search-status")

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.focus()
        await pilot.pause()
        await pilot.press(*"http://127.0.0.1:9098", "escape", "s")
        await pilot.pause(0.2)

        assert len(mutations) == 1
        assert "configured: llama.cpp" in _text(
            screen, "#settings-provider-search-status"
        )


# --- TASK-33007.9: the control after a choice or Revert; the list's box ---


def _anthropic_host(provider: str = "anthropic"):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "claude-x"}
    app.app_config["api_settings"] = {"anthropic": {"api_key": _FAKE_KEY}}
    return _SettingsCssHarness(app, "settings")


async def _until(pilot, condition) -> None:
    """Pause until ``condition()`` holds; a chain of posted messages can
    outlast one pause under xdist load."""
    for _ in range(100):
        if condition():
            return
        await pilot.pause(0.05)


def _held(screen) -> str:
    return screen._provider_setting_values_mapping()["provider"]


async def _type_filter(pilot, screen, typed: str) -> None:
    """Type ``typed`` into the Provider control and wait for its list to
    open on it, so a following Enter cannot run ahead of the filter."""
    control = screen.query_one("#settings-provider-search", Input)
    picker = screen.query_one("#settings-provider-picker", OptionList)
    control.focus()
    await pilot.pause()
    await pilot.press(*typed)
    await _until(pilot, lambda: picker.display and control.value == typed)
    assert picker.display and control.value == typed
    await pilot.pause()


#: 33 cells, the control's whole text width at 211x44 and 235x52: no cell is
#: left for the caret Textual's Input keeps after the last character.
_OOBABOOGA = "Text Generation WebUI (Oobabooga)"
_WIDER_ENDPOINT_NAME = "GPU box in the back office, behind the VPN"


def _painted(screen, control) -> str:
    return _region_rows(screen, control)[0]


def _paints_from_its_head(screen, control, name: str) -> bool:
    """Whether the control paints ``name`` from its first cell, as much of
    it as fits. Only a name at least as wide as the text area can tell."""
    cells = control.content_region.width
    assert len(name) >= cells, (name, cells)
    return name[:cells] in _painted(screen, control)


async def _expect_the_head(pilot, screen, control, name: str) -> None:
    await _until(pilot, lambda: _paints_from_its_head(screen, control, name))
    assert _paints_from_its_head(screen, control, name), _painted(screen, control)


async def _tab_away(pilot, control) -> None:
    await pilot.press("tab")
    await _until(pilot, lambda: not control.has_focus)
    await pilot.pause(0.2)  # the control's blur timer


async def _window_away_and_back(pilot, host, control) -> None:
    """The terminal window loses focus for longer than the control's blur
    timer, then regains it."""
    host.post_message(events.AppBlur())
    await _until(pilot, lambda: not control.has_focus)
    await pilot.pause(0.2)  # the control's blur timer
    host.post_message(events.AppFocus())
    await _until(pilot, lambda: control.has_focus)
    await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_provider_control_names_the_choice_never_the_typed_filter(request):
    """TASK-33007.9 AC#1: whatever was typed to find it, the control then
    shows the chosen provider's display name -- for a legacy alias, for the
    provider it already holds (nothing is staged, so only closing the list
    can put the name back) and for "Enter provider ID"."""
    host = _anthropic_host()

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)
        legacy_name = "llama.cpp (legacy alias)"
        help_id = "#settings-provider-search-status"
        resting_help = _text(screen, help_id)
        assert resting_help.startswith("configured: Anthropic ·")

        await _type_filter(pilot, screen, "legacy")
        assert _text(screen, help_id) != resting_help
        await pilot.press("enter")
        await _until(
            pilot, lambda: _held(screen) == "local_llamacpp" and not picker.display
        )
        assert _held(screen) == "local_llamacpp"
        assert control.value == legacy_name
        assert not picker.display
        assert _text(screen, help_id) == resting_help

        await _type_filter(pilot, screen, "llama")
        highlighted = picker.get_option_at_index(picker.highlighted)
        assert getattr(highlighted, "provider_id", None) == "local_llamacpp"
        await pilot.press("enter")
        await _until(pilot, lambda: not picker.display)
        assert _held(screen) == "local_llamacpp"
        assert control.value == legacy_name
        assert not picker.display
        assert _text(screen, help_id) == resting_help

        # Only "Enter provider ID" is left to choose.
        await _type_filter(pilot, screen, "zzzz")
        assert "zzzz" in _text(screen, help_id)
        await pilot.press("enter")
        manual = screen.query_one("#settings-provider-manual-value", Input)
        await _until(pilot, lambda: host.focused is manual and not picker.display)
        assert host.focused is manual
        assert control.value == legacy_name
        assert not picker.display
        assert _text(screen, help_id) == resting_help


@pytest.mark.asyncio
@private_profile_test
async def test_choosing_the_held_provider_by_its_exact_name_selects_it(request):
    """TASK-33007.9 (review round 4): Enter on the held provider's exact
    name rewrites nothing, yet the name is still selected, so the next key
    filters afresh instead of making "Anthropicx"."""
    host = _anthropic_host()

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)

        await _type_filter(pilot, screen, "Anthropic")
        await pilot.press("enter")
        await _until(pilot, lambda: not picker.display)
        assert _held(screen) == "anthropic"
        await pilot.press("x")
        await _until(pilot, lambda: control.value == "x")
        assert control.value == "x"


@pytest.mark.asyncio
@private_profile_test
async def test_a_provider_name_typed_exactly_is_the_one_chosen(request):
    """TASK-33007.9 (review round 4): Enter on "OpenAI" chooses OpenAI, not
    "Azure OpenAI", which the list shows above it."""
    host = _anthropic_host()

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)

        await _type_filter(pilot, screen, "OpenAI")
        await pilot.press("enter")
        await _until(pilot, lambda: _held(screen) != "anthropic" and not picker.display)
        assert _held(screen) == "openai"
        assert control.value == "OpenAI"


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("held", ["anthropic", "mistral"])
async def test_typing_a_legacy_alias_id_chooses_the_canonical_provider(request, held):
    """TASK-33007.9 (review rounds 5 and 6): "Mistral" is the legacy alias
    row's id, yet Enter chooses Mistral AI, not "Mistral AI (legacy alias)"
    listed last -- also when the alias is the held provider."""
    host = _anthropic_host(held)

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        picker = screen.query_one("#settings-provider-picker", OptionList)

        await _type_filter(pilot, screen, "Mistral")
        await pilot.press("enter")
        await _until(pilot, lambda: _held(screen) != held and not picker.display)
        assert _held(screen) == "mistralai"


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("revert_with", ["r", "button"])
async def test_discard_changes_shows_the_saved_provider_and_no_filtered_list(
    request, revert_with
):
    """TASK-33007.9 AC#2: r, Discard changes puts the saved provider's name
    back and leaves no filtered list open -- also when the Inspector's
    Revert (r) is clicked while the list is still open on a filter."""
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    host = _anthropic_host()

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)
        whole_list = picker.option_count

        await _type_filter(pilot, screen, "llamaf")
        await pilot.press("enter")
        await _until(pilot, lambda: _held(screen) == "local_llamafile")
        assert control.value == "Llamafile"
        # A second, unfinished filter over the staged choice.
        await _type_filter(pilot, screen, "oll")
        assert picker.option_count < whole_list

        if revert_with == "r":
            await pilot.press("escape", "r")
        else:
            await pilot.click("#settings-revert-category")
        await _until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert isinstance(host.screen, ConfirmationDialog)
        if revert_with == "r":
            await pilot.press("tab", "enter")  # Keep editing holds focus first
        else:
            await pilot.click("#confirm-button")
        # [button]: the list is closed by the control's blur timer, which can
        # land after the revert, so wait for the name as well.
        await _until(
            pilot,
            lambda: (
                host.screen is screen
                and _held(screen) == "anthropic"
                and not picker.display
                and control.value == "Anthropic"
            ),
        )

        assert _held(screen) == "anthropic"
        assert control.value == "Anthropic"
        assert not picker.display
        assert _text(screen, "#settings-provider-source") == "new-chat default"
        assert "configured: Anthropic ·" in _text(
            screen, "#settings-provider-search-status"
        )

        # Opened again, the list is whole and rests on the saved provider.
        control.focus()
        await pilot.pause()
        await pilot.press("down")
        await _until(pilot, lambda: picker.display)
        assert picker.display and picker.option_count == whole_list
        highlighted = picker.get_option_at_index(picker.highlighted)
        assert getattr(highlighted, "provider_id", None) == "anthropic"


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("size", [_SIZE, (235, 52)], ids=["211x44", "235x52"])
async def test_open_provider_list_is_drawn_whole_inside_the_card(request, size):
    """TASK-33007.9 AC#3: the open list's whole box, its scrollbar at the
    right edge included, lies inside the card and nothing clips it. The list
    draws no border of its own (spec: one frame level), so the box is all
    there is to clip."""
    host = _anthropic_host()

    async with host.run_test(size=size) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)
        card = screen.query_one("#settings-providers-models-card")
        control.focus()
        await pilot.pause()

        def drawn_whole(*parts) -> None:
            for part in parts:
                box = part.region
                assert box.area > 0, part
                assert card.content_region.contains_region(box), (part, box)
                assert screen.find_widget(part).visible_region == box, part

        await pilot.press("down")  # every provider: the list scrolls
        await _until(pilot, lambda: picker.display and picker.max_scroll_y > 0)
        await pilot.pause()
        assert picker.display and picker.max_scroll_y > 0
        drawn_whole(picker, picker.vertical_scrollbar)

        await pilot.press(*"anthro")  # a short list: no scrollbar
        await _until(
            pilot, lambda: control.value == "anthro" and picker.max_scroll_y == 0
        )
        await pilot.pause()
        assert picker.display and control.value == "anthro"
        assert picker.max_scroll_y == 0
        drawn_whole(picker)


@pytest.mark.asyncio
@private_profile_test
async def test_a_chosen_name_as_wide_as_the_control_is_painted_from_its_head(request):
    """TASK-33007.9 AC#1 (review I1): Input keeps a cell for the caret after
    the last character and scrolls to it, so a select-all that ends there
    pushed the head of a name as wide as the control out of view ("ext
    Generation WebUI (Oobabooga)"). Chosen, left and Tabbed back into, the
    control paints the name from its first cell -- and selects it from there
    when the terminal window regains focus (review round 2)."""
    host = _anthropic_host()

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)

        await _type_filter(pilot, screen, "oobab")
        await pilot.press("enter")
        await _until(pilot, lambda: _held(screen) == "oobabooga" and not picker.display)
        assert control.value == _OOBABOOGA and control.has_focus
        await pilot.pause()
        await _expect_the_head(pilot, screen, control, _OOBABOOGA)

        await _tab_away(pilot, control)
        await _expect_the_head(pilot, screen, control, _OOBABOOGA)
        await pilot.press("shift+tab")
        await _until(pilot, lambda: control.has_focus)
        await pilot.pause()
        await _expect_the_head(pilot, screen, control, _OOBABOOGA)

        # The caret is the user's to move while the field is theirs: at the
        # end it takes its cell and the row scrolls by one. Left there, the
        # name still comes to rest on its head.
        await pilot.press("end")
        await pilot.pause()
        assert not _paints_from_its_head(screen, control, _OOBABOOGA)
        await _tab_away(pilot, control)
        await _expect_the_head(pilot, screen, control, _OOBABOOGA)

        # Review round 2: Input keeps the caret when the window regains focus,
        # but this control's blur has put the name back by then. So that focus
        # selects the name from its head like any other -- with the caret left
        # at the end, where the field otherwise showed no caret at all ...
        whole_name = Selection(len(_OOBABOOGA), 0)
        await pilot.press("shift+tab")
        await _until(pilot, lambda: control.has_focus)
        await pilot.press("end")
        await pilot.pause()
        await _window_away_and_back(pilot, host, control)
        assert control.has_focus and not picker.display
        assert control.selection == whole_name
        await _expect_the_head(pilot, screen, control, _OOBABOOGA)

        # ... and after a filter was left typed, so the next key replaces the
        # name instead of landing inside it ("Texzt Generation WebUI").
        await _type_filter(pilot, screen, "oll")
        await _window_away_and_back(pilot, host, control)
        assert control.value == _OOBABOOGA and not picker.display
        assert control.selection == whole_name
        await pilot.press("z")
        await _until(pilot, lambda: control.value == "z" and picker.display)
        assert control.value == "z" and picker.display

        # Away for less than the blur timer (both events queued at once), the
        # list is still open and the filter is still the user's to finish.
        host.post_message(events.AppBlur())
        host.post_message(events.AppFocus())
        await pilot.pause(0.3)
        assert control.has_focus and picker.display
        assert control.value == "z"
        assert control.selection == Selection.cursor(1)


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("saved", "name"),
    [
        pytest.param("oobabooga", _OOBABOOGA, id="as-wide-as-the-control"),
        pytest.param(
            "custom-ep:gpu-box", _WIDER_ENDPOINT_NAME, id="wider-registry-name"
        ),
    ],
)
async def test_a_saved_wide_provider_name_is_painted_from_its_head(
    request, saved, name
):
    """TASK-33007.9 AC#1, AC#2 (review I1): a saved name as wide as the
    control, or a registry name wider than it, is painted from its first
    cell at rest and once the control is Tabbed or clicked into."""
    app = _build_test_app()
    app.app_config["custom_endpoints"] = {
        "gpu-box": {
            "display_name": _WIDER_ENDPOINT_NAME,
            "family": "llama_cpp",
            "base_url": "http://192.168.1.5:8080",
            "models": ["model-a"],
        }
    }
    app.app_config["chat_defaults"] = {"provider": saved, "model": "model-a"}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        assert control.value == name

        await _expect_the_head(pilot, screen, control, name)  # as mounted

        control.focus()
        await pilot.pause()
        await _tab_away(pilot, control)
        await _expect_the_head(pilot, screen, control, name)
        await pilot.press("shift+tab")
        await _until(pilot, lambda: control.has_focus)
        await pilot.pause()
        await _expect_the_head(pilot, screen, control, name)

        await _tab_away(pilot, control)
        await pilot.click("#settings-provider-search")
        await _until(pilot, lambda: control.has_focus)
        await pilot.pause()
        await _expect_the_head(pilot, screen, control, name)
