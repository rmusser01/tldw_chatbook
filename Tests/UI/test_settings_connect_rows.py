"""Settings ▸ Providers & Models ▸ Connect, one row per fact (TASK-33007.2).

Mounted with the real application stylesheet at 211x44, the size the spec's
mockup (c) is drawn at, and driven by real keypresses.

TASK-33007.9 adds what the Provider control shows -- and paints -- after a
choice or a Revert, and that its open list is drawn whole inside the card,
which is also checked at 235x52.

The Tab budget to Model, the key and endpoint rows, and what is painted
whole live in test_settings_connect_tab_budget.py, _key_rows.py and
_paint.py: each case mounts the whole Settings screen (~16 s on CI), and one
file of all of them outgrew a UI Fast Lane shard's 20 minutes. Their shared
helpers stay here.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Input, OptionList, Static

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


def _resting_help_counts(help_text: str, configured: int) -> bool:
    """Whether the Provider row's resting help counts ``configured`` providers.

    TASK-33007 capture fix 5, rewritten on purpose: the help counts the
    configured providers instead of naming them ("configured: Anthropic,
    Azure OpenAI +1 · 57 more" was cut at 211 and ambiguous at 235).
    """
    if configured == 0:
        return help_text.startswith("none of ") and help_text.endswith(
            " configured yet"
        )
    return help_text.startswith(f"{configured} of ") and help_text.endswith(
        " configured · listed first"
    )


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
        assert _resting_help_counts(
            _text(screen, "#settings-provider-search-status"), 1
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


_SAVED_KEY = {"api_key": _FAKE_KEY}


@pytest.mark.asyncio
@private_profile_test
async def test_provider_help_names_a_provider_once_its_save_configures_it(
    request, monkeypatch
):
    """Review finding 7: the configured count in the help is rebuilt after a
    save (it named the providers until TASK-33007 capture fix 5)."""
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
        assert _resting_help_counts(
            _text(screen, "#settings-provider-search-status"), 0
        )

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.focus()
        await pilot.pause()
        await pilot.press(*"http://127.0.0.1:9098", "escape", "s")
        await pilot.pause(0.2)

        assert len(mutations) == 1
        assert _resting_help_counts(
            _text(screen, "#settings-provider-search-status"), 1
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
        assert _resting_help_counts(resting_help, 1)

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
        assert _resting_help_counts(
            _text(screen, "#settings-provider-search-status"), 1
        )

        # Opened again, the list is whole and rests on the saved provider.
        control.focus()
        await pilot.pause()
        await pilot.press("down")
        await _until(pilot, lambda: picker.display)
        assert picker.display and picker.option_count == whole_list
        highlighted = picker.get_option_at_index(picker.highlighted)
        assert getattr(highlighted, "provider_id", None) == "anthropic"


def _squeezed(text: str) -> str:
    return "".join(text.split())


@pytest.mark.asyncio
@private_profile_test
async def test_provider_and_model_read_edited_only_when_their_own_value_is(request):
    """Captures 03 and 04 (fix 2): the draft pins provider and model beside
    any edit, so staging only a Temperature made both rows read edited *,
    and staging only a model made Provider read it too."""
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        temperature = screen.query_one("#settings-model-profile-temperature", Input)
        temperature.focus()
        await pilot.pause()
        await pilot.press(*"0.4")
        await pilot.pause()

        assert screen._category_has_unsaved_changes(SettingsCategoryId.PROVIDERS_MODELS)
        assert _text(screen, "#settings-model-profile-temperature-source") == "edited *"
        assert _text(screen, "#settings-provider-source") == "new-chat default"
        assert _text(screen, "#settings-model-source") == "new-chat default"

        screen.query_one("#settings-model-value", Input).value = "gpt-4o"
        await pilot.pause()

        assert _text(screen, "#settings-provider-source") == "new-chat default"
        assert _text(screen, "#settings-model-source") == "edited *"


@pytest.mark.asyncio
@private_profile_test
async def test_a_typed_name_prefix_highlights_the_provider_it_starts(request):
    """Checkpoint review (rebased captures): typing "llama" listed Ollama Cloud
    first (Cloud groups lead) and Enter chose it, though the user was typing
    llama.cpp's name. A provider whose name or id starts with the typed text
    now outranks one that only contains it; the held provider still wins when
    it matches that way (the legacy-alias case above)."""
    host = _anthropic_host("openai")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        picker = screen.query_one("#settings-provider-picker", OptionList)

        await _type_filter(pilot, screen, "llama")
        highlighted = picker.get_option_at_index(picker.highlighted)
        assert getattr(highlighted, "provider_id", None) == "llama_cpp"
        await pilot.press("enter")
        await _until(pilot, lambda: not picker.display)
        assert _held(screen) == "llama_cpp"
