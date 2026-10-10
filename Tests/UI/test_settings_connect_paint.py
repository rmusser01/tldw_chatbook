"""Settings ▸ Providers & Models ▸ Connect: what is painted whole (TASK-33007.9).

Split out of test_settings_connect_rows.py so the UI Fast Lane's round-robin
shards can spread the Connect cases (each mounts the whole Settings screen,
~16 s a case on CI). The open provider list's box, a wide provider name
painted from its head, and rows painted whole at both full-screen sizes.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Input, OptionList, Select, Static
from textual.widgets.input import Selection

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import (
    _SAVED_KEY,
    _SIZE,
    _anthropic_host,
    _held,
    _open_providers,
    _squeezed,
    _text,
    _type_filter,
    _until,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness

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


_SIZES = pytest.mark.parametrize("size", [_SIZE, (235, 52)], ids=["211x44", "235x52"])


@pytest.mark.asyncio
@private_profile_test
@_SIZES
async def test_provider_help_counts_configured_providers_whole(request, size):
    """Captures 01 and 03 (fix 5): "configured: Anthropic, Azure OpenAI +1 ·…"
    was cut at 211, and its whole form at 235, "+1 · 57 more", read as one
    more configured provider and then 57 more providers."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "m-1"}
    app.app_config["api_settings"] = {
        provider: dict(_SAVED_KEY) for provider in ("openai", "anthropic", "azure")
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        screen = await _open_providers(host, pilot)
        help_text = _text(screen, "#settings-provider-search-status")
        total = sum(
            len(group.options)
            for group in screen._provider_picker_groups()
            if group.group_id not in {"actions", "saved"}
        )

        assert help_text == f"3 of {total} configured · listed first"
        painted = _region_rows(screen, screen.query_one("#settings-provider-row"))[0]
        assert help_text in painted, painted


@pytest.mark.asyncio
@private_profile_test
@_SIZES
@pytest.mark.parametrize("state", ["saved", "missing", "subscription"])
async def test_api_key_placeholder_is_painted_whole(request, monkeypatch, size, state):
    """Captures 01, 01b and 01c (fix 6): the saved-key placeholder was cut to
    "Local config key saved;" at both sizes."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "anthropic", "model": "m-1"}
    anthropic = dict(_SAVED_KEY) if state == "saved" else {}
    if state == "subscription":
        anthropic["auth_source"] = "claude_subscription"
    app.app_config["api_settings"] = {"anthropic": anthropic}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        screen = await _open_providers(host, pilot)
        api_key = screen.query_one("#settings-provider-api-key", Input)

        assert api_key.placeholder
        assert len(api_key.placeholder) <= api_key.content_region.width
        assert api_key.placeholder in _region_rows(screen, api_key)[0]


@pytest.mark.asyncio
@private_profile_test
@_SIZES
async def test_sign_in_with_is_one_row_with_a_source_word_and_help(request, size):
    """Capture 01b (fix 7): Sign in with had no Source word, and its help
    took a second row even at 235. Its long copy is the Inspector's now."""
    from tldw_chatbook.UI.Screens.settings_screen import (
        ANTHROPIC_API_KEY_GUIDANCE_COPY,
        ANTHROPIC_SUBSCRIPTION_GUIDANCE_COPY,
    )

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "anthropic", "model": "m-1"}
    app.app_config["api_settings"] = {"anthropic": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        screen = await _open_providers(host, pilot)
        row = screen.query_one("#settings-provider-auth-source-row")
        selector = screen.query_one("#settings-provider-auth-source", Select)
        help_line = screen.query_one("#settings-provider-auth-source-guidance", Static)

        assert row.region.height == 1
        assert help_line.parent is row
        assert screen.query_one("#settings-provider-api-key-row").region.y == (
            row.region.bottom
        )
        assert _text(screen, "#settings-provider-auth-source-word") == "built-in"
        assert _text(screen, "#settings-provider-auth-source-guidance") == (
            "bills API credits through your key"
        )
        assert "bills API credits through your key" in _region_rows(screen, row)[0]

        selector.focus()
        await pilot.pause()
        assert _text(screen, "#settings-provider-field-guide-0") == (
            "Focused setting: Sign in with"
        )
        assert _squeezed(_text(screen, "#settings-provider-field-guide-1")) == (
            _squeezed(f"Purpose: {ANTHROPIC_API_KEY_GUIDANCE_COPY}")
        )

        selector.value = "claude_subscription"
        screen.handle_provider_auth_source_changed(
            Select.Changed(selector, "claude_subscription")
        )
        await pilot.pause()
        assert row.region.height == 1
        assert _text(screen, "#settings-provider-auth-source-word") == "edited *"
        help_text = "bills your Claude plan, not API credits"
        assert _text(screen, "#settings-provider-auth-source-guidance") == help_text
        assert help_text in _region_rows(screen, row)[0]
        # The Inspector folds the long credential path at its separators.
        assert _squeezed(_text(screen, "#settings-provider-field-guide-1")) == (
            _squeezed(f"Purpose: {ANTHROPIC_SUBSCRIPTION_GUIDANCE_COPY}")
        )
