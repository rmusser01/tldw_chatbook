"""Switch model's other ways in, Ctrl+O, and the rail Model section.

TASK-33004.7. Every path drives the shipping Console screen with real key
presses from the composer (``private_profile_test``: a scratch config the
real writers own, as in ``test_console_switch_model_keys``):

- ``/model`` opens Switch model; ``/model son`` opens it with ``son`` in Find
  and the best match highlighted, and nothing applies until Enter (AC#1).
- The rail Model section shows Streaming On/Off and keeps it current through
  Apply, a new chat and switching back (AC#2, task-338), within ADR-083's
  15-row cap (AC#4), and its action reads "Change  Alt+M" (AC#3).
- Ctrl+O opens Chat settings (AC#5), and every surface that teaches Alt+M or
  Ctrl+O names a key that works (AC#6, ADR-031 rule 4).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.console_rail_section_helpers import open_rail_section
from Tests.UI.test_console_switch_model_keys import (
    _composer,
    _console_app,
    _drain,
    _Harness,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_WORKBENCH_SHORTCUT_GROUPS,
    CONSOLE_WORKBENCH_SHORTCUTS,
    ChatScreen,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal

SWITCH_MODEL = "Switch model"
CHAT_SETTINGS = "Chat settings"


async def _type(pilot, text: str) -> None:
    for char in text:
        await pilot.press("space" if char == " " else char)
    await pilot.pause()


async def _until(pilot, predicate, what: str) -> None:
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def _open_console(harness: _Harness, pilot) -> ChatScreen:
    console = harness.screen
    assert isinstance(console, ChatScreen)
    await _wait_for_selector(console, pilot, "#console-native-composer")
    await _wait_for_selector(console, pilot, "#console-provider-chip")
    _composer(console).focus()
    await pilot.pause()
    return console


async def _switcher_ready(harness: _Harness, pilot) -> ConsoleModelPopover:
    await _until(
        pilot,
        lambda: isinstance(harness.screen, ConsoleModelPopover),
        "Switch model to open",
    )
    await harness.workers.wait_for_complete()
    await pilot.pause()
    return harness.screen


async def _close_chat_settings(harness: _Harness, console: ChatScreen, pilot) -> None:
    # Cancel, not Esc: Chat settings opens with focus in its provider
    # search, which keeps Esc for itself (that modal is P6's surface).
    await pilot.pause()
    harness.screen.query_one("#console-settings-cancel", Button).press()
    await _until(pilot, lambda: harness.screen is console, "Chat settings to close")


def _rail_value(console: ChatScreen, row: str) -> str:
    value = console.query_one(
        f"#console-model-section-{row} .console-model-section-value", Static
    )
    return str(value.render()).strip()


@pytest.mark.asyncio
@private_profile_test
async def test_model_query_opens_switch_model_with_find_filled(request) -> None:
    """AC#1: `/model son` puts `son` in Find and highlights the best match;
    nothing applies until Enter. `/model` alone opens it with Find empty."""
    app = _console_app()
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _open_console(harness, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        def pair() -> tuple[str, str | None]:
            settings = store.session_settings(session_id)
            return (settings.provider, settings.model)

        revision = store.session_settings_revision(session_id)
        await _type(pilot, "/model son")
        await pilot.press("enter")
        switcher = await _switcher_ready(harness, pilot)
        assert switcher.query_one("#console-popover-find", Input).value == "son"
        assert _composer(console).draft_text() == ""  # the command was consumed
        assert harness.focused is switcher.query_one("#console-popover-find")
        row = switcher.highlighted_row()
        assert row is not None
        assert (row.provider, row.model) == ("anthropic", "claude-sonnet-4-5")
        # Highlighting rebased only the switcher's draft; the chat is untouched.
        assert pair() == ("llama_cpp", "model-a")
        assert store.session_settings_revision(session_id) == revision

        await pilot.press("enter")
        await _drain(app)
        await _until(pilot, lambda: harness.screen is console, "the switcher to close")
        assert pair() == ("anthropic", "claude-sonnet-4-5")

        _composer(console).focus()
        await _type(pilot, "/model")
        # The first Enter accepts the open suggestion ("/model "); the second sends.
        await pilot.press("enter")
        assert _composer(console).draft_text() == "/model "
        await pilot.press("enter")
        switcher = await _switcher_ready(harness, pilot)
        assert switcher.query_one("#console-popover-find", Input).value == ""
        await pilot.press("escape")
        await _until(pilot, lambda: harness.screen is console, "Esc to close it")
        assert pair() == ("anthropic", "claude-sonnet-4-5")


@pytest.mark.asyncio
@private_profile_test
async def test_ctrl_o_and_the_rail_change_action_open_their_surfaces(request) -> None:
    """AC#3, AC#5, AC#6: Ctrl+O opens Chat settings from the composer and from
    a focused Provider/Model chip; the rail's "Change  Alt+M" opens Switch
    model; the chips' Enter and the footer's Alt+M/Ctrl+O all work."""
    app = _console_app()
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _open_console(harness, pilot)

        await pilot.press("ctrl+o")
        await _until(
            pilot,
            lambda: isinstance(harness.screen, ConsoleSettingsModal),
            "Ctrl+O to open Chat settings",
        )
        await _close_chat_settings(harness, console, pilot)

        for chip_id in ("console-provider-chip", "console-model-chip"):
            chip = console.query_one(f"#{chip_id}", Static)
            tooltip = str(chip.tooltip)
            assert f"{SWITCH_MODEL} (Alt+M)" in tooltip, tooltip
            assert f"{CHAT_SETTINGS} (Ctrl+O)" in tooltip, tooltip
            chip.focus()
            await pilot.pause()
            await pilot.press("enter")
            await _switcher_ready(harness, pilot)
            await pilot.press("escape")
            await _until(pilot, lambda: harness.screen is console, "the switcher to close")
            chip.focus()
            await pilot.pause()
            await pilot.press("ctrl+o")
            await _until(
                pilot,
                lambda: isinstance(harness.screen, ConsoleSettingsModal),
                f"Ctrl+O from #{chip_id}",
            )
            await _close_chat_settings(harness, console, pilot)

        await open_rail_section(console, pilot, "model")
        change = console.query_one("#console-model-section-configure", Button)
        assert str(change.label) == "Change  Alt+M"
        assert SWITCH_MODEL in str(change.tooltip)
        change.press()
        await _switcher_ready(harness, pilot)
        await pilot.press("escape")
        await _until(pilot, lambda: harness.screen is console, "the switcher to close")

        footer = console.query_one(AppFooterStatus)
        shown = str(footer._shortcut_display.content)
        assert "Alt+M switch model" in shown and "Ctrl+O chat settings" in shown
        # TASK-33004.7 review: at the owner's 211x44 the new model hints must
        # not push the Inspect/approval/context-rail accelerators out.
        for hint in ("Alt+I inspect", "Alt+A approval", "Alt+C context rail"):
            assert hint in shown, (hint, shown)


@pytest.mark.asyncio
@private_profile_test
async def test_rail_streaming_row_follows_apply_new_chat_and_switching_back(
    request,
) -> None:
    """AC#2 (task-338 AC#1/#2): Streaming On/Off is on the rail and follows
    the chat through Apply, a new chat and switching back to the first."""
    app = _console_app()
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _open_console(harness, pilot)
        store = console._ensure_console_chat_store()
        first = store.active_session_id
        await open_rail_section(console, pilot, "model")
        assert _rail_value(console, "streaming") == "On"

        # Apply: Streaming Off on this chat's own pair, from the keyboard.
        _composer(console).focus()
        await pilot.press("alt+m")
        switcher = await _switcher_ready(harness, pilot)
        row = switcher.highlighted_row()
        assert (row.provider, row.model) == ("llama_cpp", "model-a")
        await pilot.press("tab", "tab", "tab")
        assert harness.focused is switcher.query_one("#console-popover-streaming")
        await pilot.press("enter", "down", "enter", "shift+tab", "enter")
        await _drain(app)
        await _until(pilot, lambda: harness.screen is console, "Apply to close it")
        assert store.session_settings(first).streaming is False
        await _until(
            pilot, lambda: _rail_value(console, "streaming") == "Off", "rail Off"
        )

        # A new chat starts from the defaults: Streaming On.
        _composer(console).focus()
        await pilot.press("ctrl+t")
        await _until(
            pilot, lambda: store.active_session_id != first, "Ctrl+T to open a chat"
        )
        assert store.session_settings(store.active_session_id).streaming is True
        await _until(
            pilot, lambda: _rail_value(console, "streaming") == "On", "rail On"
        )

        # Switching back to the first chat (its tab) shows its Off again.
        await console._session._activate_native_console_session(first)
        await _until(
            pilot, lambda: _rail_value(console, "streaming") == "Off", "rail Off again"
        )


def test_footer_f1_palette_and_bindings_teach_keys_that_work():
    """AC#6 and ADR-031 rule 4: the footer, F1 help and the palette name
    Switch model (Alt+M) and Chat settings (Ctrl+O), each key is a Console
    binding to the action that opens that surface, and F1 teaches /model."""
    from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider

    keys = [key for key, _label in CONSOLE_WORKBENCH_SHORTCUTS]
    # Mockup (a)'s order: right after F1 help, ahead of the hints that drop
    # first when the footer runs out of width.
    assert keys[keys.index("F1") + 1 : keys.index("F1") + 3] == ["Alt+M", "Ctrl+O"]
    footer = dict(CONSOLE_WORKBENCH_SHORTCUTS)
    assert footer["Alt+M"] == "switch model"
    assert footer["Ctrl+O"] == "chat settings"

    help_rows = {
        key: label
        for _title, rows in CONSOLE_WORKBENCH_SHORTCUT_GROUPS
        for key, label in rows
    }
    assert SWITCH_MODEL in help_rows["Alt+M"]
    assert CHAT_SETTINGS in help_rows["Ctrl+O"]
    assert "/model [query]" in help_rows

    bindings = {
        binding.key: binding.action
        for binding in ChatScreen.BINDINGS
        if hasattr(binding, "key")
    }
    assert bindings["alt+m"] == "open_console_model_popover"
    assert bindings["ctrl+o"] == "open_console_session_settings"

    class _Screen:
        def __getattr__(self, name):
            if name.startswith("action_"):
                return name
            raise AttributeError(name)

    palette = {
        label: (action, help_text)
        for label, action, help_text in ConsoleCommandProvider._commands(
            None, _Screen()
        )
    }
    action, help_text = palette["Console: Switch model…"]
    assert action == "action_open_console_model_popover" and "(Alt+M)" in help_text
    action, help_text = palette["Console: Chat settings…"]
    assert action == "action_open_console_session_settings"
    assert "(Ctrl+O)" in help_text
    assert not any("Change model" in label for label in palette)
