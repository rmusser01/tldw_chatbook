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

# The harness mounts the real ChatScreen/config chain; per-test env redirect
# trips the config-participant admission (sibling strip suites' opt-in).
pytestmark = pytest.mark.bootstrap_profile
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
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleStatusText,
)

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
    await _wait_for_selector(console, pilot, "#console-status-text")
    # TASK-35160: the provider chip is the status line's "provider" segment
    # now; ready means the provider label is on the line.
    strip = console.query_one("#console-status-chips", ConsoleStatusChips)
    await _until(
        pilot,
        lambda: "Provider:" in strip.segment_text("provider"),
        "the provider segment to render",
    )
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
    # Rewritten on purpose by TASK-33007 capture finding 20 (round 2): no
    # view has a Cancel button any more; Esc close is the close.
    await pilot.pause()
    await pilot.press("escape")
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
    """AC#3, AC#5, AC#6: Ctrl+O opens Chat settings from the composer and
    from a focused provider/model segment of the status line; the rail's
    "Change  Alt+M" opens Switch model; the segments' Enter and the footer's
    Alt+M/Ctrl+O all work."""
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

        # TASK-35160: the provider/model chips are segments of one focusable
        # status line. Per-segment tooltips are gone -- the Alt+M/Ctrl+O
        # teaching lives on the footer (asserted below) -- and the Enter /
        # Ctrl+O routes walk the line's own segment cursor instead of
        # focusing a chip.
        for segment_name in ("provider", "model"):
            line = console.query_one("#console-status-text", ConsoleStatusText)
            line.focus()
            await pilot.pause()
            for _ in range(len(line.segment_spans)):
                if line.segment_spans[line._cursor][0] == segment_name:
                    break
                await pilot.press("right")
            assert line.segment_spans[line._cursor][0] == segment_name
            await pilot.press("enter")
            await _switcher_ready(harness, pilot)
            await pilot.press("escape")
            await _until(pilot, lambda: harness.screen is console, "the switcher to close")
            line.focus()
            await pilot.pause()
            await pilot.press("ctrl+o")
            await _until(
                pilot,
                lambda: isinstance(harness.screen, ConsoleSettingsModal),
                f"Ctrl+O from the {segment_name} segment",
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

        # The footer is read with the composer focused (capture finding 20,
        # round 2). Where focus ends after the rail's Change depended on
        # whether the Model section was already open, and with the rail
        # focused the footer leads "Esc composer · F6 panes", which pushes
        # Alt+A out at 211 columns: a footer width outside this test.
        _composer(console).focus()
        footer = console.query_one(AppFooterStatus)
        await _until(
            pilot,
            lambda: not str(footer._shortcut_display.content).startswith("Esc"),
            "the composer's footer",
        )
        shown = str(footer._shortcut_display.content)
        assert "Alt+M switch model" in shown and "Ctrl+O chat settings" in shown
        # TASK-33004.7 review: at the owner's 211x44 the new model hints must
        # not push the Inspect/approval/context-rail accelerators out.
        for hint in ("Alt+I inspect", "Alt+A approval", "Alt+C context rail"):
            assert hint in shown, (hint, shown)


@pytest.mark.asyncio
@private_profile_test
async def test_chat_settings_closed_by_esc_leaves_the_focused_widgets_footer(
    request,
) -> None:
    """TASK-33006.20: after Esc closes Chat settings, the footer is the one for
    the widget focus went back to. Opened from a rail control it leads with
    the rail's exits; opened from the composer it is the composer's, and at
    211x44 that keeps Alt+A and Alt+C, not the rail footer it had before."""
    app = _console_app()
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _open_console(harness, pilot)
        footer = console.query_one(AppFooterStatus)

        def shown() -> str:
            return str(footer._shortcut_display.content)

        await open_rail_section(console, pilot, "model")
        toggle = console.query_one("#console-rail-section-toggle-model", Button)
        for start, in_rail in ((toggle, True), (_composer(console), False)):
            start.focus()
            await pilot.pause()
            await pilot.press("ctrl+o")
            await _until(
                pilot,
                lambda: isinstance(harness.screen, ConsoleSettingsModal),
                "Ctrl+O to open Chat settings",
            )
            await _close_chat_settings(harness, console, pilot)
            assert harness.focused is start
            await _until(
                pilot,
                lambda in_rail=in_rail: (
                    shown().startswith("Esc composer · F6 panes") is in_rail
                    and "Alt+M switch model" in shown()
                ),
                f"the footer of {start.id}",
            )
        for hint in ("Alt+A approval", "Alt+C context rail"):
            assert hint in shown(), (hint, shown())


@pytest.mark.asyncio
@private_profile_test
async def test_a_dialog_over_a_ready_console_names_only_the_keys_that_act(
    request,
) -> None:
    """Capture fix 3 round 1 finding 2 (05a, 05c, 05d): under Switch model over
    a ready Console the footer still named every Console key, and none acts
    under a dialog. While one covers the Console the footer names only Ctrl+P
    and Ctrl+Q (a re-registration included): each key it leaves out is
    pressed and changes nothing. The Console's keys are back when it closes."""
    app = _console_app()
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = await _open_console(harness, pilot)
        drawn = console.query_one(AppFooterStatus).query_one("#footer-key-quit", Static)
        await _until(
            pilot, lambda: "Alt+M switch model" in str(drawn.render()), "the footer"
        )

        await pilot.press("alt+m")
        switcher = await _switcher_ready(harness, pilot)
        console._register_console_footer_shortcuts()  # a sync while covered
        await pilot.pause()
        drawn_text = str(drawn.render())
        hinted = {
            hint.split(" ", 1)[0]
            for part in drawn_text.split(" | ")
            for hint in part.split(" · ")
        }
        assert hinted == {"Ctrl+P", "Ctrl+Q"}, drawn_text

        for key in (
            "f1",
            "f6",
            "ctrl+shift+f",
            "alt+m",
            "ctrl+k",
            "ctrl+t",
            "alt+i",
            "alt+a",
            "alt+c",
        ):
            focused = harness.focused
            await pilot.press(key)
            await pilot.pause(0.2)
            assert harness.screen is switcher and harness.focused is focused, key

        await pilot.press("escape")
        await _until(pilot, lambda: harness.screen is console, "the dialog to close")
        await _until(
            pilot,
            lambda: "Alt+M switch model" in str(drawn.render()),
            "the Console's keys to come back",
        )


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
