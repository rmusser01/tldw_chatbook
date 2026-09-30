"""TASK-33625.1 + TASK-33622.2: the composer's run controls and keyboard.

Two defects in the Console composer's action row, fixed together because
they share one cause (the screen's composer Enter capture):

* TASK-33625.1 -- while a run is active the row never painted Stop. The row
  is pinned to a fixed cell budget that TASK-28227 never widened when it
  added Redirect beside Stop, so Redirect clipped to "Redir" and Stop got
  zero cells at every terminal width. Enter on the (invisible) focused Stop
  went down the send path, and no key, palette entry or slash command could
  stop the viewed tab's run either.
* TASK-33622.2 -- `ChatScreen.on_key` treated every composer descendant as
  the draft, so Enter on a focused Menu/Dictate/Stop button sent the draft
  (a paid request) instead of pressing the button, and Space typed into a
  draft that did not have focus.

Every test here mounts the REAL ChatScreen (with the app stylesheet), holds a
real run open through the provider-gateway seam, and drives real key routing
through the Pilot. Mounted Console tests that build app config must select a
fresh profile first (`lessons-testing-evidence.md`: "Tests/UI
`RecoveryRequired` at setup is a profile-selection trip").
"""

from __future__ import annotations

import time

import pytest
from rich.cells import cell_len
from textual.app import App
from textual.errors import NoWidget
from textual.widgets import Button, OptionList, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_console_dictation import FakeDictationSession
from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_console_regenerate_feedback import GatedGateway
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Console_Modules import dictation as dictation_module
from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    ConsoleComposerMenuModal,
)

#: The terminal sizes the review verified live (G1-01): 80x24 is the smallest
#: supported, 235x52 a full-screen MacBook.
_REVIEW_SIZES = [(80, 24), (120, 40), (160, 45), (235, 52)]


class _PaletteConsoleHarness(ConsoleHarness):
    """The Console harness plus the Console palette provider (Ctrl+P)."""

    COMMANDS = App.COMMANDS | {ConsoleCommandProvider}


def _held_run_host(harness=ConsoleHarness):
    """Build a Console whose next send streams one chunk, then holds."""

    app = _build_test_app()
    attach_chachanotes_db(app)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    return gateway, harness(app)


async def _wait_for(pilot, predicate, what, timeout: float = 5.0) -> None:
    """Poll ``predicate``; ``what`` may be a callable for late diagnostics."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), what() if callable(what) else what


async def _mounted(host, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    _select_llamacpp_console(console)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    return console, composer


async def _start_held_run(console, composer, pilot):
    """Send one prompt and wait until its run is streaming and stoppable."""

    composer.load_draft("stream and hold")
    await pilot.pause()
    console.query_one("#console-send-message", Button).press()
    store = console._ensure_console_chat_store()
    await _wait_for(
        pilot,
        lambda: any(
            message.role is ConsoleMessageRole.ASSISTANT
            and message.content.startswith("first-chunk")
            for message in store.messages_for_session(store.active_session_id)
        ),
        "the held run never streamed its first chunk",
    )
    stop = composer.query_one("#console-stop-generation", Button)
    await _wait_for(pilot, lambda: stop.display, "Stop never displayed mid-run")
    await pilot.pause(0.1)
    return store


def _stopped_by_user(store) -> bool:
    return any(
        message.role is ConsoleMessageRole.SYSTEM
        and "stopped by user" in message.content.lower()
        for message in store.messages_for_session(store.active_session_id)
    )


def _user_turns(store) -> int:
    return sum(
        message.role is ConsoleMessageRole.USER
        for message in store.messages_for_session(store.active_session_id)
    )


def _spy_sends(console) -> list:
    """Record every entry into the visible send path (pass-through).

    `_send_console_message_from_visible_action` is where the `ui_action`
    console_send_stage is recorded; this wraps it without changing it, so a
    key that wrongly reaches the send path shows up here.
    """

    calls: list = []
    original = console._send_console_message_from_visible_action

    async def spy(**kwargs):
        calls.append(kwargs)
        return await original(**kwargs)

    console._send_console_message_from_visible_action = spy
    return calls


def _record_notices(console, host) -> list[str]:
    """Record every toast, whichever app object the handler notifies through."""

    notices: list[str] = []
    for owner in (console.app_instance, host):
        original = owner.notify

        def recording(message, *args, _original=original, **kwargs):
            notices.append(str(message))
            return _original(message, *args, **kwargs)

        owner.notify = recording
    return notices


def _painted_at(host, widget) -> bool:
    """True when the compositor hit-tests ``widget`` across its own cells."""

    region = widget.region
    if region.width <= 0 or region.height <= 0:
        return False
    for x in (region.x, region.right - 1):
        try:
            hit, _ = host.screen.get_widget_at(x, region.y)
        except NoWidget:
            return False
        if hit is not widget:
            return False
    return True


def _focus_style(button) -> tuple:
    return (button.styles.background, button.styles.color, button.styles.text_style)


def _draft_render(composer) -> str:
    return composer.query_one("#console-command-visible-text", Static).renderable.plain


# ---------------------------------------------------------------------------
# TASK-33625.1 AC#1/#5: Stop (and Redirect) are painted whole while running
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", _REVIEW_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@private_profile_test
async def test_running_stop_is_painted_whole_inside_the_action_row(size, request):
    gateway, host = _held_run_host()
    async with host.run_test(size=size) as pilot:
        console, composer = await _mounted(host, pilot)
        send = composer.query_one("#console-send-message", Button)
        dictate = composer.query_one("#console-dictation", Button)
        await pilot.pause(0.2)
        idle_positions = (send.region.x, dictate.region.x)
        try:
            await _start_held_run(console, composer, pilot)
            actions = composer.query_one("#console-composer-actions")
            stop = composer.query_one("#console-stop-generation", Button)
            screen_region = host.screen.region
            # Neutral settle: the row's laid-out width has caught up with its
            # budget (whatever that budget is) before geometry is read.
            await _wait_for(
                pilot,
                lambda: actions.region.width == actions.styles.width.value,
                "the action row never laid out at its budget",
            )

            assert stop.focusable, "a displayed Stop must be keyboard-reachable"
            assert stop.region.width >= cell_len("Stop") + 2, (
                f"Stop is not fully labelled at {size}: region {stop.region}"
            )
            assert screen_region.contains_region(stop.region), (
                f"Stop falls outside the {size} screen: {stop.region}"
            )
            assert actions.region.contains_region(stop.region), (
                f"Stop is clipped by the action row at {size}: "
                f"row {actions.region}, Stop {stop.region}"
            )
            assert _painted_at(host, stop), f"Stop is not painted at {size}"

            redirect = composer.query_one("#console-redirect-generation", Button)
            if redirect.display:
                assert redirect.region.width >= cell_len("Redirect") + 2, (
                    f"Redirect label clipped at {size}: {redirect.region}"
                )
                assert actions.region.contains_region(redirect.region), (
                    f"Redirect clipped by the action row at {size}"
                )
                assert _painted_at(host, redirect)
            else:
                assert redirect not in host.screen.focus_chain

            # Queue and Dictate keep their full cells too -- the budget was
            # derived from the controls actually shown, not squeezed -- and a
            # run starting never shifts them (TASK-1680's row contract).
            for control in (send, dictate):
                assert actions.region.contains_region(control.region), control.id
            assert (send.region.x, dictate.region.x) == idle_positions
        finally:
            gateway.release.set()
            await pilot.pause()


# ---------------------------------------------------------------------------
# TASK-33625.1 AC#2/#5: a Tab-focused Stop shows focus and its key stops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "size"), [("enter", (80, 24)), ("space", (160, 45))]
)
@private_profile_test
async def test_tab_reaches_a_visibly_focused_stop_and_its_key_stops_the_run(
    key, size, request
):
    gateway, host = _held_run_host()
    async with host.run_test(size=size) as pilot:
        console, composer = await _mounted(host, pilot)
        try:
            store = await _start_held_run(console, composer, pilot)
            stop = composer.query_one("#console-stop-generation", Button)
            unfocused = _focus_style(stop)
            user_turns = _user_turns(store)
            sends = _spy_sends(console)

            composer.focus()
            await pilot.pause()
            for _ in range(12):
                if console.app.focused is stop:
                    break
                await pilot.press("tab")
            assert console.app.focused is stop, (
                f"Tab never reached Stop (landed on {console.app.focused!r})"
            )
            await pilot.pause()
            assert _painted_at(host, stop), "the focused Stop is not on screen"
            assert _focus_style(stop) != unfocused, (
                "the focused Stop looks exactly like the unfocused one"
            )

            await pilot.press(key)
            await _wait_for(
                pilot,
                lambda: _stopped_by_user(store),
                f"{key} on the focused Stop did not stop the run",
            )
            assert sends == [], f"{key} on Stop went down the send path"
            assert _user_turns(store) == user_turns
            # The run ended and Stop hid: focus returns to the draft, where
            # the next prompt is typed -- not stranded on a hidden Stop or
            # handed to whatever widget followed it in the focus chain.
            await _wait_for(pilot, lambda: not stop.display, "Stop never hid")
            await _wait_for(
                pilot,
                lambda: console.app.focused is composer,
                lambda: f"focus landed on {console.app.focused!r} after Stop",
            )
        finally:
            gateway.release.set()
            await pilot.pause()


# ---------------------------------------------------------------------------
# TASK-33625.1 AC#3: a key (footer only while running), palette, and /stop
# ---------------------------------------------------------------------------


def _footer_keys(console) -> list[str]:
    registration = getattr(console, "_footer_shortcut_registration", None)
    return [key for key, _label in (registration[1] if registration else ())]


@pytest.mark.asyncio
@private_profile_test
async def test_stop_key_is_advertised_only_while_running_and_stops_the_run(request):
    gateway, host = _held_run_host()
    async with host.run_test(size=(120, 40)) as pilot:
        console, composer = await _mounted(host, pilot)
        try:
            assert "Ctrl+G" not in _footer_keys(console)
            store = await _start_held_run(console, composer, pilot)
            await _wait_for(
                pilot,
                lambda: "Ctrl+G" in _footer_keys(console),
                "the stop key is not advertised in the footer mid-run",
            )
            # The user is typing a follow-up in the focused draft; the stop
            # key must stop the run without touching that draft.
            composer.focus()
            composer.load_draft("follow-up in progress")
            await pilot.pause()
            await pilot.press("ctrl+g")
            await _wait_for(
                pilot, lambda: _stopped_by_user(store), "Ctrl+G did not stop the run"
            )
            assert composer.draft_text() == "follow-up in progress"
            await _wait_for(
                pilot,
                lambda: "Ctrl+G" not in _footer_keys(console),
                "the stop key is still advertised after the run ended",
            )
        finally:
            gateway.release.set()
            await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_slash_stop_stops_the_viewed_tabs_run_and_clears_itself(request):
    gateway, host = _held_run_host()
    async with host.run_test(size=(120, 40)) as pilot:
        console, composer = await _mounted(host, pilot)
        try:
            store = await _start_held_run(console, composer, pilot)
            user_turns = _user_turns(store)
            composer.focus()
            await pilot.pause()
            for character in "/stop":
                await pilot.press(character)
            await pilot.pause()
            # The slash popup is open on a non-empty prefix, so the first
            # Enter accepts the completion (TASK-24416) and the next runs it.
            for _ in range(2):
                if _stopped_by_user(store):
                    break
                await pilot.press("enter")
                await pilot.pause(0.2)
            await _wait_for(
                pilot, lambda: _stopped_by_user(store), "/stop did not stop the run"
            )
            assert _user_turns(store) == user_turns, "/stop was sent as a prompt"
            assert composer.draft_text() == ""
        finally:
            gateway.release.set()
            await pilot.pause()


async def _run_palette_command(host, pilot, query: str, expected: str) -> None:
    """Open the real Ctrl+P palette, type ``query``, run the top hit."""

    await pilot.press("ctrl+p")
    await pilot.pause()
    for character in query:
        await pilot.press(character)
    palette = host.screen_stack[-1]
    option_list = palette.query_one(OptionList)

    def _top_prompt() -> str:
        if option_list.option_count == 0:
            return ""
        return str(option_list.get_option_at_index(0).prompt)

    await _wait_for(
        pilot,
        lambda: expected in _top_prompt(),
        f"palette top hit for {query!r} is not {expected!r}",
    )
    await pilot.press("enter")
    for _ in range(4):
        await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_palette_stop_command_stops_the_viewed_tabs_run(request):
    gateway, host = _held_run_host(_PaletteConsoleHarness)
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        try:
            store = await _start_held_run(console, composer, pilot)
            await _run_palette_command(
                host, pilot, "stop this tab", "Stop this tab's run"
            )
            await _wait_for(
                pilot,
                lambda: _stopped_by_user(store),
                "the palette stop command did not stop the run",
            )
        finally:
            gateway.release.set()
            await pilot.pause()


# ---------------------------------------------------------------------------
# TASK-33625.1 AC#4: the reason strip matches the Queue state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_running_reason_copy_matches_the_queue_button(request):
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        try:
            await _start_held_run(console, composer, pilot)
            send = composer.query_one("#console-send-message", Button)
            reason = composer.query_one("#console-send-disabled-reason", Static)
            await _wait_for(
                pilot,
                lambda: send.label.plain == "Queue",
                "Send never became Queue for the accepted run",
            )
            await pilot.pause()
            assert reason.display
            text = reason.renderable.plain
            assert "Send" not in text, f"reason {text!r} sits beside a Queue button"
            assert "queue" in text.lower(), text
        finally:
            gateway.release.set()
            await pilot.pause()


# ---------------------------------------------------------------------------
# TASK-33622.2 AC#1/#2/#6: focused composer-bar buttons own Enter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_enter_on_each_idle_composer_button_runs_its_own_action(request):
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        store = console._ensure_console_chat_store()
        composer.load_draft("hello menu test")
        await pilot.pause()
        sends = _spy_sends(console)

        # Menu: Enter opens the Composer actions menu, focus on its first item.
        menu = composer.query_one("#console-composer-menu", Button)
        menu.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(
            pilot,
            lambda: isinstance(host.screen_stack[-1], ConsoleComposerMenuModal),
            "Enter on the focused Menu did not open the Composer actions menu",
        )
        modal = host.screen_stack[-1]
        await pilot.pause()
        first_item = modal.query(".console-composer-menu-item").first(Button)
        assert modal.focused is first_item
        assert sends == [], "Enter on Menu went down the send path"
        await pilot.press("escape")
        await _wait_for(
            pilot,
            lambda: host.screen_stack[-1] is console,
            "the composer menu did not close",
        )

        # Composer ▾ collapses; Expand ▴ restores.
        composer.query_one("#console-composer-collapse", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(pilot, lambda: composer.collapsed, "Composer ▾ ignored Enter")
        composer.query_one("#console-composer-expand", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(pilot, lambda: not composer.collapsed, "Expand ignored Enter")

        assert sends == []
        assert composer.draft_text() == "hello menu test"
        assert _user_turns(store) == 0

        # Send is the one button whose own action IS the send.
        send = composer.query_one("#console-send-message", Button)
        send.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(pilot, lambda: len(sends) == 1, "Enter on Send did not send")
        await _wait_for(pilot, lambda: _user_turns(store) == 1, "no user turn")
        gateway.release.set()
        await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_enter_on_focused_dictate_starts_dictation_without_sending(
    request, monkeypatch
):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        composer.load_draft("dictate beside this")
        await pilot.pause()
        sends = _spy_sends(console)
        dictate = composer.query_one("#console-dictation", Button)
        dictate.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(
            pilot,
            lambda: fake.start_calls == 1,
            "Enter on the focused Dictate never started dictation",
        )
        assert sends == [], "Enter on Dictate went down the send path"
        gateway.release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_enter_on_running_queue_and_redirect_run_their_own_actions(request):
    gateway, host = _held_run_host()
    async with host.run_test(size=(235, 52)) as pilot:
        console, composer = await _mounted(host, pilot)
        try:
            store = await _start_held_run(console, composer, pilot)
            controller = console._ensure_console_chat_controller()
            session_id = store.active_session_id

            def queued() -> int:
                return controller.prompt_queue_registry.snapshot(
                    session_id
                ).total_count

            notices = _record_notices(console, host)

            # Redirect: its own action (a redirect attempt), never a queue.
            composer.load_draft("correct course")
            await pilot.pause()
            redirect = composer.query_one("#console-redirect-generation", Button)
            assert redirect.display, "Redirect should fit at 235 columns"
            sends = _spy_sends(console)
            redirect.focus()
            await pilot.pause()
            await pilot.press("enter")
            await _wait_for(
                pilot,
                lambda: any("redirect" in notice.lower() for notice in notices),
                "Enter on the focused Redirect did not run Redirect",
            )
            assert sends == [], "Enter on Redirect went down the send path"
            assert queued() == 0, "Enter on Redirect queued the draft instead"

            # Queue: the Send button mid-run -- its own action queues.
            send = composer.query_one("#console-send-message", Button)
            await _wait_for(
                pilot,
                lambda: send.label.plain.startswith("Queue") and not send.disabled,
                lambda: (
                    f"Queue never became available (label {send.label.plain!r}, "
                    f"disabled {send.disabled}, draft {composer.draft_text()!r})"
                ),
            )
            send.focus()
            await pilot.pause()
            await pilot.press("enter")
            await _wait_for(pilot, lambda: queued() == 1, "Enter on Queue did not queue")
        finally:
            gateway.release.set()
            await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_enter_collapses_mid_run_and_the_collapsed_stop_takes_enter(request):
    """AC#1's "Expand/Stop where shown": Enter on Composer ▾ collapses the
    composer mid-run, and Enter on the collapsed strip's Stop stops the run."""
    gateway, host = _held_run_host()
    async with host.run_test(size=(120, 40)) as pilot:
        console, composer = await _mounted(host, pilot)
        try:
            store = await _start_held_run(console, composer, pilot)
            sends = _spy_sends(console)
            composer.query_one("#console-composer-collapse", Button).focus()
            await pilot.pause()
            await pilot.press("enter")
            await _wait_for(pilot, lambda: composer.collapsed, "Composer ▾ ignored Enter")
            collapsed_stop = composer.query_one(
                "#console-collapsed-stop-generation", Button
            )
            await _wait_for(
                pilot, lambda: collapsed_stop.display, "no collapsed Stop mid-run"
            )
            collapsed_stop.focus()
            await pilot.pause()
            await pilot.press("enter")
            await _wait_for(
                pilot,
                lambda: _stopped_by_user(store),
                "Enter on the collapsed Stop did not stop the run",
            )
            assert sends == []
        finally:
            gateway.release.set()
            await pilot.pause()


# ---------------------------------------------------------------------------
# TASK-33622.2 AC#3/#4: keys never edit an unfocused draft; one focus cue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_keys_on_a_focused_button_never_edit_the_draft_or_show_its_caret(
    request,
):
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        composer.load_draft("hello")
        composer.focus()
        await pilot.pause()
        assert ConsoleComposerBar.CURSOR_GLYPH in _draft_render(composer)

        menu = composer.query_one("#console-composer-menu", Button)
        menu.focus()
        await pilot.pause()
        assert ConsoleComposerBar.CURSOR_GLYPH not in _draft_render(composer), (
            "the draft still paints its caret while Menu holds focus"
        )

        for key in ("x", "backspace", "ctrl+w"):
            await pilot.press(key)
            await pilot.pause()
            assert composer.draft_text() == "hello", key
        assert console.app.focused is menu

        # Space activates the focused button (and still never types).
        await pilot.press("space")
        await _wait_for(
            pilot,
            lambda: isinstance(host.screen_stack[-1], ConsoleComposerMenuModal),
            "Space on the focused Menu did not open the menu",
        )
        assert composer.draft_text() == "hello"
        await pilot.press("escape")
        await pilot.pause()
        gateway.release.set()


# ---------------------------------------------------------------------------
# TASK-33622.2 AC#5: the composer menu and its menu-only actions in Ctrl+P
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_palette_opens_the_composer_menu_and_its_menu_only_actions(request):
    gateway, host = _held_run_host(_PaletteConsoleHarness)
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        composer.load_draft("a draft worth improving")
        await pilot.pause()

        notices = _record_notices(console, host)

        async def _back_to_console() -> None:
            while host.screen_stack[-1] is not console:
                await pilot.press("escape")
                await pilot.pause()

        await _run_palette_command(
            host, pilot, "open composer menu", "Open composer menu"
        )
        await _wait_for(
            pilot,
            lambda: isinstance(host.screen_stack[-1], ConsoleComposerMenuModal),
            "the palette did not open the Composer actions menu",
        )
        await _back_to_console()

        await _run_palette_command(host, pilot, "attach file", "Attach file")
        await _wait_for(
            pilot,
            lambda: type(host.screen_stack[-1]).__name__ == "EnhancedFileOpen",
            "the palette Attach file did not open the file picker",
        )
        await _back_to_console()

        await _run_palette_command(
            host, pilot, "improve current draft", "Improve current draft"
        )
        await _wait_for(
            pilot,
            lambda: type(host.screen_stack[-1]).__name__ == "ConsolePromptsModal",
            "the palette Improve current draft did not open the improve flow",
        )
        await _back_to_console()

        await _run_palette_command(host, pilot, "save as chatbook", "Save as Chatbook")
        await _wait_for(
            pilot,
            lambda: any("chatbook" in notice.lower() for notice in notices),
            "the palette Save as Chatbook said nothing",
        )

        await _run_palette_command(host, pilot, "impersonate", "Impersonate")
        await _wait_for(
            pilot,
            lambda: any("impersonate" in notice.lower() for notice in notices),
            "the palette Impersonate did not run",
        )
        gateway.release.set()
        await pilot.pause()


# ---------------------------------------------------------------------------
# Contract pins for the routes above (pure, no mounted app)
# ---------------------------------------------------------------------------


def test_disabled_reason_names_the_queue_state_not_send_or_setup():
    """TASK-33625.1 AC#4: mid-run copy follows the queue-state label."""
    from tldw_chatbook.Chat.console_display_state import (
        SEND_LABEL_PREPARING,
        SEND_LABEL_QUEUE,
        SEND_LABEL_QUEUE_FULL,
        build_console_disabled_reason,
    )

    preparing = build_console_disabled_reason(
        action_id="send",
        has_draft=True,
        send_blocked=True,
        setup_blocked_reason="Wait for this turn to be accepted before queueing a message.",
        send_label=SEND_LABEL_PREPARING,
    )
    full = build_console_disabled_reason(
        action_id="send",
        has_draft=True,
        send_blocked=True,
        setup_blocked_reason="10/10 · Manage to make room",
        send_label=SEND_LABEL_QUEUE_FULL,
    )
    empty_queue = build_console_disabled_reason(
        action_id="send",
        has_draft=False,
        send_blocked=False,
        send_label=SEND_LABEL_QUEUE,
    )
    for copy in (preparing, full, empty_queue):
        assert "Send" not in copy and "setup" not in copy, copy
        assert "queue" in copy.lower(), copy
    # Idle Send keeps its own copy.
    assert (
        build_console_disabled_reason(action_id="send", has_draft=False, send_blocked=False)
        == "Send disabled: type a message"
    )


def test_queue_state_labels_match_the_prompt_queue_presentation():
    """The reason builder keys off the presentation's exact label strings."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_display_state import (
        SEND_LABEL_PREPARING,
        SEND_LABEL_QUEUE,
        SEND_LABEL_QUEUE_FULL,
    )
    from tldw_chatbook.Chat.console_prompt_queue import (
        MAX_CONSOLE_QUEUE_ENTRIES,
        PromptQueueMode,
    )
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        derive_prompt_queue_presentation,
    )

    def label(*, accepted: bool, occupies: bool, count: int) -> str:
        snapshot = SimpleNamespace(
            total_count=count,
            mode=PromptQueueMode.DRAINING,
            pause_reason=None,
            entries=(),
            revision=1,
        )
        activity = SimpleNamespace(
            accepted_live_turn=accepted, occupies_slot=occupies
        )
        return derive_prompt_queue_presentation(snapshot, activity).send_label

    assert label(accepted=False, occupies=True, count=0) == SEND_LABEL_PREPARING
    assert label(accepted=True, occupies=True, count=0) == SEND_LABEL_QUEUE
    assert (
        label(accepted=True, occupies=True, count=MAX_CONSOLE_QUEUE_ENTRIES)
        == SEND_LABEL_QUEUE_FULL
    )


def test_stop_routes_are_registered_and_documented_in_f1():
    """TASK-33625.1 AC#3: key binding, /stop, palette entry, F1 vocabulary."""
    from tldw_chatbook.Chat.console_command_grammar import (
        KIND_COMMAND,
        STOP_COMMAND_HANDLER_ID,
        default_console_registry,
    )
    from tldw_chatbook.Chat.console_command_suggestions import _COMMAND_DESCRIPTIONS
    from tldw_chatbook.UI.Console_Modules.composer_run_controls import (
        STOP_RUN_KEY,
        STOP_RUN_KEY_LABEL,
    )
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_WORKBENCH_SHORTCUT_GROUPS,
        ChatScreen,
    )

    bindings = [b for b in ChatScreen.BINDINGS if b.key == STOP_RUN_KEY]
    assert [b.action for b in bindings] == ["stop_console_run"]
    assert callable(ChatScreen.action_stop_console_run)

    parse = default_console_registry().parse("/stop")
    assert parse.kind == KIND_COMMAND and parse.name == "stop"
    assert ChatScreen._CONSOLE_COMMAND_NAME_TO_HANDLER_ID["stop"] == (
        STOP_COMMAND_HANDLER_ID
    )
    assert "stop" in _COMMAND_DESCRIPTIONS

    help_keys = {
        key for _group, rows in CONSOLE_WORKBENCH_SHORTCUT_GROUPS for key, _ in rows
    }
    assert STOP_RUN_KEY_LABEL in help_keys


def test_palette_lists_stop_and_the_composer_menu_actions_class_safe():
    """TASK-33622.2 AC#5: labels build without a mounted screen (the class
    itself and inert stubs are both used as ``screen`` by other tests)."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    labels = {label for label, _, _ in ConsoleCommandProvider._commands(None, ChatScreen)}
    for expected in (
        "Console: Stop this tab's run",
        "Console: Redirect this tab's run",
        "Console: Open composer menu",
        "Console: Attach file…",
        "Console: Save as Chatbook",
        "Console: Impersonate",
        "Console: Improve current draft…",
    ):
        assert expected in labels, expected
