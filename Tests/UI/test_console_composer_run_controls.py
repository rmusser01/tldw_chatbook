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
from textual.events import Paste
from textual.widgets import Button, OptionList, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_console_dictation import FakeDictationSession
from Tests.UI.test_console_native_chat_flow import (
    _select_llamacpp_console,
    _staged_image_attachment,
)
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


async def _settle(pilot, predicate, timeout: float = 5.0) -> None:
    """Poll ``predicate`` WITHOUT asserting: a wait that never decides.

    For layout settles that hold on pre-fix code as well, so the assertions
    that follow -- not the wait -- give the verdict.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        await pilot.pause(0.05)


async def _type(pilot, text: str) -> None:
    """Type ``text`` key by key through the real key routing."""
    for character in text:
        await pilot.press("space" if character == " " else character)


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


def _fake_dictation(monkeypatch) -> FakeDictationSession:
    """Swap the real microphone session for a recording fake."""
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    return fake


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
            # Neutral settle: Stop has been through a layout pass and the
            # row's laid-out width has caught up with its budget (whatever
            # that budget is) before geometry is read. Review: settling on
            # the row alone once read Stop as Region(0, 0, 0, 0) -- the row
            # already sat at its budget before `display: block` was laid
            # out. Both hold on pre-fix code too (Stop laid out at x=83..238,
            # off-screen), so this waits; the assertions below decide.
            await _settle(
                pilot,
                lambda: stop.region.width > 0
                and actions.region.width == actions.styles.width.value,
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
                # Redirect is paid for by the row, never by the draft floor
                # (`_redirect_fits`): where it shows, the draft keeps it.
                draft = composer.query_one("#console-command-visible-text", Static)
                assert draft.region.width >= ConsoleComposerBar.DRAFT_MIN_RENDER_WIDTH, (
                    f"Redirect squeezed the draft to {draft.region.width} at {size}"
                )
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
    fake = _fake_dictation(monkeypatch)
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

            # Pass-through spy on the controller call Redirect makes: proves
            # Redirect itself ran with the draft (review nit -- a toast match
            # also passed on the handler's refusal copy).
            redirects: list[str] = []
            original_redirect = controller.redirect_active_run

            def spy_redirect(text):
                redirects.append(text)
                return original_redirect(text)

            controller.redirect_active_run = spy_redirect

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
                lambda: redirects == ["correct course"],
                lambda: f"Enter on the focused Redirect did not run Redirect ({redirects})",
            )
            assert sends == [], "Enter on Redirect went down the send path"
            assert queued() == 0, "Enter on Redirect queued the draft instead"
            # Redirect submits the draft like Send does: the next keystroke
            # belongs to the draft again, not to a still-focused Redirect.
            await _wait_for(
                pilot,
                lambda: console.app.focused is composer,
                lambda: f"focus stayed on {console.app.focused!r} after Redirect",
            )
            await _type(pilot, " now")
            assert composer.draft_text().endswith(" now"), composer.draft_text()

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
        # Polled: the caret blinks, so one read can land on its off phase.
        await _wait_for(
            pilot,
            lambda: ConsoleComposerBar.CURSOR_GLYPH in _draft_render(composer),
            "the focused draft never painted its caret",
        )

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

        # ...on every composer button, "Composer ▾" included (the doc lists
        # it; review: its TASK-15704 capture exemption swallowed Space).
        composer.query_one("#console-composer-collapse", Button).focus()
        await pilot.pause()
        await pilot.press("space")
        await _wait_for(
            pilot, lambda: composer.collapsed, "Space on Composer ▾ did not collapse"
        )
        assert composer.draft_text() == "hello"
        gateway.release.set()


# ---------------------------------------------------------------------------
# Review blocker: pressing or clicking a composer button never strands focus
# on a button, where typing is (rightly) swallowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["click", "enter"])
@private_profile_test
async def test_after_send_the_next_prompt_is_typed_into_the_draft(
    how, request, monkeypatch
):
    """Send disables itself once it dispatches. Textual then handed focus to
    its neighbour, Dictate, so the next prompt was swallowed and its first
    Space (or its Enter) started dictation -- after a mouse click on Send as
    much as after Enter on a Tab-focused Send."""
    fake = _fake_dictation(monkeypatch)
    gateway, host = _held_run_host()
    gateway.release.set()  # runs finish at once; this test is about focus
    async with host.run_test(size=(120, 40)) as pilot:
        console, composer = await _mounted(host, pilot)
        store = console._ensure_console_chat_store()
        composer.focus()
        composer.load_draft("first prompt")
        await pilot.pause()
        send = composer.query_one("#console-send-message", Button)
        if how == "click":
            await pilot.click("#console-send-message")
        else:
            send.focus()
            await pilot.pause()
            await pilot.press("enter")
        await _wait_for(
            pilot, lambda: _user_turns(store) == 1, "no first turn", timeout=20
        )
        # The run has fully finished (a turn is recorded before its run
        # starts, so `run_active` alone can read False too early).
        await _wait_for(
            pilot,
            lambda: any(
                message.role is ConsoleMessageRole.ASSISTANT
                and "final-chunk" in message.content
                for message in store.messages_for_session(store.active_session_id)
            )
            and not composer.run_active,
            "the first run never finished",
            timeout=20,
        )
        await pilot.pause()

        await _type(pilot, "second prompt")
        assert composer.draft_text() == "second prompt", (
            f"the next prompt was swallowed (focus {console.app.focused!r})"
        )
        assert fake.start_calls == 0, "a keystroke of the next prompt dictated"
        # Enter sends it (the send path is entered -- this harness does not
        # complete a second turn even for two plain draft sends).
        sends = _spy_sends(console)
        await pilot.press("enter")
        await _wait_for(pilot, lambda: len(sends) == 1, "Enter did not send it")
        assert fake.start_calls == 0, "Enter on the next prompt started dictation"


@pytest.mark.asyncio
@private_profile_test
async def test_clicking_composer_buttons_keeps_typing_in_the_draft(
    request, monkeypatch
):
    """A mouse click presses a composer button without focusing it."""
    fake = _fake_dictation(monkeypatch)
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        composer.focus()
        composer.load_draft("hello")
        await pilot.pause()

        await pilot.click("#console-composer-menu")
        await _wait_for(
            pilot,
            lambda: isinstance(host.screen_stack[-1], ConsoleComposerMenuModal),
            "clicking Menu did not open the Composer actions menu",
        )
        await pilot.press("escape")
        await _wait_for(
            pilot, lambda: host.screen_stack[-1] is console, "menu did not close"
        )
        await _type(pilot, "ab")
        assert composer.draft_text() == "helloab", (
            f"typing after the menu was lost (focus {console.app.focused!r})"
        )

        await pilot.click("#console-dictation")
        await _wait_for(pilot, lambda: fake.start_calls == 1, "Dictate click ignored")
        assert console.app.focused is composer, console.app.focused
        await _type(pilot, "cd")
        assert composer.draft_text() == "helloabcd"
        gateway.release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_enter_on_clear_attachment_returns_focus_to_the_draft(request):
    """✕ hides itself once the attachment is gone; its focus must land in the
    draft, not on whichever sibling Textual picks (Send, then swallowed)."""
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.set_pending_attachment(session.id, _staged_image_attachment())
        console._sync_console_control_bar()
        clear = composer.query_one("#console-clear-attachment", Button)
        await _wait_for(pilot, lambda: clear.display, "✕ never appeared")
        clear.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for(
            pilot,
            lambda: store.pending_attachment(session.id) is None,
            "Enter on ✕ did not clear the attachment",
        )
        await _wait_for(
            pilot,
            lambda: console.app.focused is composer,
            lambda: f"focus landed on {console.app.focused!r} after ✕",
        )
        await _type(pilot, "ab")
        assert composer.draft_text() == "ab"
        gateway.release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_undo_chord_still_reaches_the_draft_from_a_focused_button(request):
    """Typing is swallowed on a focused button, but the draft-wide chords
    (undo/redo, paging) keep their route (TASK-33622.2's passthrough)."""
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        composer.focus()
        composer.load_draft("hello")
        await pilot.pause()
        await _type(pilot, " world")
        assert composer.draft_text() == "hello world"
        menu = composer.query_one("#console-composer-menu", Button)
        menu.focus()
        await pilot.pause()
        await pilot.press("ctrl+z")
        await _wait_for(
            pilot,
            lambda: composer.draft_text() != "hello world",
            "Ctrl+Z on a focused composer button never reached the draft",
        )
        assert "hello world".startswith(composer.draft_text())
        assert console.app.focused is menu
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
# PR #2934 review: the setup gate, Redirect's reservation, a paste's target
# ---------------------------------------------------------------------------


def _setup_blocked_palette_host():
    """A Console whose first-run setup card blocks it (empty OpenAI key)."""

    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    return _PaletteConsoleHarness(app)


@pytest.mark.asyncio
@private_profile_test
async def test_palette_composer_actions_are_inert_while_setup_blocks(
    request, monkeypatch
):
    """The setup card is embedded in the Console, not a pushed screen, so
    Ctrl+P still lists the composer entries under it. Like every other
    Console palette action, they must not open the menu or run an action
    over the card (Qodo, PR #2934)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    host = _setup_blocked_palette_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for(
            pilot,
            console._console_setup_modal_blocking,
            "the setup card never blocked the Console",
        )
        chosen: list[str] = []
        original_choice = console._handle_console_composer_menu_choice

        def recording_choice(action_id):
            chosen.append(action_id)
            return original_choice(action_id)

        console._handle_console_composer_menu_choice = recording_choice

        for query, expected in (
            ("open composer menu", "Open composer menu"),
            ("attach file", "Attach file"),
            ("impersonate", "Impersonate"),
        ):
            await _run_palette_command(host, pilot, query, expected)
            await pilot.pause(0.2)
            assert host.screen_stack[-1] is console, (
                f"palette {expected!r} opened "
                f"{type(host.screen_stack[-1]).__name__} over the setup card"
            )
            assert chosen == [], f"palette {expected!r} ran {chosen} under setup"


#: Parameter sentinel: one terminal column under the width at which the
#: expanded row first reserves Redirect's cells (`_redirect_fits`).
_UNDER_REDIRECT_THRESHOLD = "under-threshold"


def _redirect_threshold_columns(host, composer) -> int:
    """The narrowest terminal width whose expanded row reserves Redirect.

    Read from the laid-out expanded row (terminal width minus row width is
    the fixed chrome) and the same budget terms `_redirect_fits` sums.
    """

    chrome = host.size.width - composer._expanded_row_width()
    needed = (
        ConsoleComposerBar.LEFT_CLUSTER_WIDTH
        + composer._actions_row_width(redirect_budgeted=True)
        + ConsoleComposerBar.ADVISORY_MARGIN_ALLOWANCE
        + ConsoleComposerBar.DRAFT_MIN_RENDER_WIDTH
        + ConsoleComposerBar.SEND_REASON_MAX_WIDTH
    )
    return needed + chrome


def _record_laid_out_frames(console, sample) -> list:
    """Record ``sample()`` after every layout pass of ``console``.

    Wraps the screen's own ``_refresh_layout`` (pass-through), so each entry
    is the geometry of a frame the compositor actually laid out -- the first
    entry after an action is the first frame the user sees, before any
    ``call_after_refresh`` correction lands.
    """

    frames: list = []
    original = console._refresh_layout

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        frames.append(sample())
        return result

    console._refresh_layout = recording
    return frames


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("collapsed_width", "redirect_shown"),
    [
        (220, True),
        (120, False),
        # One column under Redirect's threshold (149 today): the collapsed
        # bar's own content width is two cells wider than the expanded row
        # (the collapsed presentation drops its padding), so measuring THAT
        # reserved Redirect here while collapsed and shifted Send 10 cells
        # after the first frame. Derived, so it tracks the budget constants.
        (_UNDER_REDIRECT_THRESHOLD, False),
    ],
)
@private_profile_test
async def test_redirect_reservation_survives_a_resize_while_collapsed(
    collapsed_width, redirect_shown, request
):
    """A resize while collapsed measured the HIDDEN expanded row (0 cells)
    and dropped Redirect's reservation. Expanding keeps the bar's one-row
    size, so no resize re-derived it: the row widened later -- at the first
    keystroke or run start -- and shifted Send and Dictate, the very row
    contract the reservation exists to keep (Qodo, PR #2934). The narrow
    case pins the other half: a reservation kept while collapsed must still
    be dropped where Redirect no longer fits.

    Both halves hold from the FIRST frame after expanding (PR #2934
    checkpoint review): keeping the stale 235-column reservation while
    collapsed and re-measuring after the expand painted one frame at 120
    columns with the draft at 23 cells (under its 32-cell floor) and Send at
    x=76, then shifted Send and Dictate 10 cells on the next frame."""
    gateway, host = _held_run_host()
    async with host.run_test(size=(235, 52)) as pilot:
        console, composer = await _mounted(host, pilot)
        send = composer.query_one("#console-send-message", Button)
        dictate = composer.query_one("#console-dictation", Button)
        redirect = composer.query_one("#console-redirect-generation", Button)
        draft = composer.query_one("#console-command-visible-text", Static)

        def geometry():
            return (send.region.x, dictate.region.x, draft.region.width)

        await _wait_for(
            pilot, lambda: composer._redirect_budgeted, "235 columns never reserved"
        )
        if collapsed_width == _UNDER_REDIRECT_THRESHOLD:
            collapsed_width = _redirect_threshold_columns(host, composer) - 1
        try:
            console._set_console_composer_collapsed(True)
            await _wait_for(pilot, lambda: composer.collapsed, "never collapsed")
            await pilot.pause(0.2)
            await pilot.resize_terminal(collapsed_width, 52)
            await pilot.pause(0.3)
            frames = _record_laid_out_frames(
                console, lambda: (send.region.width, geometry())
            )
            console._set_console_composer_collapsed(False)
            await _wait_for(
                pilot,
                lambda: not composer.collapsed and send.region.width > 0,
                "the composer never expanded",
            )
            await pilot.pause(0.3)
            expanded_positions = (send.region.x, dictate.region.x)
            settled = geometry()
            laid_out = [frame for width, frame in frames if width > 0]
            assert laid_out, "no frame laid the expanded row out"
            assert laid_out[0] == settled, (
                f"the first frame after expanding at {collapsed_width} columns "
                f"painted (Send x, Dictate x, draft width) {laid_out[0]}, then "
                f"settled at {settled}: Redirect's reservation was stale while "
                f"collapsed (frames {laid_out})"
            )
            assert laid_out[0][2] >= ConsoleComposerBar.DRAFT_MIN_RENDER_WIDTH, (
                f"the first frame squeezed the draft to {laid_out[0][2]} cells"
            )

            await _start_held_run(console, composer, pilot)
            if redirect_shown:
                await _wait_for(
                    pilot,
                    lambda: redirect.display,
                    f"Redirect never showed at {collapsed_width} columns",
                )
                await _settle(pilot, lambda: redirect.region.width > 0)
            else:
                assert not redirect.display, (
                    f"Redirect shown at {collapsed_width} columns"
                )
            assert (send.region.x, dictate.region.x) == expanded_positions, (
                "Send/Dictate shifted after expanding: Redirect's reservation "
                f"was stale after the collapsed resize ({expanded_positions} -> "
                f"{(send.region.x, dictate.region.x)})"
            )
        finally:
            gateway.release.set()
            await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_paste_on_a_focused_button_lands_in_a_focused_draft(request):
    """A paste while Menu held focus edited a draft that showed no caret
    (AC#3/#4's one-focus rule). A paste is a deliberate gesture aimed at the
    draft, not stray typing, so the draft takes focus and the paste lands
    where the caret shows it (Qodo, PR #2934)."""
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        composer.load_draft("hello")
        menu = composer.query_one("#console-composer-menu", Button)
        menu.focus()
        await pilot.pause()
        assert console.app.focused is menu

        menu.post_message(Paste(" world"))
        await _wait_for(
            pilot, lambda: composer.draft_text() != "hello", "the paste never landed"
        )
        assert composer.draft_text() == "hello world"
        assert console.app.focused is composer, (
            f"the paste edited the draft but focus stayed on {console.app.focused!r}"
        )
        await _wait_for(
            pilot,
            lambda: ConsoleComposerBar.CURSOR_GLYPH in _draft_render(composer),
            "the pasted-into draft never painted its caret",
        )
        gateway.release.set()


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
        queue_blocked_reason=(
            "Wait for this turn to be accepted before queueing a message."
        ),
        send_label=SEND_LABEL_PREPARING,
    )
    full = build_console_disabled_reason(
        action_id="send",
        has_draft=True,
        send_blocked=True,
        queue_blocked_reason="10/10 · Manage to make room",
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
    # Review: a real setup/attachment blocker outranks the queue state -- a
    # "Preparing..." label must not mask it behind queue copy.
    for label in (SEND_LABEL_PREPARING, SEND_LABEL_QUEUE_FULL):
        masked = build_console_disabled_reason(
            action_id="send",
            has_draft=True,
            send_blocked=True,
            setup_blocked_reason="Choose a model before sending.",
            queue_blocked_reason="Wait for this turn to be accepted.",
            send_label=label,
        )
        assert masked == "Send blocked — choose a model to continue", masked


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
