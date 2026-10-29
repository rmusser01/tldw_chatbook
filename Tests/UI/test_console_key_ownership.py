"""Console key ownership: bare letters, type-to-compose and Esc from the transcript.

Console UX review 2026-09-29, umbrella TASK-33622 ("keys reach the control
the user aimed at"), two P1 children fixed together because they share one
contract:

* TASK-33622.3 (G4-12) -- a printable key typed while a status-strip, rail or
  header control had focus was dropped, and the screen's bare ``y`` binding
  opened the full-screen Trace from any of them (the rest of the sentence then
  ran as Trace commands). Owner ruling 2026-10-27, recorded as an ADR-031
  refinement: single-letter screen actions fire only from the transcript (the
  announced list mode); elsewhere a printable key that no binding claims types
  into the composer.
* TASK-33622.4 (G4-13) -- ``ConsoleTranscript.on_key`` stopped every Esc, so the
  screen's ``escape -> focus_console_composer_home`` fallback never ran from a
  focused transcript, and prose typed there fired Edit/Copy/Regenerate.

Every case mounts the REAL ChatScreen (app stylesheet, real store, real key
routing through the Pilot). The file carries ``bootstrap_profile`` so it runs
under the plain ``pytest <file>`` command (`lessons-testing-evidence.md`:
"Tests/UI `RecoveryRequired` at setup is a profile-selection trip").
"""

from __future__ import annotations

import time

import pytest
from textual.app import App, ComposeResult
from textual.events import Key
from textual.screen import Screen
from textual.widgets import Button, Input, Static

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_conversation_actions import ConversationMenuTarget
from tldw_chatbook.Chat.console_workspace_actions import WorkspaceMenuTarget
from tldw_chatbook.UI.Console_Modules import composer_run_controls as run_controls
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_WORKBENCH_SHORTCUT_GROUPS,
    CONSOLE_WORKBENCH_SHORTCUTS,
    ChatScreen,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
    ConsoleConversationActionMenu,
)
from tldw_chatbook.Widgets.Console.console_message_more_menu import (
    message_more_menus_on_screen,
)
from tldw_chatbook.Widgets.Console.console_selection_menu import (
    selection_menus_on_screen,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscriptActionButton,
)
from tldw_chatbook.Widgets.Console.console_workspace_action_menu import (
    ConsoleWorkspaceActionMenu,
)

pytestmark = pytest.mark.bootstrap_profile

#: The cue the composer shows while focus is elsewhere (TASK-33622.3 AC#4).
_COMPOSER_CUE = "Esc to type"
#: The transcript's announcement while it holds focus (TASK-33622.4 AC#3).
_TRANSCRIPT_ANNOUNCEMENT = "Transcript keys · Esc to type"


def _host() -> ConsoleHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


async def _wait_for(pilot, predicate, what, timeout: float = 5.0) -> None:
    """Poll ``predicate``; ``what`` may be a callable for late diagnostics."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), what() if callable(what) else what


async def _type(pilot, text: str) -> None:
    """Type ``text`` key by key through the real key routing."""
    for character in text:
        await pilot.press("space" if character == " " else character)


async def _mounted(host, pilot, *, messages: bool = True):
    """Mount the Console, let its activation focus restore land, seed a chat.

    ``ChatScreen`` arms ``set_timer(0.2, _restore_console_workbench_focus)`` on
    mount; a case that moves focus before it runs can have the composer
    refocused under it (`lessons-testing-evidence.md`, TASK-33620.15.8). The
    wait is bounded on the composer taking focus, which is what the restore
    does when nothing owns focus.
    """
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    restore = console._restore_console_workbench_focus

    def restore_pending() -> bool:
        return any(
            timer._callback == restore
            and timer._task is not None
            and not timer._task.done()
            for timer in list(console._timers)
        )

    await _wait_for(pilot, lambda: not restore_pending(), "focus restore never ran")
    await pilot.pause()
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    if messages:
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.USER,
            content="first question",
        )
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="an answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for(
            pilot,
            lambda: len(transcript._visible_messages()) == 2,
            "the seeded messages never reached the transcript",
        )
    composer.focus()
    await _wait_for(pilot, lambda: composer.draft_has_focus, "composer never focused")
    return console, composer, transcript


def _spy_trace(console) -> list[str]:
    """Record every Trace launch (the ``y`` binding's only effect)."""
    calls: list[str] = []
    console._review_selection.open_trajectory_view = lambda: calls.append("trace")
    return calls


def _spy_transcript_actions(transcript) -> list[tuple]:
    """Record every transcript single-letter command that would run."""
    calls: list[tuple] = []
    transcript.action_invoke_selected_action = lambda action_id: calls.append(
        ("action", action_id)
    )
    transcript.action_open_review_notes = lambda: calls.append(("notes",))
    transcript.action_enter_text_selection = lambda: calls.append(("select-text",))
    return calls


async def _focus_transcript_with_f6(console, pilot, transcript) -> None:
    """Reach the transcript the way a keyboard user does: F6 between panes."""
    for _ in range(5):
        if console.focused is transcript:
            return
        await pilot.press("f6")
        await pilot.pause()
    assert console.focused is transcript, f"F6 never reached it: {console.focused!r}"


def _menus(console) -> list:
    """The in-screen popup menus still attached, removal not yet scheduled."""
    return [
        menu
        for menu in (
            *message_more_menus_on_screen(console),
            *selection_menus_on_screen(console),
        )
        if not getattr(menu, "_pruning", False)
    ]


def _focus_in(console, menu) -> bool:
    focused = console.focused
    return focused is not None and (focused is menu or menu in focused.ancestors)


async def _open_message_menu(console, pilot, transcript, kind: str):
    """Open one of the transcript's focus-taking menus from the keyboard.

    ``more``: F6, j, the selected message's More action, Enter.
    ``selection``: F6, j, ``s`` (keyboard text selection), ``l l``, Enter.

    Returns:
        The opened menu, once it holds focus.
    """
    await _focus_transcript_with_f6(console, pilot, transcript)
    await pilot.press("j")
    await _wait_for(
        pilot, lambda: transcript.selected_message_id is not None, "j selected nothing"
    )
    if kind == "more":
        selector = f"#console-message-action-more-{transcript.selected_message_id}"
        await _wait_for(
            pilot, lambda: bool(transcript.query(selector)), "no More action shown"
        )
        more = transcript.query_one(selector, Button)
        more.focus()
        await _wait_for(pilot, lambda: console.focused is more, "More never focused")
        await pilot.press("enter")
        menus = lambda: message_more_menus_on_screen(console)  # noqa: E731
    else:
        await pilot.press("s")
        await _wait_for(
            pilot,
            lambda: transcript._kb_selection_row is not None,
            "s did not arm keyboard text selection",
        )
        await pilot.press("l", "l", "enter")
        menus = lambda: selection_menus_on_screen(console)  # noqa: E731
    await _wait_for(
        pilot,
        lambda: bool(menus()) and _focus_in(console, menus()[0]),
        lambda: f"{kind} menu never took focus: {console.focused!r}",
    )
    return menus()[0]


def _draft_render(composer) -> str:
    return composer.query_one("#console-command-visible-text", Static).renderable.plain


def _footer_text(console) -> str:
    return console.query_one(AppFooterStatus).shortcut_text


# ---------------------------------------------------------------------------
# TASK-33622.4: Esc from the transcript returns to the composer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f6_j_esc_esc_returns_to_the_composer_and_prose_types_there():
    """AC#5 (and AC#1, AC#4): F6 to the transcript, j, Esc, Esc, 'look around'.

    The first Esc clears the selection, the second has nothing left to clear
    and must reach the screen's return-to-composer fallback. On the pre-fix
    code the transcript stopped every Esc: focus stayed on it, and 'look
    around' re-selected a message with 'k' and fired Regenerate with 'r'.
    """
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        trace = _spy_trace(console)
        actions = _spy_transcript_actions(transcript)
        stack = list(host.screen_stack)

        await _focus_transcript_with_f6(console, pilot, transcript)
        await pilot.press("j")
        await _wait_for(
            pilot,
            lambda: transcript.selected_message_id is not None,
            "j selected nothing",
        )

        await pilot.press("escape")
        await pilot.pause()
        assert transcript.selected_message_id is None
        assert console.focused is transcript, "the first Esc only clears"

        await pilot.press("escape")
        await pilot.pause()
        assert composer.draft_has_focus, (
            f"the second Esc left focus on {console.focused!r}"
        )

        await _type(pilot, "look around")
        assert composer.draft_text() == "look around"
        assert actions == []
        assert trace == []
        assert transcript.selected_message_id is None
        assert list(host.screen_stack) == stack


@pytest.mark.asyncio
async def test_esc_clears_text_selection_then_message_then_returns_to_composer():
    """AC#2: each layer is cleared, in order, before Esc returns to typing."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        await _focus_transcript_with_f6(console, pilot, transcript)
        await pilot.press("j")
        await _wait_for(
            pilot,
            lambda: transcript.selected_message_id is not None,
            "j selected nothing",
        )
        selected = transcript.selected_message_id
        await pilot.press("s")
        await _wait_for(
            pilot,
            lambda: transcript._kb_selection_row is not None,
            "s did not arm keyboard text selection",
        )

        await pilot.press("escape")
        await pilot.pause()
        assert transcript._kb_selection_row is None
        assert transcript.selected_message_id == selected
        assert console.focused is transcript

        await pilot.press("escape")
        await pilot.pause()
        assert transcript.selected_message_id is None
        assert console.focused is transcript

        await pilot.press("escape")
        await pilot.pause()
        assert composer.draft_has_focus, (
            f"Esc with nothing left to clear left focus on {console.focused!r}"
        )


@pytest.mark.asyncio
async def test_esc_on_a_focused_message_action_clears_then_returns_to_composer():
    """AC#2 from the action row: the first Esc clears the selection and keeps
    keyboard focus on the transcript, the next one returns to typing."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        await _focus_transcript_with_f6(console, pilot, transcript)
        await pilot.press("j")
        await _wait_for(
            pilot,
            lambda: bool(transcript.query(ConsoleTranscriptActionButton)),
            "the selected message never showed its action row",
        )
        action = next(
            button
            for button in transcript.query(ConsoleTranscriptActionButton)
            if not button.disabled
        )
        action.focus()
        await pilot.pause()
        assert console.focused is action

        await pilot.press("escape")
        await pilot.pause()
        assert transcript.selected_message_id is None
        assert console.focused is transcript, (
            f"Esc on an action left focus on {console.focused!r}"
        )

        await pilot.press("escape")
        await pilot.pause()
        assert composer.draft_has_focus


@pytest.mark.asyncio
async def test_a_focused_transcript_announces_its_keys_and_the_way_back():
    """AC#3: while the transcript has focus the footer leads with the
    announcement, and it goes away with the focus."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        assert _TRANSCRIPT_ANNOUNCEMENT not in _footer_text(console)

        await _focus_transcript_with_f6(console, pilot, transcript)
        await _wait_for(
            pilot,
            lambda: _footer_text(console).startswith(_TRANSCRIPT_ANNOUNCEMENT),
            lambda: f"footer while the transcript has focus: {_footer_text(console)!r}",
        )
        # The keys the announcement means: list navigation and Trace.
        assert "j/k select" in _footer_text(console)
        assert "Y trace" in _footer_text(console)
        # Enter shows the message actions here; it does not send.
        assert "Enter send" not in _footer_text(console)

        await pilot.press("escape")
        await _wait_for(pilot, lambda: composer.draft_has_focus, "Esc did not return")
        await _wait_for(
            pilot,
            lambda: _TRANSCRIPT_ANNOUNCEMENT not in _footer_text(console),
            lambda: f"footer after Esc: {_footer_text(console)!r}",
        )


# ---------------------------------------------------------------------------
# TASK-33622.3: typed text reaches the composer; bare 'y' only in the list mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_after_dismissing_help_lands_in_the_draft():
    """AC#1 / AC#6 (1): click header Help, Esc, type a sentence.

    Dismissing the dialog returns focus to the Help button (TASK-16211); on
    the pre-fix code every key typed there was dropped.
    """
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, _transcript = await _mounted(host, pilot)
        trace = _spy_trace(console)
        await pilot.click("#console-control-help")
        await _wait_for(
            pilot,
            lambda: host.screen_stack[-1] is not console,
            "Help did not open",
        )
        await pilot.press("escape")
        await _wait_for(
            pilot, lambda: host.screen_stack[-1] is console, "Help did not close"
        )
        await pilot.pause()

        await _type(pilot, "say hello world")
        assert composer.draft_text() == "say hello world"
        assert composer.draft_has_focus
        assert trace == []
        assert host.screen_stack[-1] is console


@pytest.mark.parametrize(
    "control",
    [
        "#console-status-text",  # the status strip (the Library chip's successor)
        "#console-control-help",  # a header action
        "#console-terminal-open",  # a left-rail button
        "alt+i",  # the Inspect rail's first control, reached by its key
    ],
)
@pytest.mark.asyncio
async def test_y_on_a_non_text_control_types_instead_of_opening_trace(control):
    """AC#2, AC#3 and AC#6 (2): 'y' on a status-strip, header or rail control
    moves focus to the composer and types there; it never opens Trace."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, _transcript = await _mounted(host, pilot)
        composer.load_draft("sa")
        await pilot.pause()
        trace = _spy_trace(console)
        if control.startswith("#"):
            target = console.query_one(control)
            target.focus()
            await _wait_for(
                pilot, lambda: console.focused is target, f"{control} never focused"
            )
        else:
            await pilot.press(control)
            await _wait_for(
                pilot,
                lambda: console._console_inspector_active(),
                "Alt+I did not focus the Inspect rail",
            )

        await pilot.press("y")
        await pilot.pause()
        assert trace == [], f"'y' on {control} opened Trace"
        assert composer.draft_has_focus, f"'y' left focus on {console.focused!r}"
        assert composer.draft_text() == "say"
        assert host.screen_stack[-1] is console


@pytest.mark.asyncio
async def test_y_from_the_transcript_still_opens_trace_and_f1_lists_it():
    """AC#3: Trace stays reachable by its documented key from the transcript,
    and F1 help lists that key."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, _composer, transcript = await _mounted(host, pilot)
        trace = _spy_trace(console)
        await _focus_transcript_with_f6(console, pilot, transcript)
        await pilot.press("y")
        await pilot.pause()
        assert trace == ["trace"]

    transcript_keys = dict(dict(CONSOLE_WORKBENCH_SHORTCUT_GROUPS)["Transcript"])
    assert "y" in transcript_keys, transcript_keys
    assert "trace" in transcript_keys["y"].lower()
    # The general footer no longer advertises a key that only the
    # transcript honours (ADR-031 rule 4).
    assert ("Y", "trace") not in CONSOLE_WORKBENCH_SHORTCUTS


@pytest.mark.asyncio
async def test_bound_keys_text_fields_and_space_keep_their_keys():
    """AC#2's 'with no binding': keys a focused control owns stay there."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, _transcript = await _mounted(host, pilot)
        composer.load_draft("keep")
        await pilot.pause()

        # Space belongs to a focused button (the ARIA activation key); it
        # never starts typing into the draft.
        help_button = console.query_one("#console-control-help", Button)
        help_button.focus()
        await _wait_for(pilot, lambda: console.focused is help_button, "no Help focus")
        await pilot.press("space")
        await pilot.pause()
        assert composer.draft_text() == "keep"
        assert not composer.draft_has_focus

        # A text field owns every printable key. One key is enough: it is the
        # only one type-to-compose could take (focus would have moved), and a
        # second key would race the rail tray's rebuild of its search field,
        # which can swallow it (TASK-33622.6, a separate defect).
        rail_filter = console.query_one("#console-workspace-conversation-search", Input)
        rail_filter.focus()
        await _wait_for(
            pilot, lambda: console.focused is rail_filter, "no filter focus"
        )
        await _type(pilot, "a")
        await pilot.pause()
        assert composer.draft_text() == "keep"
        assert not composer.draft_has_focus
        current = console.query_one("#console-workspace-conversation-search", Input)
        assert current.value == "a"


@pytest.mark.asyncio
async def test_the_composer_shows_a_cue_while_focus_is_elsewhere():
    """AC#4: an unfocused composer says how to get back to typing."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        assert _COMPOSER_CUE not in _draft_render(composer)

        await _focus_transcript_with_f6(console, pilot, transcript)
        await _wait_for(
            pilot,
            lambda: _COMPOSER_CUE in _draft_render(composer),
            lambda: f"unfocused empty composer shows {_draft_render(composer)!r}",
        )

        composer.load_draft("half a thought")
        await _wait_for(
            pilot,
            lambda: (
                _COMPOSER_CUE in _draft_render(composer)
                and "half a thought" in _draft_render(composer)
            ),
            lambda: (
                f"unfocused composer with a draft shows {_draft_render(composer)!r}"
            ),
        )

        composer.focus()
        await _wait_for(
            pilot,
            lambda: _COMPOSER_CUE not in _draft_render(composer),
            lambda: f"focused composer still shows {_draft_render(composer)!r}",
        )


class _RowList(Static, can_focus=True):
    """A focusable list control that owns one row key in its own BINDINGS."""

    BINDINGS = [("s", "star", "Star")]
    starred = 0

    def action_star(self) -> None:
        self.starred += 1


class _TypeToComposeScreen(Screen):
    """The attributes `type_to_compose` reads from the Console screen."""

    _resume_navigation_startup_in_progress = False
    _console_terminal_workspace = None
    _is_descendant_or_self = staticmethod(ChatScreen._is_descendant_or_self)

    def compose(self) -> ComposeResult:
        yield _RowList("rows", id="rows")
        yield Input(id="field")
        yield ConsoleComposerBar(id="console-native-composer")


class _TypeToComposeApp(App[None]):
    def on_mount(self) -> None:
        self.push_screen(_TypeToComposeScreen())


@pytest.mark.asyncio
async def test_type_to_compose_leaves_keys_a_binding_or_text_field_owns():
    """AC#2's 'with no binding': the policy itself, without the full Console.

    A key the focused control binds stays with it, a text field keeps every
    key, and only an unclaimed printable moves focus to the draft.
    """
    app = _TypeToComposeApp()
    async with app.run_test(size=(120, 20)) as pilot:
        screen = app.screen
        rows = screen.query_one("#rows", _RowList)
        field = screen.query_one("#field", Input)
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)

        rows.focus()
        await pilot.pause()
        run_controls.type_to_compose(screen, composer, Key("s", "s"))
        assert screen.focused is rows, "a bound row key left its control"
        run_controls.type_to_compose(screen, composer, Key("space", " "))
        assert screen.focused is rows, "Space left a focused control"
        run_controls.type_to_compose(screen, composer, Key("x", "x"))
        assert composer.draft_has_focus, "an unclaimed letter did not move"

        field.focus()
        await pilot.pause()
        run_controls.type_to_compose(screen, composer, Key("x", "x"))
        assert screen.focused is field, "a text field lost a typed key"


@pytest.mark.asyncio
async def test_type_to_compose_leaves_the_arrival_window_to_the_arrival_claim():
    """TASK-33620.15.17 compatibility: while the activation's arrival window
    is open (focus still where the activation found it), only
    `claim_arrival_key` decides where a typed key goes. It refuses when the
    restore would pick another pane or the setup card is about to block --
    including before the card's first sync has made the draft unfocusable --
    and type-to-compose must not take that key to the draft behind it."""
    app = _TypeToComposeApp()
    async with app.run_test(size=(120, 20)) as pilot:
        screen = app.screen
        rows = screen.query_one("#rows", _RowList)
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        rows.focus()
        await pilot.pause()

        screen._console_arrival_focus = rows
        run_controls.type_to_compose(screen, composer, Key("x", "x"))
        assert screen.focused is rows, "type-to-compose claimed an arrival key"

        # Once the window has closed, the same key composes.
        screen._console_arrival_focus = None
        run_controls.type_to_compose(screen, composer, Key("x", "x"))
        assert composer.draft_has_focus


# ---------------------------------------------------------------------------
# Popup menus: Esc closes them first; a typed letter folds them and types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["more", "selection"])
@pytest.mark.asyncio
async def test_esc_closes_an_open_message_menu_before_returning_to_the_composer(
    kind,
):
    """TASK-33622.4 AC#2: Esc folds an open message menu (a message's More
    menu, the text-selection menu) and clears the selection before it
    returns to the composer -- one layer per press."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        await _open_message_menu(console, pilot, transcript, kind)

        await pilot.press("escape")
        await _wait_for(
            pilot, lambda: not _menus(console), f"Esc left the {kind} menu open"
        )
        await pilot.pause()
        assert transcript.selected_message_id is not None, "Esc cleared two layers"
        assert not composer.draft_has_focus, "Esc skipped the selection"

        await pilot.press("escape")
        await _wait_for(
            pilot,
            lambda: transcript.selected_message_id is None,
            "the second Esc did not clear the selection",
        )
        await pilot.pause()
        assert console.focused is transcript

        await pilot.press("escape")
        await _wait_for(
            pilot,
            lambda: composer.draft_has_focus,
            lambda: f"the last Esc left focus on {console.focused!r}",
        )


@pytest.mark.parametrize("kind", ["more", "selection"])
@pytest.mark.asyncio
async def test_a_letter_typed_in_an_open_message_menu_folds_it_and_types(kind):
    """TASK-33622.3 review: a menu that takes focus has no use for a letter.
    The letter lands in the draft AND the menu folds -- it must not float on
    without focus, out of Esc's reach (Esc from the composer does not close
    these menus). Focus stays in the draft: the selection menu's unmount
    restore must not pull it back to the transcript."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        trace = _spy_trace(console)
        await _open_message_menu(console, pilot, transcript, kind)

        await pilot.press("x")
        await pilot.pause()
        assert composer.draft_has_focus, f"'x' left focus on {console.focused!r}"
        assert composer.draft_text() == "x"
        await _wait_for(
            pilot,
            lambda: not _menus(console),
            lambda: f"the {kind} menu stayed open: {_menus(console)!r}",
        )
        await pilot.pause()
        await pilot.pause()
        assert composer.draft_has_focus, (
            f"closing the {kind} menu moved focus to {console.focused!r}"
        )

        await _type(pilot, "yz")
        assert composer.draft_text() == "xyz"
        assert trace == []
        assert transcript.selection_manager.state.selection is None, (
            "the text-selection highlight outlived its menu"
        )


def _row_menu(kind: str):
    """A real row action menu: the Workspaces tree's or a conversation row's."""
    if kind == "workspace":
        return ConsoleWorkspaceActionMenu(
            target=WorkspaceMenuTarget(workspace_id="w1", name="One"),
            opener_id="rows",
            screen_x=0,
            screen_y=0,
        )
    return ConsoleConversationActionMenu(
        target=ConversationMenuTarget(conversation_id="c1", title="One"),
        opener_id="rows",
        screen_x=0,
        screen_y=0,
    )


class _RowMenuScreen(_TypeToComposeScreen):
    """The policy harness plus one real row action menu."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self._kind = kind

    def compose(self) -> ComposeResult:
        yield from super().compose()
        yield _row_menu(self._kind)


class _RowMenuApp(App[None]):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self._kind = kind

    def on_mount(self) -> None:
        self.push_screen(_RowMenuScreen(self._kind))


@pytest.mark.parametrize("kind", ["workspace", "conversation"])
@pytest.mark.asyncio
async def test_type_to_compose_folds_a_row_menu_without_handing_focus_back(kind):
    """TASK-33622.3 review: the conversation- and workspace-row menus fold the
    same way, and their dismissal does not hand focus back to the opener."""
    app = _RowMenuApp(kind)
    async with app.run_test(size=(120, 20)) as pilot:
        screen = app.screen
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        menu_type = (
            ConsoleWorkspaceActionMenu
            if kind == "workspace"
            else ConsoleConversationActionMenu
        )
        menu = screen.query_one(menu_type)
        await _wait_for(
            pilot, lambda: _focus_in(screen, menu), "the row menu never took focus"
        )

        run_controls.type_to_compose(screen, composer, Key("x", "x"))
        assert composer.draft_has_focus, "an unclaimed letter did not move"
        await _wait_for(
            pilot, lambda: not screen.query(menu_type), "the row menu stayed open"
        )
        await pilot.pause()
        assert composer.draft_has_focus, "the menu handed focus back to its opener"
