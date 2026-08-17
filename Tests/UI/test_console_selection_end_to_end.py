"""End-to-end console selection: quote routing into the composer (task F).

Closes the loop task E's menu opened: ``ConsoleSelectionQuoteRequested``
bubbles from the transcript to ``ChatScreen``, which inserts the selection
as a block quote into the native composer at the caret. Also covers the
screen-level click-outside dismissal of a mounted selection menu (a click
on any non-transcript widget, e.g. the composer, folds the menu).

Phase 2 (task 5): ``ConsoleSideChatRequested`` bubbles from the transcript
to ``ChatScreen``, which resolves the configured side-chat model + prompt
template, builds the ephemeral side-chat service over the (fake) provider
gateway, and pushes ``ConsoleSideChatModal`` exactly once -- More Details
with the rendered template auto-sent, Ask in Side Chat freeform.

Phase 3 (task 3): the selection menu offers ``Request changes | LGTM |
Comment`` only when the selection sits in agent output (ASSISTANT- or
TOOL-role rows, or diff rows -- product decision 2026-08-16: the agent's
own prose replies are review targets; USER rows never are), run-gated
through the owning screen's run-status seam, and each action makes the
transcript post ``ConsoleSelectionFeedbackRequested`` with the capped
quote before clearing the selection UI.

Phase 3 (task 5): the ChatScreen consumes that request -- comment modal,
then the structured message (action header + ``> ``-quoted selection +
optional comment) routed as the next user message through the prompt
queue. The queue seam queues behind an active run and sends immediately
otherwise; the composer draft is never touched, and a modal cancel
(Escape/Cancel/backdrop) abandons the whole feedback.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Input

from Tests.UI.test_console_left_rail import make_console_pilot
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import (
    FEEDBACK_ACTIVE_RUN_STATUSES,
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_feedback_comment_modal import (
    ConsoleFeedbackCommentModal,
)
from tldw_chatbook.Widgets.Console.console_selection import TextSelection
from tldw_chatbook.Widgets.Console.console_selection_menu import (
    ConsoleSelectionFeedbackRequested,
    ConsoleSelectionMenu,
    ConsoleSelectionQuoteRequested,
    ConsoleSideChatRequested,
)
from tldw_chatbook.Widgets.Console.console_side_chat_modal import ConsoleSideChatModal
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
    _SELECTION_FEEDBACK_ACTIVE_RUN_STATUSES,
)


class _ComposerApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleComposerBar(id="console-native-composer")


@pytest.mark.asyncio
async def test_insert_quote_prepends_quote_markers():
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_quote("line one\nline two")
        assert "> line one\n> line two" in composer.draft_text()


@pytest.mark.asyncio
async def test_insert_quote_blank_lines_get_bare_marker():
    """An empty selection line quotes as a bare ``>`` (a real block quote)."""
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_quote("first\n\nlast")
        assert composer.draft_text().endswith("> first\n>\n> last")


@pytest.mark.asyncio
async def test_insert_quote_empty_selection_is_noop():
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_text("existing")
        composer.insert_quote("")
        composer.insert_quote("\n")
        assert composer.draft_text() == "existing"


@pytest.mark.asyncio
async def test_insert_quote_lands_at_caret_not_end():
    """The quote splices at the caret, wherever it sits in the draft."""
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_text("hello world")
        for _ in range(6):
            composer.move_cursor_left()  # caret between "hello" and " world"
        composer.insert_quote("X")
        assert "hello> X world" in composer.draft_text()


@pytest.mark.asyncio
async def test_insert_quote_unfocused_composer_lands_at_end():
    """An unfocused composer still inserts: the caret is not focus-bound.

    Spec fallback: the caret always exists in the segment model and sits at
    the end of a freshly initialised (never focused) draft.
    """
    app = _ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_quote("tail insert")
        assert composer.draft_text().endswith("> tail insert")


@pytest.mark.asyncio
async def test_screen_routes_quote_request_into_composer():
    """ChatScreen consumes ConsoleSelectionQuoteRequested into the draft."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        screen.post_message(ConsoleSelectionQuoteRequested(quote="hello world"))
        await pilot.pause()
        assert "> hello world" in composer.draft_text()


@pytest.mark.asyncio
async def test_empty_quote_request_notifies_nothing():
    """A cleared row range quotes nothing -- and toasts nothing (final review).

    If the row range was cleared while the menu was open (streaming
    replace, reconciliation), Add to chat posts an empty quote:
    ``insert_quote`` already no-ops, and the screen must not claim
    "Added selection to composer" for an insert that never happened.
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        notifications: list[str] = []
        pilot.app.notify = lambda message, **kwargs: notifications.append(str(message))
        draft_before = composer.draft_text()

        screen.post_message(ConsoleSelectionQuoteRequested(quote=""))
        await pilot.pause()

        assert composer.draft_text() == draft_before
        assert notifications == []

        # Control: a real selection still inserts and still notifies.
        screen.post_message(ConsoleSelectionQuoteRequested(quote="real text"))
        await pilot.pause()
        assert "> real text" in composer.draft_text()
        assert notifications == ["Added selection to composer"]


@pytest.mark.asyncio
async def test_click_outside_transcript_dismisses_selection_menu():
    """A click on a non-transcript widget (the composer) folds the menu."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        transcript = screen.query_one("#console-native-transcript", ConsoleTranscript)
        # Mount the menu exactly as a real selection release would (onto the
        # transcript); only the click-outside seam is under test here.
        await transcript.mount(ConsoleSelectionMenu(screen_x=2, screen_y=2))
        await pilot.pause()
        assert screen.query_one(ConsoleSelectionMenu)

        await pilot.click("#console-native-composer")
        await pilot.pause()
        assert not screen.query(ConsoleSelectionMenu)


# ---------------------------------------------------------------------------
# Side chat (phase 2, task 5)
# ---------------------------------------------------------------------------


@dataclass
class _FakeResolution:
    provider: str = "llama_cpp"
    model: str = "local-model"
    ready: bool = True
    visible_copy: str = ""


class _FakeSideChatGateway:
    """Offline provider gateway: records sends, streams one short reply."""

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []
        self.stream_calls = 0

    async def resolve_for_send(self, selection):
        del selection
        return _FakeResolution()

    async def stream_chat(self, resolution, messages):
        self.stream_calls += 1
        self.messages.append(messages)
        del resolution
        yield "ok"


@asynccontextmanager
async def _side_chat_console_pilot(gateway: _FakeSideChatGateway):
    """Real ChatScreen console with the provider gateway injected offline."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        yield pilot, console


def _capture_side_chat_pushes(app) -> list[ConsoleSideChatModal]:
    """Wrap ``app.push_screen`` so each pushed screen lands in a list."""
    pushed: list[object] = []
    original_push = app.push_screen

    def capturing_push(screen_arg, *args, **kwargs):
        pushed.append(screen_arg)
        return original_push(screen_arg, *args, **kwargs)

    app.push_screen = capturing_push  # type: ignore[method-assign]
    return pushed


def _patch_side_chat_config(monkeypatch, *, model: str, template: str) -> None:
    """Pin the two side-chat config reads the screen handler makes."""
    real_get = chat_screen_module.get_cli_setting

    def pinned_get(section, *args, **kwargs):
        if section == "console" and args:
            if args[0] == "sidechat_model":
                return model
            if args[0] == "sidechat_prompt_template":
                return template
        return real_get(section, *args, **kwargs)

    monkeypatch.setattr(chat_screen_module, "get_cli_setting", pinned_get)


async def _await_condition(condition, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail("condition was not met in time")
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_screen_pushes_one_side_chat_modal_with_rendered_template(
    monkeypatch,
):
    """More Details: one modal, auto-sending the rendered prompt template."""
    gateway = _FakeSideChatGateway()
    _patch_side_chat_config(
        monkeypatch,
        model="zai/glm-test",
        template="Explain {selection} in depth",
    )
    async with _side_chat_console_pilot(gateway) as (pilot, console):
        pushed = _capture_side_chat_pushes(pilot.app)
        console.post_message(
            ConsoleSideChatRequested(quote="hello world", mode="more-details")
        )
        await pilot.pause()

        modals = [item for item in pushed if isinstance(item, ConsoleSideChatModal)]
        assert len(pushed) == 1
        assert len(modals) == 1
        modal = modals[0]
        assert modal._auto_send_prompt == "Explain hello world in depth"
        assert modal._quote == "hello world"
        assert modal._sidechat_model == "zai/glm-test"
        assert modal._service.gateway is gateway
        # Session fallback: derived from the ready llama.cpp session config.
        assert modal._provider_selection.provider == "llama_cpp"

        # More Details auto-sends on mount (offline fake gateway).
        await _await_condition(lambda: gateway.stream_calls == 1)
        user_content = gateway.messages[0][1]["content"]
        assert "Explain hello world in depth" in user_content
        assert "hello world" in user_content

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_screen_pushes_ask_mode_modal_without_auto_prompt(monkeypatch):
    """Ask in Side Chat: one modal, no auto-send, freeform prompt visible."""
    gateway = _FakeSideChatGateway()
    _patch_side_chat_config(
        monkeypatch,
        model="",
        template="Give me more details about: {selection}",
    )
    async with _side_chat_console_pilot(gateway) as (pilot, console):
        pushed = _capture_side_chat_pushes(pilot.app)
        console.post_message(ConsoleSideChatRequested(quote="a quote", mode="ask"))
        await pilot.pause()
        await pilot.pause()

        modals = [item for item in pushed if isinstance(item, ConsoleSideChatModal)]
        assert len(pushed) == 1
        assert len(modals) == 1
        modal = modals[0]
        assert modal._auto_send_prompt is None
        assert modal._quote == "a quote"
        assert modal._sidechat_model == ""
        assert gateway.stream_calls == 0  # nothing sent until the user asks

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_empty_side_chat_quote_pushes_no_modal(monkeypatch):
    """A cleared row range opens nothing -- no modal, no send (T5 review).

    Same blank-selection window as the add-to-chat guard: if the row
    range was cleared while the menu was open (streaming replace,
    reconciliation), More Details posts an empty quote, and the screen
    must not push a modal (let alone auto-send a contentless prompt).
    Mirrors ``test_empty_quote_request_notifies_nothing``.
    """
    gateway = _FakeSideChatGateway()
    _patch_side_chat_config(
        monkeypatch,
        model="",
        template="Give me more details about: {selection}",
    )
    async with _side_chat_console_pilot(gateway) as (pilot, console):
        pushed = _capture_side_chat_pushes(pilot.app)
        console.post_message(
            ConsoleSideChatRequested(quote="   \n  ", mode="more-details")
        )
        await pilot.pause()
        await pilot.pause()

        assert pushed == []  # whitespace-only quote: no modal at all
        assert gateway.stream_calls == 0

        # Control: a real quote still pushes exactly one modal.
        console.post_message(
            ConsoleSideChatRequested(quote="real text", mode="more-details")
        )
        await pilot.pause()
        modals = [item for item in pushed if isinstance(item, ConsoleSideChatModal)]
        assert len(pushed) == 1
        assert len(modals) == 1

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_drag_more_details_opens_side_chat_modal_end_to_end():
    """Drag → menu → More Details: the modal mounts showing the selection."""
    gateway = _FakeSideChatGateway()
    async with _side_chat_console_pilot(gateway) as (pilot, console):
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER,
                    content="hello selectable world",
                    id="m1",
                )
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()
        row = console.query_one("#console-message-m1", ConsoleTranscriptMessage)
        region = transcript.region

        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(
                selection=TextSelection(row.id, 0, 5),
                screen_x=region.x + 4,
                screen_y=region.y + 2,
            )
        )
        await pilot.pause()
        assert console.query_one(ConsoleSelectionMenu)  # menu mounted at release

        await pilot.click("#console-selection-more-details")
        await pilot.pause()
        await pilot.pause()

        modal = pilot.app.screen
        assert isinstance(modal, ConsoleSideChatModal)
        assert modal._quote == "hello"
        assert modal._auto_send_prompt is not None
        assert "hello" in modal._auto_send_prompt
        assert not console.query(ConsoleSelectionMenu)  # menu cleaned up
        await _await_condition(lambda: gateway.stream_calls == 1)

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_drag_ask_mode_opens_side_chat_modal_end_to_end():
    """Drag → menu → Ask in Side Chat: modal mounts waiting for a question."""
    gateway = _FakeSideChatGateway()
    async with _side_chat_console_pilot(gateway) as (pilot, console):
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER,
                    content="hello selectable world",
                    id="m1",
                )
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()
        row = console.query_one("#console-message-m1", ConsoleTranscriptMessage)
        region = transcript.region

        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(
                selection=TextSelection(row.id, 0, 5),
                screen_x=region.x + 4,
                screen_y=region.y + 2,
            )
        )
        await pilot.pause()

        await pilot.click("#console-selection-ask-side-chat")
        await pilot.pause()
        await pilot.pause()

        modal = pilot.app.screen
        assert isinstance(modal, ConsoleSideChatModal)
        assert modal._quote == "hello"
        assert modal._auto_send_prompt is None
        assert gateway.stream_calls == 0

        await pilot.press("escape")
        await pilot.pause()


# ---------------------------------------------------------------------------
# Selection feedback actions (phase 3, task 3)
# ---------------------------------------------------------------------------


class _StubRunStatusScreen(Screen):
    """Harness screen exposing the ChatScreen run-status seam verbatim."""

    def __init__(self, run_status: str) -> None:
        super().__init__()
        self._run_status = run_status

    def _current_console_run_status_value(self) -> str:
        return self._run_status


class _FeedbackTranscriptApp(App[None]):
    """Drag -> menu harness with app-level capture of feedback requests.

    The default screen is a plain ``Screen`` (no
    ``_current_console_run_status_value`` attribute: run gating closed,
    exactly like any non-ChatScreen host) unless ``run_status`` stubs the
    seam -- the transcript must derive the menu's ``run_active`` kwarg
    through ``getattr`` rather than assume a ChatScreen.
    """

    def __init__(self, *, role: ConsoleMessageRole, run_status: str | None = None):
        super().__init__()
        self._role = role
        self._run_status = run_status
        self.feedback_events: list[ConsoleSelectionFeedbackRequested] = []

    def get_default_screen(self) -> Screen:
        if self._run_status is None:
            return Screen()
        return _StubRunStatusScreen(self._run_status)

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")

    # Module-level Message classes carry no widget namespace, so the
    # auto-generated handler is ``on_console_selection_feedback_requested``
    # (matching how ``on_console_side_chat_requested`` works for phase 2).
    def on_console_selection_feedback_requested(
        self, event: ConsoleSelectionFeedbackRequested
    ) -> None:
        self.feedback_events.append(event)


async def _drag_select_first_row(
    pilot, app: _FeedbackTranscriptApp, *, start: int = 0, end: int = 4
) -> ConsoleTranscriptMessage:
    """Finish a drag over the first row body and post the selection event.

    Mirrors the real release path: the manager finishes with a non-empty
    selection, the transcript posts ``TranscriptTextSelected``, and the
    selection menu mounts on the screen at the release cell.
    """
    transcript = app.query_one(ConsoleTranscript)
    transcript.set_messages(
        [
            ConsoleChatMessage(
                role=app._role, content="tool ran and wrote things", id="m1"
            )
        ]
    )
    await transcript.refresh_messages()
    await pilot.pause()
    row = app.query_one("#console-message-m1", ConsoleTranscriptMessage)
    region = transcript.region

    transcript.selection_manager.begin_drag(row.id, start)
    transcript.selection_manager.extend_drag(row.id, end)
    row.set_selection_range(start, end)
    transcript.selection_manager.finish_drag()
    transcript.post_message(
        ConsoleTranscript.TranscriptTextSelected(
            selection=TextSelection(row.id, start, end),
            screen_x=region.x + 4,
            screen_y=region.y + 2,
        )
    )
    await pilot.pause()
    return row


@pytest.mark.asyncio
async def test_drag_on_tool_role_row_shows_feedback_entries():
    """A selection in agent output (TOOL-role plain row) mounts the menu
    with the three feedback entries."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.TOOL, run_status="streaming")
    async with app.run_test(size=(80, 40)) as pilot:
        await _drag_select_first_row(pilot, app)
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.query_one("#console-selection-request-changes", Button)
        assert menu.query_one("#console-selection-lgm", Button)
        assert menu.query_one("#console-selection-comment", Button)


@pytest.mark.asyncio
async def test_drag_on_user_row_hides_feedback_entries():
    """A selection over the user's own message offers no feedback actions."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.USER, run_status="streaming")
    async with app.run_test(size=(80, 40)) as pilot:
        await _drag_select_first_row(pilot, app)
        menu = app.query_one(ConsoleSelectionMenu)
        assert not menu.query("#console-selection-request-changes")
        assert not menu.query("#console-selection-lgm")
        assert not menu.query("#console-selection-comment")
        assert not menu.query("#console-selection-feedback-hint")


async def _drag_select_first_markdown_row(
    pilot, app: _FeedbackTranscriptApp, *, start: int = 0, end: int = 4
) -> ConsoleMarkdownMessage:
    """Finish a drag over the first markdown row body and post the selection
    event.

    ASSISTANT-role messages render through ``ConsoleMarkdownMessage`` (the
    default ``assistant_markdown`` toggle in this config-less harness), so
    this mirrors ``_drag_select_first_row`` over the markdown row's
    character-granular source domain.
    """
    transcript = app.query_one(ConsoleTranscript)
    transcript.set_messages(
        [
            ConsoleChatMessage(
                role=app._role, content="answer prose worth reviewing", id="m1"
            )
        ]
    )
    await transcript.refresh_messages()
    await pilot.pause()
    row = app.query_one("#console-message-m1", ConsoleMarkdownMessage)
    region = transcript.region

    transcript.selection_manager.begin_drag(row.id, start)
    transcript.selection_manager.extend_drag(row.id, end)
    row.set_selection_range(start, end)
    transcript.selection_manager.finish_drag()
    transcript.post_message(
        ConsoleTranscript.TranscriptTextSelected(
            selection=TextSelection(row.id, start, end),
            screen_x=region.x + 4,
            screen_y=region.y + 2,
        )
    )
    await pilot.pause()
    return row


@pytest.mark.asyncio
async def test_drag_on_assistant_markdown_row_shows_feedback_entries():
    """Product decision 2026-08-16: assistant PROSE is agent output too -- a
    selection over the agent's own markdown reply mounts the menu with the
    three feedback entries, armed under an active run."""
    app = _FeedbackTranscriptApp(
        role=ConsoleMessageRole.ASSISTANT, run_status="streaming"
    )
    async with app.run_test(size=(80, 40)) as pilot:
        await _drag_select_first_markdown_row(pilot, app)
        menu = app.query_one(ConsoleSelectionMenu)
        assert not menu.query_one("#console-selection-request-changes", Button).disabled
        assert not menu.query_one("#console-selection-lgm", Button).disabled
        assert not menu.query_one("#console-selection-comment", Button).disabled
        assert not menu.query("#console-selection-feedback-hint")


@pytest.mark.asyncio
async def test_drag_on_assistant_markdown_row_run_gating_without_status_seam():
    """Assistant-prose feedback follows the same run gating as tool output:
    with no run-status seam on the owning screen, Request changes and LGTM
    render disabled (with the hint), Comment stays enabled."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.ASSISTANT)
    async with app.run_test(size=(80, 40)) as pilot:
        await _drag_select_first_markdown_row(pilot, app)
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.query_one("#console-selection-request-changes", Button).disabled
        assert menu.query_one("#console-selection-lgm", Button).disabled
        assert not menu.query_one("#console-selection-comment", Button).disabled
        assert menu.query_one("#console-selection-feedback-hint")


@pytest.mark.asyncio
async def test_run_gating_when_screen_lacks_status_seam():
    """No run-status attribute on the owning screen: Request changes and
    LGTM render disabled (with the hint), Comment stays enabled."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.TOOL)
    async with app.run_test(size=(80, 40)) as pilot:
        await _drag_select_first_row(pilot, app)
        menu = app.query_one(ConsoleSelectionMenu)
        request = menu.query_one("#console-selection-request-changes", Button)
        lgm = menu.query_one("#console-selection-lgm", Button)
        comment = menu.query_one("#console-selection-comment", Button)
        assert request.disabled
        assert lgm.disabled
        assert not comment.disabled
        assert menu.query_one("#console-selection-feedback-hint")


@pytest.mark.asyncio
async def test_run_gating_with_idle_status():
    """An idle run status gates the same as a missing seam."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.TOOL, run_status="idle")
    async with app.run_test(size=(80, 40)) as pilot:
        await _drag_select_first_row(pilot, app)
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.query_one("#console-selection-request-changes", Button).disabled
        assert menu.query_one("#console-selection-lgm", Button).disabled
        assert not menu.query_one("#console-selection-comment", Button).disabled
        assert menu.query_one("#console-selection-feedback-hint")


@pytest.mark.asyncio
async def test_active_run_status_enables_feedback_entries():
    """A streaming run unlocks Request changes and LGTM (no hint line)."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.TOOL, run_status="streaming")
    async with app.run_test(size=(80, 40)) as pilot:
        await _drag_select_first_row(pilot, app)
        menu = app.query_one(ConsoleSelectionMenu)
        assert not menu.query_one("#console-selection-request-changes", Button).disabled
        assert not menu.query_one("#console-selection-lgm", Button).disabled
        assert not menu.query_one("#console-selection-comment", Button).disabled
        assert not menu.query("#console-selection-feedback-hint")


def test_feedback_active_run_statuses_have_single_source_of_truth():
    """The transcript's string set and the screen's tuple both derive from
    the canonical ``FEEDBACK_ACTIVE_RUN_STATUSES`` next to ``ConsoleRunStatus``.

    Pins the derivation (final-review finding): a status added to one site
    without the canonical set must fail here, not silently drift.
    """
    assert set(FEEDBACK_ACTIVE_RUN_STATUSES) == {
        ConsoleRunStatus.VALIDATING,
        ConsoleRunStatus.STREAMING,
        ConsoleRunStatus.CHECKING_CITATIONS,
        ConsoleRunStatus.RETRYING,
    }
    assert _SELECTION_FEEDBACK_ACTIVE_RUN_STATUSES == {
        status.value for status in FEEDBACK_ACTIVE_RUN_STATUSES
    }
    assert set(chat_screen_module.CONSOLE_ACTIVE_RUN_STATUSES) == set(
        FEEDBACK_ACTIVE_RUN_STATUSES
    )


@pytest.mark.parametrize(
    ("run_status", "expect_armed"),
    [
        *(  # the four canonical active statuses arm the feedback actions
            (status.value, True) for status in sorted(FEEDBACK_ACTIVE_RUN_STATUSES)
        ),
        *(  # every idle/terminal status keeps them gated
            (status.value, False)
            for status in (
                ConsoleRunStatus.COMPLETED,
                ConsoleRunStatus.BLOCKED,
                ConsoleRunStatus.STOPPED,
                ConsoleRunStatus.FAILED,
                ConsoleRunStatus.IDLE,
            )
        ),
    ],
)
@pytest.mark.asyncio
async def test_run_gating_parametrized(run_status: str, expect_armed: bool):
    """Only the canonical ``FEEDBACK_ACTIVE_RUN_STATUSES`` arm Request
    changes / LGTM; Comment never gates and the hint appears exactly
    when the other two are gated."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.TOOL, run_status=run_status)
    async with app.run_test(size=(80, 40)) as pilot:
        await _drag_select_first_row(pilot, app)
        menu = app.query_one(ConsoleSelectionMenu)
        assert (
            menu.query_one("#console-selection-request-changes", Button).disabled
            is not expect_armed
        )
        assert (
            menu.query_one("#console-selection-lgm", Button).disabled is not expect_armed
        )
        assert not menu.query_one("#console-selection-comment", Button).disabled
        if expect_armed:
            assert not menu.query("#console-selection-feedback-hint")
        else:
            assert menu.query_one("#console-selection-feedback-hint")


@pytest.mark.asyncio
async def test_comment_posts_selection_feedback_requested_and_cleans_up():
    """Pressing Comment posts one app-level ConsoleSelectionFeedbackRequested
    carrying the capped quote, then clears the whole selection UI."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.TOOL, run_status="idle")
    async with app.run_test(size=(80, 40)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _drag_select_first_row(pilot, app)
        assert row.get_selection_text() == "tool"  # control: selection live

        await pilot.click("#console-selection-comment")
        await pilot.pause()

        assert len(app.feedback_events) == 1
        event = app.feedback_events[0]
        assert event.action == "comment"
        assert event.quote == "tool"
        # Cleanup mirrors Add-to-chat: highlight cleared, drag state
        # cancelled, origin row dropped, menu removed.
        assert row.get_selection_text() == ""
        assert transcript.selection_manager.state.selection is None
        assert transcript._selection_origin_row is None
        assert not app.screen.query(ConsoleSelectionMenu)


# ---------------------------------------------------------------------------
# Selection feedback routing (phase 3, task 5)
# ---------------------------------------------------------------------------


class _RecordingPromptQueue:
    """Test double for ``ConsolePromptQueueUIController``: records dispatches.

    The real controller owns the send semantics (queue behind an active
    run, send immediately otherwise, refusal toasts); the routing tests
    only need to see WHAT text the screen hands it -- and that nothing
    else is touched, especially not the live composer draft.
    """

    def __init__(self) -> None:
        self.dispatched: list[str] = []

    async def dispatch(self, draft: str, *, stash: object = None) -> None:
        self.dispatched.append(draft)


def _stub_comment_modal(screen, comment: str | None) -> list:
    """Replace the app's ``push_screen_wait`` with a canned resolver.

    Records every modal instance the handler pushes (so tests can pin the
    action/quote it was built with) and resolves with ``comment``.
    """
    pushed: list[ConsoleFeedbackCommentModal] = []

    async def _resolve(modal, *args, **kwargs):
        pushed.append(modal)
        return comment

    screen.app.push_screen_wait = _resolve  # type: ignore[method-assign]
    return pushed


async def _run_feedback_request(
    pilot, *, action: str, quote: str, comment: str | None
):
    """Post one feedback request on the real console screen, stubbed seams.

    Returns ``(queue, pushed_modals, composer, draft_before)``; the
    composer is pre-loaded with an in-progress draft so "the draft is
    untouched" is a real assertion, not a vacuous one.
    """
    screen = pilot.app.screen
    queue = _RecordingPromptQueue()
    screen._prompt_queue = queue
    pushed = _stub_comment_modal(screen, comment)
    composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
    composer.insert_text("in-progress user draft")
    draft_before = composer.draft_text()
    screen.post_message(ConsoleSelectionFeedbackRequested(action=action, quote=quote))
    await pilot.pause()
    await pilot.pause()
    return queue, pushed, composer, draft_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "header"),
    [
        ("request-changes", "[Request changes]"),
        ("lgm", "[LGTM]"),
        ("comment", "[Comment]"),
    ],
)
async def test_feedback_routes_composed_message_via_prompt_queue(action, header):
    """Header + quoted selection + comment, dispatched through the queue.

    The ONLY send seam: the text goes to ``_prompt_queue.dispatch`` (which
    queues behind an active run / sends immediately), never to the
    composer draft.
    """
    async with make_console_pilot() as pilot:
        queue, pushed, composer, draft_before = await _run_feedback_request(
            pilot,
            action=action,
            quote="fix the retry loop",
            comment="tighten error paths",
        )
        assert queue.dispatched == [
            f"{header}\n> fix the retry loop\ntighten error paths"
        ]
        assert len(pushed) == 1
        assert pushed[0]._action == action
        assert pushed[0]._quote == "fix the retry loop"
        assert composer.draft_text() == draft_before == "in-progress user draft"


@pytest.mark.asyncio
async def test_feedback_without_comment_omits_comment_block():
    """A comment-less submit composes header + quote only."""
    async with make_console_pilot() as pilot:
        queue, _, composer, draft_before = await _run_feedback_request(
            pilot, action="comment", quote="the selection", comment=""
        )
        assert queue.dispatched == ["[Comment]\n> the selection"]
        assert composer.draft_text() == draft_before


@pytest.mark.asyncio
async def test_feedback_multiline_quote_prefixes_every_line():
    """Every quoted line gains ``> ``; blank lines become a bare ``>``.

    Mirrors ``ConsoleComposerBar.insert_quote``'s block-quote rendering so
    the same selection reads identically in the draft and in feedback.
    """
    async with make_console_pilot() as pilot:
        queue, _, composer, draft_before = await _run_feedback_request(
            pilot,
            action="request-changes",
            quote="line one\nline two\n\nline four",
            comment="rework this",
        )
        assert queue.dispatched == [
            "[Request changes]\n> line one\n> line two\n>\n> line four\nrework this"
        ]
        assert composer.draft_text() == draft_before


@pytest.mark.asyncio
async def test_feedback_empty_quote_dispatches_nothing():
    """A cleared row range dispatches nothing -- the modal never opens.

    Same blank-selection window as the Add-to-chat / side-chat guards:
    the row range was cleared while the menu was open.
    """
    async with make_console_pilot() as pilot:
        queue, pushed, composer, draft_before = await _run_feedback_request(
            pilot, action="comment", quote="   \n  ", comment="never reached"
        )
        assert queue.dispatched == []
        assert pushed == []
        assert composer.draft_text() == draft_before


@pytest.mark.asyncio
async def test_feedback_modal_escape_cancels_without_dispatch():
    """Escape/cancel (modal returns None) abandons the whole feedback."""
    async with make_console_pilot() as pilot:
        queue, pushed, composer, draft_before = await _run_feedback_request(
            pilot, action="request-changes", quote="fix this", comment=None
        )
        assert len(pushed) == 1  # the modal did open...
        assert queue.dispatched == []  # ...and its cancellation sent nothing
        assert composer.draft_text() == draft_before


@pytest.mark.asyncio
async def test_feedback_real_modal_submit_dispatches_composed_text():
    """Full loop with the real modal: request -> type -> Submit -> queue."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        queue = _RecordingPromptQueue()
        screen._prompt_queue = queue
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("untouched draft")
        screen.post_message(
            ConsoleSelectionFeedbackRequested(action="lgm", quote="ship it")
        )
        await pilot.pause()

        modal = pilot.app.screen
        assert isinstance(modal, ConsoleFeedbackCommentModal)
        modal.query_one("#console-feedback-comment-input", Input).value = "nice work"
        await pilot.click("#console-feedback-comment-submit")
        await pilot.pause()
        await pilot.pause()

        assert queue.dispatched == ["[LGTM]\n> ship it\nnice work"]
        assert composer.draft_text() == "untouched draft"


@pytest.mark.asyncio
async def test_feedback_real_modal_empty_submit_omits_comment_block():
    """Full loop with the real modal: request -> empty Submit -> queue.

    The comment is optional (spec §3): an empty submit dismisses ``""`` (NOT
    ``None``), so the feedback still dispatches — header + quote only, no
    comment block. This keeps the modal's `if comment:` branch live through
    the real loop, not just via the stubbed-modal tests above.
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        queue = _RecordingPromptQueue()
        screen._prompt_queue = queue
        screen.post_message(
            ConsoleSelectionFeedbackRequested(action="lgm", quote="ship it")
        )
        await pilot.pause()

        modal = pilot.app.screen
        assert isinstance(modal, ConsoleFeedbackCommentModal)
        assert modal.query_one("#console-feedback-comment-input", Input).value == ""
        await pilot.click("#console-feedback-comment-submit")
        await pilot.pause()
        await pilot.pause()

        assert queue.dispatched == ["[LGTM]\n> ship it"]


@pytest.mark.asyncio
async def test_feedback_real_modal_escape_dispatches_nothing():
    """Full loop with the real modal: request -> Escape -> nothing sent."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        queue = _RecordingPromptQueue()
        screen._prompt_queue = queue
        screen.post_message(
            ConsoleSelectionFeedbackRequested(action="comment", quote="a note")
        )
        await pilot.pause()
        assert isinstance(pilot.app.screen, ConsoleFeedbackCommentModal)

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert queue.dispatched == []
        assert pilot.app.screen is screen  # modal popped; console back on top


@pytest.mark.asyncio
async def test_drag_release_click_never_wipes_selection_for_menu_actions():
    """Live spike 2026-08-16 ("buttons don't work after the first one").

    The drag's synthesized release Click reaches the transcript's
    ``on_click`` with ``just_finished`` already consumed by the row guard
    (whose ``or`` chain short-circuits past ``consume_release_click`` and
    does not stop the event). The transcript then treated the artifact as
    a click-outside: ``_remove_selection_menu()`` wiped the row selection
    BEFORE the menu's action read it, so every selection-dependent action
    saw an empty quote and silently no-oped -- the first menu of a session
    usually won the queue race, later ones reliably lost it. The release
    click must die at the row (both tokens consumed, event stopped), and
    the transcript's own guard must run before any dismissal cleanup.
    """
    from textual.events import MouseDown, MouseMove, MouseUp

    from tldw_chatbook.Widgets.Console.console_selection_menu import (
        ConsoleSelectionMenu,
    )
    from tldw_chatbook.Widgets.Console.console_side_chat_modal import (
        ConsoleSideChatModal,
    )

    def raw(event_cls, x, y, button=0):
        return event_cls(
            widget=None, x=x, y=y, delta_x=0, delta_y=0, button=button,
            shift=False, meta=False, ctrl=False,
        )

    async def drag_menu_and_click_ask(pilot) -> bool:
        screen = pilot.app.screen
        row = screen.query_one("#console-message-mm0")
        md = row.query_one(".console-markdown-body")
        br = md.region
        pilot.app.post_message(raw(MouseDown, br.x + 2, br.y, button=1))
        await pilot.pause()
        pilot.app.post_message(raw(MouseMove, br.x + 10, br.y, button=0))
        await pilot.pause()
        pilot.app.post_message(raw(MouseUp, br.x + 10, br.y, button=1))
        await pilot.pause()
        await pilot.pause()
        menus = screen.query(ConsoleSelectionMenu)
        if not menus:
            return False
        menu = menus[0]
        sel_at_click = row.get_selection_text()
        btn = menu.query_one("#console-selection-ask-side-chat")
        r = btn.region
        cx, cy = r.x + r.width // 2, r.y
        pilot.app.post_message(raw(MouseMove, cx, cy, button=0))
        await pilot.pause()
        pilot.app.post_message(raw(MouseDown, cx, cy, button=1))
        await pilot.pause()
        pilot.app.post_message(raw(MouseUp, cx, cy, button=1))
        await pilot.pause()
        await pilot.pause()
        modals = [s for s in pilot.app.screen_stack if isinstance(s, ConsoleSideChatModal)]
        return bool(modals) and bool(sel_at_click)

    async with make_console_pilot(size=(80, 32)) as pilot:
        screen = pilot.app.screen
        transcript = screen.query_one(ConsoleTranscript)
        from tldw_chatbook.Chat.console_chat_models import (
            ConsoleChatMessage,
            ConsoleMessageRole,
        )

        transcript.set_messages([
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="first selection text",
                id="mm0",
            )
        ])
        await transcript.refresh_messages()
        await pilot.pause(0.4)

        first = await drag_menu_and_click_ask(pilot)
        assert first, "first side chat should open with the quote intact"
        await pilot.press("escape")
        await pilot.pause(0.3)

        transcript.set_messages([
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="second selection text",
                id="mm0",
            )
        ])
        await transcript.refresh_messages()
        await pilot.pause(0.4)
        row = screen.query_one("#console-message-mm0")
        assert row.get_selection_text() == ""  # clean slate for round two

        second = await drag_menu_and_click_ask(pilot)
        assert second, (
            "second selection's menu action must still carry the quote "
            "(drag-release click wiped it before the action read it)"
        )
