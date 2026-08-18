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
import json
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


@pytest.mark.asyncio
async def test_feedback_event_carries_the_quoted_row_as_its_anchor():
    """task-17169: durable feedback has to say WHAT it was about. The quote
    alone cannot survive a re-render or a re-run -- the anchor is the row's
    message id, which the sidecar row is keyed to."""
    app = _FeedbackTranscriptApp(role=ConsoleMessageRole.TOOL, run_status="idle")
    async with app.run_test(size=(80, 40)) as pilot:
        row = await _drag_select_first_row(pilot, app)

        await pilot.click("#console-selection-comment")
        await pilot.pause()

        assert app.feedback_events[0].anchor_message_id == row.message_id


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

    def presentation_for(self, session_id, **kwargs):
        """The sync tick reads the queue presentation unconditionally; hand
        it the REAL dataclass in its idle empty-queue shape so every widget
        consumer (chips, shelf, send button) sees a total object."""
        from tldw_chatbook.UI.Console_Modules.prompt_queue import (
            ConsolePromptQueuePresentation,
        )

        return ConsolePromptQueuePresentation(
            revision=0,
            count=0,
            send_label="Send",
            send_enabled=True,
            send_tooltip="",
            shelf_visible=False,
            state_label="",
            paused=False,
            next_preview="",
            pause_label="Pause",
            primary_action="send",
            pause_enabled=False,
        )

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
    pilot, *, action: str, quote: str, comment: str | None, anchor_message_id=None
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
    screen.post_message(
        ConsoleSelectionFeedbackRequested(
            action=action, quote=quote, anchor_message_id=anchor_message_id
        )
    )
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


# ---------------------------------------------------------------------------
# Durable feedback audit record (task-17169, phase 4)
# ---------------------------------------------------------------------------


class _RecordingStore:
    """Captures record_feedback_event / record_feedback_annotation calls."""

    def __init__(self, result=True, boom=False, boom_annotation=False):
        self.calls: list[dict] = []
        self.annotation_calls: list[dict] = []
        self._result = result
        self._boom = boom
        self._boom_annotation = boom_annotation

    def record_feedback_event(self, session_id, **kwargs):
        self.calls.append({"session_id": session_id, **kwargs})
        if self._boom:
            raise RuntimeError("store exploded")
        return self._result

    def record_feedback_annotation(self, session_id, **kwargs):
        self.annotation_calls.append({"session_id": session_id, **kwargs})
        if self._boom or self._boom_annotation:
            raise RuntimeError("store exploded")
        return "anno-1" if self._result else None


def _stub_feedback_store(screen, store):
    controller = screen._ensure_console_chat_controller()
    controller.store.record_feedback_event = store.record_feedback_event  # type: ignore[method-assign]
    controller.store.record_feedback_annotation = store.record_feedback_annotation  # type: ignore[method-assign]
    return controller


@pytest.mark.asyncio
async def test_dispatched_feedback_is_recorded_against_its_anchor():
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore()
        controller = _stub_feedback_store(screen, store)

        await _run_feedback_request(
            pilot,
            action="request-changes",
            quote="fix the retry loop",
            comment="tighten error paths",
            anchor_message_id="msg-42",
        )

        assert store.calls == [
            {
                "session_id": controller.store.active_session_id,
                "anchor_message_id": "msg-42",
                "action": "request-changes",
                "quote": "fix the retry loop",
                "comment": "tighten error paths",
            }
        ]


@pytest.mark.asyncio
async def test_abandoned_feedback_records_nothing():
    """Escape/Cancel abandons the whole feedback -- there is no event to
    audit, so the ledger must not gain a row for it."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore()
        _stub_feedback_store(screen, store)

        queue, _pushed, _composer, _draft = await _run_feedback_request(
            pilot,
            action="lgm",
            quote="ship it",
            comment=None,
            anchor_message_id="msg-42",
        )

        assert queue.dispatched == []
        assert store.calls == []


@pytest.mark.asyncio
async def test_feedback_without_an_anchor_still_dispatches():
    """No origin row means no audit anchor -- but the user's feedback is the
    point, and losing it over a missing audit record would be backwards."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore()
        _stub_feedback_store(screen, store)

        queue, *_ = await _run_feedback_request(
            pilot,
            action="lgm",
            quote="ship it",
            comment="",
            anchor_message_id=None,
        )

        assert queue.dispatched == ["[LGTM]\n> ship it"]
        assert store.calls == []


@pytest.mark.asyncio
async def test_a_failing_audit_write_never_costs_the_user_their_feedback():
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore(boom=True)
        _stub_feedback_store(screen, store)

        queue, *_ = await _run_feedback_request(
            pilot,
            action="lgm",
            quote="ship it",
            comment="",
            anchor_message_id="msg-42",
        )

        assert len(store.calls) == 1
        assert queue.dispatched == ["[LGTM]\n> ship it"]


@pytest.mark.asyncio
async def test_feedback_reaches_the_real_database_unmocked(tmp_path):
    """The screen -> store -> DB chain with NOTHING stubbed on the store.

    Every other test here replaces ``record_feedback_event`` with a double,
    which proves the screen CALLS it but would pass just as happily if the
    store's own write were inert. This repo has shipped an inert feature
    behind exactly that shape of mock before, so the real chain gets its own
    test.

    The console harness app carries no ChaChaNotes DB, so its store is built
    without persistence. The test installs a real one through the screen's
    own ``_console_chat_store`` setter (plus the controller's cached reference,
    which the runtime swap does not reach) -- so the only thing the test does
    is CONSTRUCT the store; every write after that is production code against
    a real database.
    """
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "selection_e2e")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            screen._prompt_queue = _RecordingPromptQueue()
            _stub_comment_modal(screen, "needs a retry bound")
            store = ConsoleChatStore(persistence=ChatPersistenceService(db))
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            # The controller caches the store it was built with, so the
            # runtime swap alone would leave the screen writing to the
            # harness's persistence-less store.
            controller.store = store
            assert getattr(store.persistence, "db", None) is db  # control

            session = store.ensure_session(title="Feedback e2e")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="retrying forever on 429",
                persist=True,
            )

            screen.post_message(
                ConsoleSelectionFeedbackRequested(
                    action="request-changes",
                    quote="retrying forever on 429",
                    anchor_message_id=assistant.id,
                )
            )
            await pilot.pause()
            await pilot.pause()

            rows = [
                row
                for row in db.get_trajectory_rows(conversation_id)
                if row.event_kind == "user_feedback"
            ]
            assert len(rows) == 1, "no user_feedback row reached the database"
            assert rows[0].message_id == assistant.persisted_message_id
            payload = json.loads(rows[0].payload_json)
            assert payload["action"] == "request-changes"
            assert payload["comment"] == "needs a retry bound"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_comment_with_text_also_records_an_annotation():
    """Slice 2 of the both-homes decision: Comment persists an annotation in
    addition to its sidecar event. Only Comment -- the spec's "Comment ...
    additionally persists an annotation" -- and only with an actual note
    (an empty submit has nothing to mark the row with)."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore()
        controller = _stub_feedback_store(screen, store)

        await _run_feedback_request(
            pilot,
            action="comment",
            quote="the retry loop",
            comment="tighten error paths",
            anchor_message_id="msg-42",
        )

        assert store.annotation_calls == [
            {
                "session_id": controller.store.active_session_id,
                "anchor_message_id": "msg-42",
                "quote": "the retry loop",
                "comment": "tighten error paths",
            }
        ]
        # The sidecar audit event still fires alongside it.
        assert len(store.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["request-changes", "lgm"])
async def test_non_comment_actions_record_no_annotation(action):
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore()
        _stub_feedback_store(screen, store)

        await _run_feedback_request(
            pilot,
            action=action,
            quote="q",
            comment="a note",
            anchor_message_id="msg-42",
        )

        assert store.annotation_calls == []
        assert len(store.calls) == 1  # sidecar audit still recorded


@pytest.mark.asyncio
async def test_empty_comment_records_no_annotation():
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore()
        _stub_feedback_store(screen, store)

        queue, *_ = await _run_feedback_request(
            pilot,
            action="comment",
            quote="q",
            comment="",
            anchor_message_id="msg-42",
        )

        assert store.annotation_calls == []
        assert queue.dispatched == ["[Comment]\n> q"]


@pytest.mark.asyncio
async def test_failing_annotation_write_never_blocks_the_dispatch():
    """boom_annotation only: the sidecar write succeeds, the annotation write
    explodes -- the shared guard still lets the dispatch through."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore(boom_annotation=True)
        _stub_feedback_store(screen, store)

        queue, *_ = await _run_feedback_request(
            pilot,
            action="comment",
            quote="q",
            comment="note",
            anchor_message_id="msg-42",
        )

        assert len(store.annotation_calls) == 1
        assert queue.dispatched == ["[Comment]\n> q\nnote"]


@pytest.mark.asyncio
async def test_comment_annotation_reaches_the_real_database_unmocked(tmp_path):
    """Same mock-gap closure as the sidecar's unmocked test, for slice 2:
    a Comment's annotation row read back out of real SQLite."""
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "anno_e2e")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            screen._prompt_queue = _RecordingPromptQueue()
            _stub_comment_modal(screen, "tighten error paths")
            store = ConsoleChatStore(persistence=ChatPersistenceService(db))
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            controller.store = store

            session = store.ensure_session(title="Annotation e2e")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="the retry loop",
                persist=True,
            )

            screen.post_message(
                ConsoleSelectionFeedbackRequested(
                    action="comment",
                    quote="the retry loop",
                    anchor_message_id=assistant.id,
                )
            )
            await pilot.pause()
            await pilot.pause()

            rows = db.get_transcript_annotations(conversation_id)
            assert len(rows) == 1, "no annotation row reached the database"
            assert rows[0]["row_key"] == f"message:{assistant.persisted_message_id}"
            assert rows[0]["comment"] == "tighten error paths"
            # And the sidecar audit event landed alongside it.
            feedback_rows = [
                row
                for row in db.get_trajectory_rows(conversation_id)
                if row.event_kind == "user_feedback"
            ]
            assert len(feedback_rows) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_rapid_double_trigger_dispatches_one_feedback():
    """Qodo (PR #1723, d63cd21f0): the flow worker is deliberately
    non-exclusive (an exclusive cancel would strand a mounted modal -- the
    EvalsScreen rationale), so mutual exclusion is a guard instead: a second
    request while one flow is in flight is ignored. Phase 4 raised the
    stakes from a duplicate chat message to duplicate DURABLE records."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        queue = _RecordingPromptQueue()
        screen._prompt_queue = queue
        pushed = _stub_comment_modal(screen, "once")
        store = _RecordingStore()
        _stub_feedback_store(screen, store)

        for _ in range(2):
            screen.post_message(
                ConsoleSelectionFeedbackRequested(
                    action="comment", quote="q", anchor_message_id="msg-42"
                )
            )
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        assert len(pushed) == 1
        assert queue.dispatched == ["[Comment]\n> q\nonce"]
        assert len(store.calls) == 1
        assert len(store.annotation_calls) == 1


@pytest.mark.asyncio
async def test_feedback_flow_can_run_again_after_completion():
    """The in-flight guard must clear on every exit path -- a latched flag
    would silently kill the feature after its first use."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        queue = _RecordingPromptQueue()
        screen._prompt_queue = queue
        _stub_comment_modal(screen, "note")
        store = _RecordingStore()
        _stub_feedback_store(screen, store)

        for round_no in range(2):
            screen.post_message(
                ConsoleSelectionFeedbackRequested(
                    action="lgm", quote=f"q{round_no}", anchor_message_id="msg-42"
                )
            )
            await pilot.pause()
            await pilot.pause()

        assert queue.dispatched == ["[LGTM]\n> q0\nnote", "[LGTM]\n> q1\nnote"]


@pytest.mark.asyncio
async def test_keyboard_only_journey_selects_and_dispatches_feedback():
    """Phase 5 e2e: j/k -> s -> motions -> Enter -> Comment, no mouse at any
    step until the menu (whose buttons are the same either way). The
    keyboard path must produce the same dispatch and the same durable
    records as a mouse drag."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        controller = screen._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        assistant = controller.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="keyboard journey target",
            persist=False,
        )
        # Sync with the REAL queue still installed -- the tick consults it
        # (presentation_for); the doubles go in only once the rows exist.
        await screen._sync_native_console_transcript()
        await pilot.pause()
        queue = _RecordingPromptQueue()
        screen._prompt_queue = queue
        _stub_comment_modal(screen, "kb note")
        store = _RecordingStore()
        _stub_feedback_store(screen, store)

        transcript = screen.query_one(ConsoleTranscript)
        transcript.focus()
        await pilot.pause()
        # j selects the (only) message; s arms the mode; l l extends.
        await pilot.press("j")
        assert transcript.selected_message_id == assistant.id
        await pilot.press("s")
        assert transcript._kb_selection_row is not None, "s did not arm the mode"
        await pilot.press("l", "l")
        assert transcript._kb_selection_row is not None, "mode lost after motions"
        await pilot.press("enter")
        await pilot.pause()

        menu = screen.query_one(ConsoleSelectionMenu)
        assert not menu.query_one("#console-selection-comment").disabled
        await pilot.click("#console-selection-comment")
        await pilot.pause()
        await pilot.pause()

        quote = "key"  # 3 chars: entry selected 1, l l extended to 3
        assert queue.dispatched == [f"[Comment]\n> {quote}\nkb note"]
        assert store.calls[0]["anchor_message_id"] == assistant.id
        assert store.calls[0]["quote"] == quote
        assert store.annotation_calls[0]["comment"] == "kb note"
