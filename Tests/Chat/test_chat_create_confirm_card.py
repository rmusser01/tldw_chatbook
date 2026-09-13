"""The chat-create confirm card and its task-state plumbing.

Covers the widget half (``ChatCreateConfirmCard.set_payload``/
``ChatCreateDecided``), the ``TaskResumeState.pending_chat_create``
plumbing, and the ``ChatTaskCards`` mount/display gate from
``.superpowers/sdd/2026-09-11-agent-chat-fork-spawn/task-6-brief.md``.

The single load-bearing contract under test throughout this file is the
``request_id`` round-trip (task-5, ``console_chat_controller.py``): a
decision that does not echo back the pending round's exact ``request_id``
is silently dropped by ``ConsoleChatController.resolve_pending_chat_create``,
leaving the worker thread blocked until its timeout.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual import on
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Chat_Widgets.chat_create_confirm_card import (
    ChatCreateConfirmCard,
)


def _payload(**over):
    base = {
        "tool": "fork_chat",
        "title": "W: DB migration",
        "opening_prompt": "Please plan the schema migration.",
        "instructions": "Focus only on the DB migration.",
        "fork_source_title": "API redesign",
        "fork_message_count": 12,
        "request_id": "r-1",
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    "allow,remember",
    [(True, False), (True, True), (False, False)],
)
def test_decisions_carry_request_id(allow, remember):
    card = ChatCreateConfirmCard()
    card.set_payload(_payload())
    messages = []
    card.post_message = lambda m: messages.append(m)  # capture instead of pump
    card._decide(allow=allow, remember=remember)
    assert len(messages) == 1
    decided = messages[0]
    assert decided.allow is allow
    assert decided.remember is remember
    assert decided.request_id == "r-1"


def test_clear_payload_hides_card():
    card = ChatCreateConfirmCard()
    card.set_payload(_payload())
    card.set_payload(None)
    assert card.display is False


def test_header_reflects_tool():
    card = ChatCreateConfirmCard()
    card.set_payload(_payload(tool="new_chat"))
    assert "new chat" in card._header_text().lower()
    card.set_payload(_payload())
    assert "fork" in card._header_text().lower()


# ---------------------------------------------------------------------------
# Final-review fix wave (Finding 1): the fork-summary line renders only when
# the controller's enrichment keys are PRESENT, and the run id line renders
# only when the payload carries a run_id.
# ---------------------------------------------------------------------------


def test_body_renders_fork_line_when_enrichment_keys_present():
    card = ChatCreateConfirmCard()
    card.set_payload(_payload())
    body = card._body_text()
    assert "Copies 12 messages from 'API redesign' into the new chat." in body


def test_body_omits_fork_line_when_enrichment_keys_absent():
    """An un-enriched payload (no producer set the keys) must not render the
    dead "Copies ? messages from ''" line."""
    payload = _payload()
    del payload["fork_source_title"]
    del payload["fork_message_count"]
    card = ChatCreateConfirmCard()
    card.set_payload(payload)
    assert "Copies" not in card._body_text()


def test_body_renders_present_zero_count():
    """A present count of 0 is honest (an empty fork source) and renders."""
    card = ChatCreateConfirmCard()
    card.set_payload(_payload(fork_message_count=0, fork_source_title="Empty"))
    assert "Copies 0 messages from 'Empty' into the new chat." in card._body_text()


def test_body_shows_run_id_when_present():
    card = ChatCreateConfirmCard()
    card.set_payload(_payload(run_id="run-9"))
    assert "Requested by agent run run-9." in card._body_text()


def test_body_omits_run_id_line_when_absent():
    payload = _payload()
    payload.pop("run_id", None)
    card = ChatCreateConfirmCard()
    card.set_payload(payload)
    assert "Requested by agent run" not in card._body_text()


class _CardHarnessApp(ConsolidatedCSSApp):
    """Minimal host for a caller-built `ChatCreateConfirmCard`."""

    def __init__(self, card: ChatCreateConfirmCard) -> None:
        super().__init__()
        self._card = card

    def compose(self) -> ComposeResult:
        yield self._card


@pytest.mark.asyncio
async def test_payload_set_before_mount_renders_on_mount():
    """set_payload tolerates an unmounted card (the bare-widget tests above
    rely on it); mounting afterwards must still render and show it."""
    from textual.widgets import Static

    card = ChatCreateConfirmCard()
    card.set_payload(_payload())
    app = _CardHarnessApp(card)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert card.display is True
        header = str(card.query_one("#chat-create-header", Static).render())
        assert "fork this chat" in header.lower()


# ---------------------------------------------------------------------------
# TaskResumeState plumbing
# ---------------------------------------------------------------------------


def test_state_carries_and_serializes_a_pending_chat_create():
    state = TaskResumeState(pending_chat_create={"tool": "fork_chat"})
    assert state.has_pending_chat_create() is True
    assert state.to_dict()["pending_chat_create"] == {"tool": "fork_chat"}


def test_restored_state_drops_the_pending_chat_create_so_no_dead_card_appears():
    """A restored pending chat-create must never come back as an actionable
    card -- the confirm it belongs to is a live worker round keyed by
    ``request_id`` on the controller, and a round that survived a save/
    restore cannot still be armed (see ``from_dict``'s docstring for the
    skill-confirm fields this mirrors)."""
    state = TaskResumeState(
        summary="Keep me",
        pending_chat_create={"tool": "fork_chat", "request_id": "round-1"},
    )
    restored = TaskResumeState.from_dict(state.to_dict())
    assert restored.pending_chat_create is None
    assert restored.has_pending_chat_create() is False
    assert restored.summary == "Keep me"


def test_state_without_a_pending_chat_create():
    assert TaskResumeState().has_pending_chat_create() is False


# ---------------------------------------------------------------------------
# ChatTaskCards mount/display gate + request_id round-trip through the
# real mount path (Task 5 review carry-forward).
# ---------------------------------------------------------------------------


class _CardsHarnessApp(ConsolidatedCSSApp):
    """Minimal host for `ChatTaskCards` that records `ChatCreateDecided`."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[tuple[bool, bool]] = []
        self.request_ids: list[str | None] = []

    def compose(self) -> ComposeResult:
        yield ChatTaskCards()

    @on(ChatCreateConfirmCard.ChatCreateDecided)
    def _capture(self, event: ChatCreateConfirmCard.ChatCreateDecided) -> None:
        self.decisions.append((event.allow, event.remember))
        self.request_ids.append(event.request_id)


@pytest.mark.asyncio
async def test_task_cards_mount_and_round_trip_a_chat_create_decision():
    """sync_state shows the card, and a real button press echoes the armed
    round's exact request_id back on ChatCreateDecided -- the id the screen
    forwards to resolve_pending_chat_create."""
    app = _CardsHarnessApp()
    async with app.run_test() as pilot:
        cards = app.query_one(ChatTaskCards)
        # ADR-097 lazy mount: the card module is not resident until a
        # pending payload first mounts it (same pattern as the question
        # card), so an idle ChatTaskCards has NO card to query.
        from textual.css.query import NoMatches

        with pytest.raises(NoMatches):
            cards.query_one(ChatCreateConfirmCard)

        cards.sync_state(
            TaskResumeState(pending_chat_create=_payload(request_id="round-42"))
        )
        await pilot.pause()
        card = cards.query_one(ChatCreateConfirmCard)
        assert card.display is True
        assert cards.display is True

        await pilot.click("#chat-create-allow")
        await pilot.pause()
        assert app.decisions == [(True, False)]
        assert app.request_ids == ["round-42"]
        assert card.display is False

        # Tearing the round down hides the whole task-cards surface again.
        cards.sync_state(TaskResumeState())
        await pilot.pause()
        assert cards.display is False


# ---------------------------------------------------------------------------
# ChatScreen wiring: pending-chat-create state bridge + ChatCreateDecided
# handler. Mirrors Tests/UI/test_skill_script_confirm_card.py's screen-
# wiring section (mock host, ChatScreen constructed without a running app).
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chat_host():
    host = Mock()
    host.app_config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7,
        }
    }
    host.chat_sidebar_collapsed = False
    host.chat_right_sidebar_collapsed = False
    host.notify = Mock()
    host.run_worker = Mock()
    host.bell = Mock()
    host.chachanotes_db = Mock(db_path="/tmp/uat-card-test.db")
    return host


def test_set_console_pending_chat_create_preserves_other_resume_fields(
    mock_chat_host,
):
    screen = ChatScreen(mock_chat_host)
    screen.set_task_resume_state(
        TaskResumeState(summary="Keep me", last_step="Also keep")
    )

    payload = {"tool": "fork_chat", "title": "W: db", "request_id": "r1"}
    screen._skill._set_console_pending_chat_create(payload)

    state = screen._task_resume_state
    assert state.summary == "Keep me"
    assert state.last_step == "Also keep"
    assert state.pending_chat_create == payload

    screen._skill._set_console_pending_chat_create(None)
    assert screen._task_resume_state.pending_chat_create is None
    assert screen._task_resume_state.summary == "Keep me"


def test_chat_screen_forwards_chat_create_decided_to_controller_with_request_id(
    mock_chat_host,
):
    """The decision must carry the pending round's exact request_id
    through to resolve_pending_chat_create, or the resolve is silently
    dropped (task-5's ConsoleChatController)."""
    screen = ChatScreen(mock_chat_host)
    controller = Mock()
    screen._console_chat_controller = controller

    event = ChatCreateConfirmCard.ChatCreateDecided(True, True, "round-7")
    screen.handle_console_chat_create_decided(event)

    controller.resolve_pending_chat_create.assert_called_once_with(
        True, True, request_id="round-7"
    )


def test_chat_screen_chat_create_decided_handler_tolerates_no_controller(
    mock_chat_host,
):
    screen = ChatScreen(mock_chat_host)
    screen._console_chat_controller = None

    event = ChatCreateConfirmCard.ChatCreateDecided(True, False, "round-7")
    screen.handle_console_chat_create_decided(event)  # must not raise


def test_console_view_hooks_exposes_chat_create_sinks(mock_chat_host):
    """Live-UAT regression probe: the hooks dict the runtime attaches must
    carry both chat-create sinks (set_pending via the skill module's setter,
    complete via the screen's completion method)."""
    screen = ChatScreen(mock_chat_host)
    hooks = screen.console_view_hooks()
    assert "set_pending_chat_create" in hooks
    assert "complete_agent_chat_create" in hooks
    assert callable(hooks["complete_agent_chat_create"])
