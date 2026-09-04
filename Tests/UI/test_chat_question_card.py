"""PRD Feature A: the question card, its state plumbing, and the card slot."""

from __future__ import annotations

import pytest
from textual import on
from textual.app import ComposeResult
from textual.widgets import Input, RadioButton, RadioSet, SelectionList, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


def _payload(
    n_questions: int = 2,
    n_options: int = 2,
    *,
    request_id: str = "round-1",
    timeout: float = 0.0,
):
    return {
        "request_id": request_id,
        "session_id": "s1",
        "timeout_seconds": timeout,
        "deadline_monotonic": None,
        "asked_by": "agent",
        "questions": [
            {
                "question": f"Question {q}?",
                "header": f"Q{q}",
                "multiSelect": q % 2 == 1,
                "options": [
                    {"label": f"opt{q}{o}", "description": f"desc {o}"}
                    for o in range(n_options)
                ],
            }
            for q in range(n_questions)
        ],
    }


# --- state ---------------------------------------------------------------


def test_state_carries_and_serializes_a_pending_question():
    state = TaskResumeState(pending_question=_payload())
    assert state.has_pending_question() is True
    assert state.to_dict()["pending_question"]["request_id"] == "round-1"


def test_restored_state_drops_the_pending_question_so_no_dead_card_appears():
    restored = TaskResumeState.from_dict(
        TaskResumeState(pending_question=_payload()).to_dict()
    )
    assert restored.pending_question is None


# --- card under the real CSS ---------------------------------------------


class _Harness(ConsolidatedCSSApp):
    def __init__(self):
        super().__init__()
        self.answered = []

    def compose(self) -> ComposeResult:
        yield ChatTaskCards(id="chat-task-cards")

    @on(ChatTaskCards.QuestionAnswered)
    def _record(self, event) -> None:
        self.answered.append((event.request_id, event.answers))


async def _mount(app, pilot, payload):
    cards = app.query_one(ChatTaskCards)
    cards.sync_state(TaskResumeState(pending_question=payload))
    await pilot.pause()
    await pilot.pause()
    return cards.query_one("#chat-question-card")


@pytest.mark.asyncio
async def test_card_is_absent_until_a_question_arrives_then_renders_every_section():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        cards = app.query_one(ChatTaskCards)
        assert not list(cards.query("#chat-question-card")), "lazy: nothing mounted at boot"
        card = await _mount(app, pilot, _payload())
        assert cards.display is True and card.display is True
        sections = list(card.query(".question-section"))
        assert len(sections) == 2
        assert "2 questions" in str(card.query_one("#question-title", Static).render())
        assert isinstance(sections[0].query_one(".question-options"), RadioSet)
        assert isinstance(sections[1].query_one(".question-options"), SelectionList)
        assert all(section.query(Input) for section in sections), "Other on every question"
        assert str(card.query_one("#question-deadline", Static).render()) == ""


@pytest.mark.asyncio
async def test_submit_returns_selections_other_text_and_unanswered_with_the_request_id():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(3))
        sections = list(card.query(".question-section"))
        list(sections[0].query(RadioButton))[1].value = True
        sections[1].query_one(SelectionList).select(0)
        sections[1].query_one(SelectionList).select(1)
        sections[2].query_one(Input).value = "something else"
        await pilot.pause()
        await pilot.click("#question-submit")
        await pilot.pause()
        assert app.answered == [
            (
                "round-1",
                [
                    {"question": "Question 0?", "selected": ["opt01"], "other_text": None, "unanswered": False},
                    {"question": "Question 1?", "selected": ["opt10", "opt11"], "other_text": None, "unanswered": False},
                    {"question": "Question 2?", "selected": [], "other_text": "something else", "unanswered": False},
                ],
            ),
        ]
        assert card.display is False


@pytest.mark.asyncio
async def test_partial_submit_marks_the_skipped_question_unanswered():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(2))
        first_section = next(iter(card.query(".question-section")))
        next(iter(first_section.query(RadioButton))).value = True
        await pilot.pause()
        await pilot.click("#question-submit")
        await pilot.pause()
        ((_, answers),) = app.answered
        assert answers[1] == {
            "question": "Question 1?",
            "selected": [],
            "other_text": None,
            "unanswered": True,
        }


@pytest.mark.asyncio
async def test_number_keys_select_within_the_focused_question():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(1, 3))
        picker = card.query_one(RadioSet)
        picker.focus()
        await pilot.pause()
        await pilot.press("3")
        await pilot.pause()
        assert picker.pressed_index == 2
        other = card.query_one(Input)
        other.focus()
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()
        assert other.value == "2", "digits typed into Other are text, not selections"
        assert picker.pressed_index == 2


@pytest.mark.asyncio
async def test_enter_submits_from_anywhere_in_the_card():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(1))
        card.query_one(RadioSet).focus()
        await pilot.pause()
        await pilot.press("2")
        await pilot.press("enter")
        await pilot.pause()
        assert app.answered and app.answered[0][1][0]["selected"] == ["opt01"]


@pytest.mark.asyncio
async def test_resync_of_the_same_round_keeps_the_users_selection():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(1))
        list(card.query(RadioButton))[1].value = True
        await pilot.pause()
        app.query_one(ChatTaskCards).sync_state(
            TaskResumeState(pending_question=_payload(1, timeout=30))
        )
        await pilot.pause()
        assert card.query_one(RadioSet).pressed_index == 1, "same request_id: no rebuild"
        assert (
            str(card.query_one("#question-deadline", Static).render())
            == "Auto-continues in 0:30"
        )
        app.query_one(ChatTaskCards).sync_state(
            TaskResumeState(pending_question=_payload(1, request_id="round-2"))
        )
        await pilot.pause()
        await pilot.pause()
        fresh = card.query_one("#question-round-2-0").query_one(RadioSet)
        assert fresh.pressed_index == -1, "new round: fresh sections"


@pytest.mark.asyncio
async def test_clearing_hides_the_card_and_the_slot():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(1))
        cards = app.query_one(ChatTaskCards)
        cards.sync_state(TaskResumeState())
        await pilot.pause()
        assert card.display is False and cards.display is False


@pytest.mark.asyncio
async def test_four_by_four_card_stays_bounded_under_bundled_css():
    """AC-A13: 4 questions x 4 described options must not eat the transcript."""
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(4, 4))
        await pilot.pause()
        assert card.region.height <= 24, card.region
        sections = card.query_one("#question-sections")
        assert sections.region.height <= 15, sections.region
        submit = card.query_one("#question-submit")
        assert submit.region.height > 0 and submit.region.y < 40, "Submit stays reachable"
