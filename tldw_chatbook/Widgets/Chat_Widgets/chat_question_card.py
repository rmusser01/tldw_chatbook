"""The ``ask_user`` question card (PRD Feature A: A2-A5, A7, A11).

Mounted lazily by ``ChatTaskCards`` on the first pending question, in the
same slot above the transcript where approvals appear. One section per
question -- header chip, question text, options, an always-present "Other"
input -- inside a bounded, scrolling container so four described questions
cannot push the transcript off screen.

The round-trip contract: ``set_questions(payload)`` stores
``payload["request_id"]`` and ``QuestionAnswered`` echoes it back; the
controller strict-matches it, so a stale submit is dropped, never
misapplied to a newer round.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widgets import Button, Input, RadioButton, RadioSet, SelectionList, Static
from textual.widgets.selection_list import Selection

from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards

_NUMBER_KEYS = {"1": 0, "2": 1, "3": 2, "4": 3}


def format_question_deadline(timeout_seconds: float | None) -> str:
    """Return the countdown copy for an armed question deadline (PRD A7).

    Mirrors ``chat_approval_card.format_approval_deadline``: say nothing
    rather than invent a number.

    Args:
        timeout_seconds: Remaining seconds, or None/0 when no deadline.

    Returns:
        ``"Auto-continues in M:SS"`` or ``""``.
    """
    try:
        total = int(timeout_seconds or 0)
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    return f"Auto-continues in {total // 60}:{total % 60:02d}"


class ChatQuestionCard(Container):
    """Multiple-choice questions from the agent, answered in place."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit_answers", "Submit answers", show=False, priority=True),
    ]

    BUNDLED_CSS = """
    ChatQuestionCard {
        height: auto;
        max-height: 24;
        border: round $accent;
        padding: 0 1;
    }
    ChatQuestionCard > #question-title {
        height: 1;
        text-style: bold;
    }
    ChatQuestionCard > #question-deadline {
        height: auto;
        color: $text-muted;
    }
    ChatQuestionCard > #question-sections {
        height: auto;
        max-height: 15;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    ChatQuestionCard .question-section {
        height: auto;
        margin-bottom: 1;
    }
    ChatQuestionCard .question-header {
        height: 1;
        color: $accent;
        text-style: bold;
    }
    ChatQuestionCard .question-text {
        height: auto;
    }
    ChatQuestionCard .question-options {
        height: auto;
        border: none;
        padding: 0;
    }
    ChatQuestionCard .question-other {
        height: 3;
    }
    ChatQuestionCard > #question-actions {
        height: 3;
        align-horizontal: right;
    }
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Start hidden with no round; ``set_questions`` shows it."""
        super().__init__(*args, **kwargs)
        self.display = False
        self._payload: dict[str, Any] | None = None
        self._request_id: str | None = None
        self._rendered_request_id: str | None = None
        self._questions: list[dict[str, Any]] = []
        self._deadline_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Yield the title, deadline line, scrolling sections, and Submit.

        Returns:
            The composed child widgets.
        """
        yield Static("", id="question-title", markup=False)
        yield Static("", id="question-deadline", markup=False)
        yield VerticalScroll(id="question-sections")
        yield Horizontal(
            Button("Submit", id="question-submit", variant="primary"),
            id="question-actions",
        )

    def on_mount(self) -> None:
        """Paint a payload that arrived before the children existed."""
        if self._payload:
            self._paint()

    def set_questions(self, payload: dict[str, Any] | None) -> None:
        """Show the card for ``payload``, or hide it when None.

        A payload carrying the SAME ``request_id`` as the one on screen only
        refreshes the deadline copy -- ``ChatTaskCards.sync_state`` re-syncs
        every card on any task-state change, and rebuilding the sections
        would wipe the user's half-made selections.

        Args:
            payload: ``{"questions", "asked_by", "timeout_seconds",
                "request_id", ...}`` from the controller, or None.
        """
        if not payload:
            self._stop_deadline_timer()
            self.display = False
            self._payload = None
            self._request_id = None
            self._rendered_request_id = None
            self._questions = []
            try:
                self.query_one("#question-sections", VerticalScroll).remove_children()
            except NoMatches:
                pass
            return
        self._payload = dict(payload)
        self._request_id = payload.get("request_id")
        self._questions = [dict(q) for q in (payload.get("questions") or [])]
        self.display = True
        self._paint()

    def collect_answers(self) -> list[dict[str, Any]]:
        """Read every section into PRD A6 answer dicts, in question order.

        Returns:
            One ``{"question", "selected", "other_text", "unanswered"}`` per
            question; a question with neither a selection nor Other text is
            ``unanswered`` (A5: partial submission is allowed).
        """
        from tldw_chatbook.Agents.ask_user_questions import clean_other_text

        answers: list[dict[str, Any]] = []
        sections = list(self.query(".question-section"))
        for index, question in enumerate(self._questions):
            options = question.get("options") or []
            selected: list[str] = []
            other: str | None = None
            if index < len(sections):
                section = sections[index]
                if question.get("multiSelect"):
                    picker = section.query_one(SelectionList)
                    selected = [
                        str(options[i]["label"])
                        for i in picker.selected
                        if 0 <= i < len(options)
                    ]
                else:
                    radio = section.query_one(RadioSet)
                    if 0 <= radio.pressed_index < len(options):
                        selected = [str(options[radio.pressed_index]["label"])]
                other = clean_other_text(section.query_one(Input).value)
            answers.append(
                {
                    "question": str(question.get("question", "")),
                    "selected": selected,
                    "other_text": other,
                    "unanswered": not selected and other is None,
                }
            )
        return answers

    def action_submit_answers(self) -> None:
        """Submit whatever is answered (Enter anywhere in the card, A4/A5)."""
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Submit on the Submit button.

        Args:
            event: The button press; only ``#question-submit`` is consumed.
        """
        if event.button.id != "question-submit":
            return
        event.stop()
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter inside an Other box submits the whole card.

        Args:
            event: The Input's submit event.
        """
        event.stop()
        self._submit()

    def on_key(self, event: events.Key) -> None:
        """``1``-``4`` pick an option in the focused question's picker (A4).

        Digits typed into an Other input are text and are left alone.

        Args:
            event: The key event.
        """
        index = _NUMBER_KEYS.get(event.key)
        if index is None:
            return
        focused = self.app.focused
        if not isinstance(focused, (RadioSet, SelectionList)):
            return
        if self not in focused.ancestors:
            return
        event.stop()
        if isinstance(focused, RadioSet):
            buttons = list(focused.query(RadioButton))
            if index < len(buttons):
                buttons[index].value = True
        elif index < focused.option_count:
            focused.toggle(index)

    def _submit(self) -> None:
        self._stop_deadline_timer()
        answers = self.collect_answers()
        request_id = self._request_id
        self.display = False
        self.post_message(ChatTaskCards.QuestionAnswered(answers, request_id))

    def _paint(self) -> None:
        try:
            title = self.query_one("#question-title", Static)
            deadline = self.query_one("#question-deadline", Static)
            sections = self.query_one("#question-sections", VerticalScroll)
        except NoMatches:
            return
        payload = self._payload or {}
        label = str(payload.get("asker_label") or "").strip()[:40]
        if payload.get("asked_by") == "sub-agent":
            who = f"Sub-agent '{label}'" if label else "A sub-agent"
        else:
            who = "The agent"
        count = len(self._questions)
        title.update(f"{who} has {count} question{'s' if count != 1 else ''} for you:")
        self._sync_deadline(deadline)
        if self._rendered_request_id == self._request_id:
            return
        self._rendered_request_id = self._request_id
        sections.remove_children()
        key = (self._request_id or "none")[:8]
        sections.mount_all(
            self._build_section(key, index, question)
            for index, question in enumerate(self._questions)
        )

    def _remaining_seconds(self) -> float | None:
        """Seconds left on the round's absolute deadline, or None when unarmed."""
        payload = self._payload or {}
        deadline = payload.get("deadline_monotonic")
        if not deadline:
            return None
        return max(0.0, float(deadline) - time.monotonic())

    def _sync_deadline(self, label: Static) -> None:
        """Render the countdown and keep it ticking while a deadline is armed.

        The controller enforces ``deadline_monotonic``; the card only shows
        it. With a deadline the label counts down once a second from the
        remaining time (never the arm-time ``timeout_seconds`` a late mount
        would overstate); without one it shows the static copy for a
        payload that carries only ``timeout_seconds``, or nothing.

        Args:
            label: The ``#question-deadline`` Static.
        """
        remaining = self._remaining_seconds()
        if remaining is None:
            self._stop_deadline_timer()
            label.update(format_question_deadline((self._payload or {}).get("timeout_seconds")))
            return
        label.update(format_question_deadline(remaining) or "Auto-continues now")
        if remaining <= 0:
            self._stop_deadline_timer()
        elif self._deadline_timer is None:
            self._deadline_timer = self.set_interval(1.0, self._tick_deadline)

    def _tick_deadline(self) -> None:
        try:
            label = self.query_one("#question-deadline", Static)
        except NoMatches:
            self._stop_deadline_timer()
            return
        self._sync_deadline(label)

    def _stop_deadline_timer(self) -> None:
        if self._deadline_timer is not None:
            self._deadline_timer.stop()
            self._deadline_timer = None

    @staticmethod
    def _option_prompt(option: dict[str, Any]) -> str:
        label = str(option.get("label", ""))
        description = str(option.get("description") or "")
        return f"{label} — {description}" if description else label

    def _build_section(self, key: str, index: int, question: dict[str, Any]) -> Vertical:
        options = question.get("options") or []
        picker: RadioSet | SelectionList
        if question.get("multiSelect"):
            picker = SelectionList(
                *[
                    Selection(self._option_prompt(option), i)
                    for i, option in enumerate(options)
                ],
                classes="question-options",
            )
        else:
            picker = RadioSet(
                *[RadioButton(self._option_prompt(option)) for option in options],
                classes="question-options",
            )
        return Vertical(
            Static(str(question.get("header", "")), classes="question-header", markup=False),
            Static(str(question.get("question", "")), classes="question-text", markup=False),
            picker,
            Input(placeholder="Other…", classes="question-other"),
            id=f"question-{key}-{index}",
            classes="question-section",
        )
