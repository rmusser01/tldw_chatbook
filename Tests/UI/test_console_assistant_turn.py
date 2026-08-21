"""Focused tests for Console Assistant-turn presentation primitives."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Widgets.Console.console_assistant_turn import (
    ConsoleActivityActivated,
    ConsoleActivityDisclosure,
    ConsoleActivityHeader,
    ConsoleAssistantTurnWidget,
)


class ActivityHarness(App[None]):
    """Mount two independent disclosures and apply their emitted state."""

    def __init__(self, *disclosures: ConsoleActivityDisclosure) -> None:
        super().__init__()
        self.disclosures = disclosures
        self.activations: list[ConsoleActivityActivated] = []

    def compose(self) -> ComposeResult:
        yield from self.disclosures

    def on_console_activity_activated(self, event: ConsoleActivityActivated) -> None:
        self.activations.append(event)
        disclosure = self.query_one(
            f"#console-activity-disclosure-{event.message_id}",
            ConsoleActivityDisclosure,
        )
        disclosure.sync_state(
            expanded=(not disclosure.expanded)
            if event.toggle_requested
            else disclosure.expanded,
            selected=True,
        )


def _disclosure(
    activity_message_id: str,
    *,
    label: str = "Thinking",
    status: str = "done",
    selected: bool = False,
    actions: tuple[Static, ...] = (),
    details: tuple[Static, ...] = (),
) -> ConsoleActivityDisclosure:
    return ConsoleActivityDisclosure(
        activity_message_id,
        label,
        status,
        selected=selected,
        action_widgets=actions,
        detail_widgets=details,
    )


@pytest.mark.asyncio
async def test_disclosures_are_independently_collapsed_by_default() -> None:
    app = ActivityHarness(
        _disclosure("activity-one", details=(Static("one detail"),)),
        _disclosure("activity-two", details=(Static("two detail"),)),
    )

    async with app.run_test():
        first = app.query_one(
            "#console-activity-disclosure-activity-one", ConsoleActivityDisclosure
        )
        second = app.query_one(
            "#console-activity-disclosure-activity-two", ConsoleActivityDisclosure
        )

        assert not first.expanded
        assert not second.expanded
        assert not first.query_one(".console-activity-detail-stack").display
        assert not second.query_one(".console-activity-detail-stack").display


@pytest.mark.asyncio
async def test_expandable_header_click_enter_and_space_emit_original_id() -> None:
    disclosure = _disclosure(
        "activity-expandable",
        label="fs_[literal]",
        status="success",
        details=(Static("detail"),),
    )
    app = ActivityHarness(disclosure)

    async with app.run_test() as pilot:
        header = app.query_one(ConsoleActivityHeader)
        assert header.can_focus
        assert header.renderable.plain == "▸ fs_[literal] · success"

        await pilot.click("#console-activity-header-activity-expandable")
        await pilot.pause()
        assert [
            (event.message_id, event.toggle_requested) for event in app.activations
        ] == [("activity-expandable", True)]
        assert header.renderable.plain == "▾ fs_[literal] · success"

        header.focus()
        await pilot.press("enter")
        await pilot.press("space")
        await pilot.press("x")
        await pilot.pause()

    assert [
        (event.message_id, event.toggle_requested) for event in app.activations
    ] == [
        ("activity-expandable", True),
        ("activity-expandable", True),
        ("activity-expandable", True),
    ]


@pytest.mark.asyncio
async def test_no_detail_activity_is_focusable_and_selects_without_toggle() -> None:
    app = ActivityHarness(_disclosure("thinking-empty"))

    async with app.run_test() as pilot:
        header = app.query_one(ConsoleActivityHeader)
        assert header.can_focus
        assert header.renderable.plain == "Thinking · done"
        assert "▸" not in header.renderable.plain
        assert "▾" not in header.renderable.plain

        header.focus()
        await pilot.press("enter")
        await pilot.pause()

        disclosure = app.query_one(ConsoleActivityDisclosure)
        assert disclosure.selected
        assert not disclosure.expanded

    assert [
        (event.message_id, event.toggle_requested) for event in app.activations
    ] == [("thinking-empty", False)]


@pytest.mark.asyncio
async def test_selected_actions_precede_hidden_detail_while_collapsed() -> None:
    disclosure = _disclosure(
        "activity-selected",
        selected=True,
        actions=(Static("Copy", id="activity-action"),),
        details=(Static("private detail", id="activity-detail"),),
    )
    app = ActivityHarness(disclosure)

    async with app.run_test():
        children = list(disclosure.children)
        assert [child.id for child in children] == [
            "console-activity-header-activity-selected",
            "console-activity-actions-activity-selected",
            "console-activity-detail-activity-selected",
        ]
        assert children[1].display
        assert not children[2].display
        assert disclosure.query_one("#activity-action").display


@pytest.mark.asyncio
async def test_parent_applied_expansion_leaves_sibling_collapsed() -> None:
    app = ActivityHarness(
        _disclosure("first", details=(Static("first detail"),)),
        _disclosure("second", details=(Static("second detail"),)),
    )

    async with app.run_test() as pilot:
        await pilot.click("#console-activity-header-first")
        await pilot.pause()

        first = app.query_one(
            "#console-activity-disclosure-first", ConsoleActivityDisclosure
        )
        second = app.query_one(
            "#console-activity-disclosure-second", ConsoleActivityDisclosure
        )
        assert first.expanded
        assert first.query_one(".console-activity-detail-stack").display
        assert not second.expanded
        assert not second.query_one(".console-activity-detail-stack").display


class AssistantTurnHarness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.header = Static("Assistant", id="assistant-header")
        self.answer = Static("Final answer", id="assistant-answer")
        self.turn = ConsoleAssistantTurnWidget(
            "assistant-one",
            self.header,
            (Static("Thinking", id="activity-old"),),
            self.answer,
            (Static("Sources", id="assistant-adjunct"),),
        )

    def compose(self) -> ComposeResult:
        yield self.turn


@pytest.mark.asyncio
async def test_assistant_turn_dom_order_is_header_activity_answer_adjunct() -> None:
    app = AssistantTurnHarness()

    async with app.run_test():
        assert [child.id for child in app.turn.children] == [
            "assistant-header",
            "console-assistant-activities-assistant-one",
            "assistant-answer",
            "console-assistant-adjuncts-assistant-one",
        ]
        assert app.turn.assistant_message_id == "assistant-one"


@pytest.mark.asyncio
async def test_replacing_activities_preserves_turn_header_and_answer_identity() -> None:
    app = AssistantTurnHarness()

    async with app.run_test():
        turn = app.turn
        original_turn = turn
        original_header = app.header
        original_answer = app.answer
        replacement = Static("Tool", id="activity-old")

        await turn.replace_activity_widgets(
            (replacement, Static("Thinking again", id="activity-new"))
        )

        assert turn is original_turn
        assert turn.query_one("#assistant-header") is original_header
        assert turn.query_one("#assistant-answer") is original_answer
        assert turn.query_one("#activity-old") is replacement
        assert len(turn.query("#activity-old")) == 1
        assert len(turn.query("#activity-new")) == 1
        assert [child.id for child in turn.activity_stack.children] == [
            "activity-old",
            "activity-new",
        ]
