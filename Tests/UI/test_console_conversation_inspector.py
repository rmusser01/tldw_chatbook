"""Conversation Inspector modal scaffold (task-8): three tabs, Costs rows
render, per-turn drill-in lazy-loads captures.

Mirrors ``Tests/UI/test_console_context_modal.py``'s harness idiom -- a bare
``App`` that pushes the modal directly on mount, driven with ``run_test``/
``pilot`` -- rather than the full ``ConsoleHarness`` app (this widget never
touches the Console screen/store itself; every input is precomputed and
handed in at construction, same shape as ``ConsoleCostModal``/
``ConsoleContextModal``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static, TabPane

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRow, ConsoleCostRowTotals
from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
    InspectorTurn,
)


def _row(index: int = 0) -> ConsoleCostRow:
    return ConsoleCostRow(
        index=index,
        role="assistant",
        model="m",
        uncached_input=10,
        cache_read=0,
        cache_write=0,
        output=5,
        cost_usd=0.001,
        estimated=False,
    )


def _totals() -> ConsoleCostRowTotals:
    return ConsoleCostRowTotals(
        total_tokens=15,
        total_cost_usd=0.001,
        has_estimated_entries=False,
        row_count=1,
    )


def _turn(
    index: int = 0, message_id: str = "p1", native_message_id: str = "n1"
) -> InspectorTurn:
    return InspectorTurn(
        message_id=message_id,
        native_message_id=native_message_id,
        index=index,
        role="assistant",
        preview="hi",
    )


async def _noop_snapshot() -> ConsoleContextSnapshot:
    return ConsoleContextSnapshot(current_messages=[], next_send_payload={})


async def _empty_exchanges_loader(
    _native_message_id: str,
) -> list[tuple[ExchangeCapture, bool]]:
    return []


def _default_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = dict(
        rows=[_row()],
        totals=_totals(),
        turns=[_turn()],
        exchanges_loader=_empty_exchanges_loader,
        snapshot_factory=_noop_snapshot,
    )
    kwargs.update(overrides)
    return kwargs


class InspectorHarness(App):
    def __init__(self, **modal_kwargs: object) -> None:
        super().__init__()
        self._modal_kwargs = modal_kwargs

    def compose(self) -> ComposeResult:
        yield Static("background")

    def on_mount(self) -> None:
        self.push_screen(ConsoleConversationInspector(**self._modal_kwargs))


async def _wait_until(
    pilot: object, predicate: Callable[[], bool], attempts: int = 30
) -> None:
    """Poll ``predicate`` across event-loop ticks instead of a fixed sleep --
    the lazy drill-in runs on a real Textual worker, so the number of pauses
    needed to observe its effect is not guaranteed constant."""
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()  # type: ignore[attr-defined]
    assert predicate()


@pytest.mark.asyncio
async def test_three_tabs_render() -> None:
    app = InspectorHarness(**_default_kwargs())

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        tab_pane_ids = {pane.id for pane in modal.query(TabPane)}
        assert tab_pane_ids == {
            "inspector-costs",
            "inspector-exchange",
            "inspector-next-send",
        }


@pytest.mark.asyncio
async def test_costs_rows_render_and_totals() -> None:
    app = InspectorHarness(**_default_kwargs())

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        rows_container = modal.query_one(
            "#console-inspector-costs-rows", VerticalScroll
        )
        collapsibles = list(rows_container.query(Collapsible))
        assert len(collapsibles) == 1
        # Reuses ConsoleCostModal._format_row's exact format (Step 3 moves
        # it here verbatim).
        assert "in:10" in collapsibles[0].title

        totals = modal.query_one("#console-inspector-costs-totals", Static)
        assert "15 tokens" in str(totals.renderable)


@pytest.mark.asyncio
async def test_loader_called_lazily_only_on_expand() -> None:
    calls: list[str] = []

    async def spy_loader(
        native_message_id: str,
    ) -> list[tuple[ExchangeCapture, bool]]:
        calls.append(native_message_id)
        return []

    app = InspectorHarness(**_default_kwargs(exchanges_loader=spy_loader))

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        assert calls == []

        modal = app.screen
        collapsible = modal.query_one("#console-inspector-cost-row-0", Collapsible)
        collapsible.collapsed = False
        await _wait_until(pilot, lambda: calls != [])

        assert calls == ["n1"]

        # Collapsing and re-expanding must not fetch a second time -- the
        # brief's "if expanding and not yet loaded" contract.
        collapsible.collapsed = True
        await pilot.pause()
        collapsible.collapsed = False
        await pilot.pause()
        await pilot.pause()

        assert calls == ["n1"]


@pytest.mark.asyncio
async def test_no_capture_recorded_row() -> None:
    app = InspectorHarness(**_default_kwargs())

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        collapsible = modal.query_one("#console-inspector-cost-row-0", Collapsible)
        collapsible.collapsed = False

        def _has_body_text() -> bool:
            return any(
                "No capture recorded for this turn" in str(static.renderable)
                for static in collapsible.query(Static)
            )

        await _wait_until(pilot, _has_body_text)


def _capture(run_tag: str, seq: int, created_at: str, model: str) -> ExchangeCapture:
    return ExchangeCapture(
        run_tag=run_tag,
        seq=seq,
        created_at=created_at,
        provider="anthropic",
        model=model,
        endpoint=None,
        request={},
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=(),
    )


@pytest.mark.asyncio
async def test_multi_run_captures_ordered_by_created_at_not_run_tag() -> None:
    """Carried item (task-8 brief): the store/DB order captures by
    ``(run_tag, seq)`` STRING, which is not chronological across multiple
    runs on one message -- ``run-a`` sorts before ``run-b`` alphabetically
    even though ``run-b``'s call happened first. The widget must re-sort
    by ``(created_at, seq)`` rather than trust the loader's own order."""
    early = _capture("run-b", 1, "2026-08-17T10:00:00Z", "model-early")
    late = _capture("run-a", 1, "2026-08-17T11:00:00Z", "model-late")

    async def loader(
        _native_message_id: str,
    ) -> list[tuple[ExchangeCapture, bool]]:
        # Deliberately handed back in run_tag order (late-chronologically
        # first) to prove the widget re-sorts rather than trusting it.
        return [(late, False), (early, False)]

    app = InspectorHarness(**_default_kwargs(exchanges_loader=loader))

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        collapsible = modal.query_one("#console-inspector-cost-row-0", Collapsible)
        collapsible.collapsed = False

        def _both_rendered() -> bool:
            texts = [str(s.renderable) for s in collapsible.query(Static)]
            return any("model-early" in t for t in texts) and any(
                "model-late" in t for t in texts
            )

        await _wait_until(pilot, _both_rendered)

        call_texts = [
            str(s.renderable)
            for s in collapsible.query(Static)
            if str(s.renderable).startswith("call ")
        ]
        assert len(call_texts) == 2
        early_index = next(i for i, t in enumerate(call_texts) if "model-early" in t)
        late_index = next(i for i, t in enumerate(call_texts) if "model-late" in t)
        assert early_index < late_index, (
            f"expected created_at order (early before late), got {call_texts!r}"
        )


@pytest.mark.asyncio
async def test_costs_pane_empty_state() -> None:
    app = InspectorHarness(**_default_kwargs(rows=[], turns=[]))

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        rows_container = modal.query_one(
            "#console-inspector-costs-rows", VerticalScroll
        )
        assert not list(rows_container.query(Collapsible))
        labels = [str(s.renderable) for s in rows_container.query(Static)]
        assert any("No priced or estimated messages" in text for text in labels)


@pytest.mark.asyncio
async def test_initial_tab_selects_the_requested_pane() -> None:
    app = InspectorHarness(**_default_kwargs(initial_tab="inspector-next-send"))

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        from textual.widgets import TabbedContent

        tabs = modal.query_one("#console-inspector-tabs", TabbedContent)
        assert tabs.active == "inspector-next-send"


@pytest.mark.asyncio
async def test_escape_dismisses_with_none() -> None:
    app = InspectorHarness(**_default_kwargs())

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConsoleConversationInspector)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ConsoleConversationInspector)
