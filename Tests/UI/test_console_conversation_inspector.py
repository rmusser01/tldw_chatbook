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
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static, TabPane
from textual.widgets._collapsible import CollapsibleTitle

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import (
    ConsoleCostRow,
    ConsoleCostRowTotals,
    build_cost_rows,
)
from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
    InspectorTurn,
)


def _row(index: int = 0, model: str = "m") -> ConsoleCostRow:
    return ConsoleCostRow(
        index=index,
        role="assistant",
        model=model,
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


def _rendered_title(collapsible: Collapsible) -> str:
    """Plain text of a Collapsible's ACTUAL rendered title label.

    Review finding 1/2: ``Collapsible.title`` alone can't reveal markup
    mangling -- Textual parses a Collapsible's title as markup by default
    (``CollapsibleTitle.__init__`` -> ``Content.from_text(label)``, whose
    ``markup`` default is ``True``), so a raw ``"x" in collapsible.title``
    assertion would still pass even if the label had been silently eaten
    or had raised. Reading the mounted ``CollapsibleTitle`` widget's own
    ``.label.plain`` is what actually appears on screen.
    """
    return collapsible.query_one(CollapsibleTitle).label.plain


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
        # it here verbatim). Asserts on the RENDERED label, not the raw
        # ``.title`` attribute -- see ``_rendered_title``'s docstring.
        assert "in:10" in _rendered_title(collapsibles[0])

        totals = modal.query_one("#console-inspector-costs-totals", Static)
        assert "15 tokens" in str(totals.renderable)


@pytest.mark.asyncio
async def test_collapsible_title_is_not_markup_parsed() -> None:
    """Regression pin (review finding 1): a Collapsible's title IS
    markup-parsed by Textual unless built with ``Content.from_text(...,
    markup=False)``. Proven against installed Textual: a model id
    containing ``"[test]"`` was silently eaten to an empty label, and one
    containing ``"[/]"`` raised ``MarkupError`` inside ``compose()``,
    taking the whole modal down with it (``ConsoleCostModal`` avoided this
    by rendering the same string through ``Static(..., markup=False)``;
    the move to a ``Collapsible`` title dropped that guard).

    Two rows: one whose model contains ``"[test]"`` (must survive intact
    in the rendered label, not be eaten), one whose model contains
    ``"[/]"`` (the modal opening at all, with no exception, is the proof
    the row didn't raise)."""
    eatable_row = _row(index=0, model="model-[test]")
    raising_row = _row(index=1, model="model-[/]")

    app = InspectorHarness(
        **_default_kwargs(
            rows=[eatable_row, raising_row],
            turns=[
                _turn(index=0, message_id="p1", native_message_id="n1"),
                _turn(index=1, message_id="p2", native_message_id="n2"),
            ],
        )
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        rows_container = modal.query_one(
            "#console-inspector-costs-rows", VerticalScroll
        )
        collapsibles = {c.id: c for c in rows_container.query(Collapsible)}
        assert set(collapsibles) == {
            "console-inspector-cost-row-0",
            "console-inspector-cost-row-1",
        }

        eaten_title = _rendered_title(collapsibles["console-inspector-cost-row-0"])
        assert "model-[test]" in eaten_title

        raising_title = _rendered_title(collapsibles["console-inspector-cost-row-1"])
        assert "model-[/]" in raising_title


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


@pytest.mark.asyncio
async def test_loader_failure_shows_a_distinct_message_and_allows_retry() -> None:
    """Review finding 3: an ``exchanges_loader`` exception (e.g.
    ``get_message_exchanges`` raising ``CharactersRAGDBError``) must NOT be
    folded into the "no captures" empty state -- that would permanently
    misreport a transient failure as "this turn was never captured", with
    no way to retry short of reopening the whole modal. It renders a
    DISTINCT message, and the row is un-marked as loaded so a
    collapse/re-expand tries again."""
    calls = 0

    async def flaky_loader(
        _native_message_id: str,
    ) -> list[tuple[ExchangeCapture, bool]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return []

    app = InspectorHarness(**_default_kwargs(exchanges_loader=flaky_loader))

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        collapsible = modal.query_one("#console-inspector-cost-row-0", Collapsible)
        collapsible.collapsed = False

        def _shows_failure_message() -> bool:
            return any(
                "Could not load captures for this turn" in str(static.renderable)
                for static in collapsible.query(Static)
            )

        await _wait_until(pilot, _shows_failure_message)
        assert calls == 1

        # Distinct wording from the genuine "no captures" empty state --
        # a caller must be able to tell "failed, retry" apart from
        # "there really is nothing here".
        texts = [str(s.renderable) for s in collapsible.query(Static)]
        assert not any("No capture recorded for this turn" in t for t in texts)

        # Collapse/re-expand retries -- the failed row was NOT permanently
        # marked as loaded (contrast with test_loader_called_lazily_only_
        # on_expand's success case, which must NOT retry).
        collapsible.collapsed = True
        await pilot.pause()
        collapsible.collapsed = False

        await _wait_until(pilot, lambda: calls == 2)


def _capture(
    run_tag: str,
    seq: int,
    created_at: str,
    model: str,
    usage_json: str | None = None,
) -> ExchangeCapture:
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
        usage_json=usage_json,
        omitted_keys=(),
    )


def test_call_cost_line_prices_through_the_same_path_as_build_cost_rows() -> None:
    """Review finding 6 (closing item): pins the "same pricing path as
    ``build_cost_rows``" guarantee with an actual test rather than just
    code reading. Builds a real, catalog-priced ``ProviderUsage`` for a
    known model, prices an equivalent row through ``build_cost_rows``, and
    asserts ``ConsoleConversationInspector._call_cost_line`` on a capture
    carrying that SAME usage (serialized to JSON, as a real capture would
    store it) reproduces the identical dollar figure -- not a hardcoded
    price, so this stays correct even if the catalog's rates change."""
    usage = ProviderUsage(
        uncached_input=1000,
        output=500,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    row_message = SimpleNamespace(content="hi", usage=usage, role="assistant")
    [priced_row] = build_cost_rows(
        [row_message], provider="anthropic", model="claude-sonnet-4-6"
    )
    assert priced_row.cost_usd is not None  # sanity: this model IS priced

    capture = _capture(
        "run-1", 1, "2026-08-20T10:00:00Z", "claude-sonnet-4-6", usage.to_json()
    )

    line = ConsoleConversationInspector._call_cost_line(capture)

    assert line == f"${priced_row.cost_usd:.4f}"
    assert line != "unpriced"


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
