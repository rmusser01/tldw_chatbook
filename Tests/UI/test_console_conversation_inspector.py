"""Conversation Inspector modal scaffold (task-8): three tabs, Costs rows
render, per-turn drill-in lazy-loads captures.

Mirrors ``Tests/UI/test_console_context_modal.py``'s harness idiom -- a bare
``App`` that pushes the modal directly on mount, driven with ``run_test``/
``pilot`` -- rather than the full ``ConsoleHarness`` app (this widget never
touches the Console screen/store itself; every input is precomputed and
handed in at construction, same shape as the two standalone modals this
one replaced -- both retired outright in task-10).
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    Static,
    TabPane,
    TextArea,
)
from textual.widgets._collapsible import CollapsibleTitle
from textual.worker import WorkerState

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import (
    ConsoleCostRow,
    ConsoleCostRowTotals,
    build_cost_rows,
)
from tldw_chatbook.Chat.console_exchange_capture import (
    CAPTURE_REQUEST_ALLOWLIST,
    ExchangeCapture,
    build_request_capture,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    CLOSE_BUTTON_ID,
    TAB_COSTS,
    TAB_EXCHANGE,
    TAB_NEXT_SEND,
    ConsoleConversationInspector,
    InspectorTurn,
    _SAMPLING_EXCLUDED_REQUEST_KEYS,
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
    index: int = 0,
    message_id: str = "p1",
    native_message_id: str = "n1",
    role: str = "assistant",
) -> InspectorTurn:
    return InspectorTurn(
        message_id=message_id,
        native_message_id=native_message_id,
        index=index,
        role=role,
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
        # Scoped to the OUTER TabbedContent's own direct panes: task-10's
        # Next Send tab nests its own inner TabbedContent (Current / Next
        # Send sub-tabs, ported from the retired standalone context modal),
        # so an unscoped ``modal.query(TabPane)`` now also picks up those
        # two nested TabPanes -- not what this test is pinning. Reading the
        # switcher's direct ``.children`` (not a recursive query) keeps
        # this scoped to one level.
        outer_tabs = modal.query_one("#console-inspector-tabs")
        switcher = outer_tabs.query_one(ContentSwitcher)
        tab_pane_ids = {pane.id for pane in switcher.children}
        assert tab_pane_ids == {
            "inspector-costs",
            "inspector-exchange",
            "inspector-next-send",
        }


@pytest.mark.asyncio
async def test_exchange_tab_states_adapter_boundary_caveat() -> None:
    """Review finding I2: the spec requires the adapter-boundary caveat
    STATED IN THE UI (twice), not just the User Guide -- a user on the
    Exchange tab has no other in-surface signal that capture happens at the
    provider-adapter boundary (not the raw HTTP layer) and that llama.cpp
    is the one exception."""
    app = InspectorHarness(**_default_kwargs(initial_tab=TAB_EXCHANGE))

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        caveat = modal.query_one("#console-inspector-exchange-caveat", Static)
        text = str(caveat.renderable)
        assert "adapter" in text.lower()
        assert "llama.cpp" in text


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
        # Reuses the retired standalone cost modal's own
        # ``_format_row``'s exact format (Step 3 moved it here verbatim).
        # Asserts on the RENDERED label, not the raw
        # ``.title`` attribute -- see ``_rendered_title``'s docstring.
        assert "in:10" in _rendered_title(collapsibles[0])

        totals = modal.query_one("#console-inspector-costs-totals", Static)
        assert "15 tokens" in str(totals.renderable)


@pytest.mark.asyncio
async def test_exchange_tab_only_shows_turns_with_a_cost_row() -> None:
    """Review finding M5, and its own regression closed by the final
    re-review (task-18300). ``chat_screen.py`` builds one ``InspectorTurn``
    per transcript MESSAGE, but ``build_cost_rows`` skips non-contributing
    ones (no usage, no non-blank content) -- e.g. a bare tool/user message
    at index 1 here, with no matching cost row. That turn is still
    filtered out (de-clutter, M5's original goal).

    But M5's original ``and``-only predicate over-corrected: it also
    dropped an ASSISTANT turn with no cost row, which is exactly the
    "Stop pressed before the first token" shape (blank content, no usage
    -- ``build_cost_rows`` emits nothing for it, but the message is still
    marked and persisted and its "stopped" capture is still flushed). That
    turn (index 3) has no cost row either, but MUST still render an
    Exchange-tab row -- and expanding it must reach its capture, not "No
    capture recorded for this turn" -- because dropping it would make
    "what did I send that hung?" unreachable, defeating a core promise of
    this tab. This assertion is red-proof: reverting the fix to the
    `and`-only predicate makes it fail (turn 3 goes missing)."""
    stopped_capture = _capture(
        "run-stop", 0, "2026-08-20T10:00:00Z", "m", status="stopped"
    )

    async def loader(native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        if native_message_id == "n4":
            return [(stopped_capture, False)]
        return []

    app = InspectorHarness(
        **_default_kwargs(
            rows=[_row(index=0), _row(index=2)],
            turns=[
                _turn(index=0, message_id="p1", native_message_id="n1"),
                _turn(index=1, message_id="p2", native_message_id="n2", role="user"),
                _turn(index=2, message_id="p3", native_message_id="n3"),
                _turn(
                    index=3,
                    message_id="p4",
                    native_message_id="n4",
                    role="assistant",
                ),
            ],
            exchanges_loader=loader,
            initial_tab=TAB_EXCHANGE,
        )
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        turns_container = modal.query_one(
            "#console-inspector-exchange-turns", VerticalScroll
        )
        collapsibles = {c.id: c for c in turns_container.query(Collapsible)}
        assert set(collapsibles) == {
            "console-inspector-exchange-turn-0",
            "console-inspector-exchange-turn-2",
            "console-inspector-exchange-turn-3",
        }

        # Load-bearing half: turn 3 (no cost row, assistant role, the
        # stop-before-first-token shape) must actually resolve to its
        # capture on expand, not fall back to the "no captures" empty
        # state -- rendering the row alone would not prove the loader
        # path still works for a turn outside `contributing_indices`.
        turn_3 = collapsibles["console-inspector-exchange-turn-3"]
        turn_3.collapsed = False

        def _turn_3_loaded() -> bool:
            return bool(turn_3.query(Collapsible)) or any(
                "No capture recorded for this turn" in str(static.renderable)
                for static in turn_3.query(Static)
            )

        await _wait_until(pilot, _turn_3_loaded)

        call_collapsibles = {c.id for c in turn_3.query(Collapsible)}
        assert call_collapsibles == {"console-inspector-exchange-call-3-0"}
        texts = [str(static.renderable) for static in turn_3.query(Static)]
        assert not any("No capture recorded for this turn" in t for t in texts)


@pytest.mark.asyncio
async def test_collapsible_title_is_not_markup_parsed() -> None:
    """Regression pin (review finding 1): a Collapsible's title IS
    markup-parsed by Textual unless built with ``Content.from_text(...,
    markup=False)``. Proven against installed Textual: a model id
    containing ``"[test]"`` was silently eaten to an empty label, and one
    containing ``"[/]"`` raised ``MarkupError`` inside ``compose()``,
    taking the whole modal down with it (the retired standalone cost modal
    avoided this by rendering the same string through
    ``Static(..., markup=False)``; the move to a ``Collapsible`` title
    dropped that guard).

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
    *,
    status: str = "complete",
    request: dict | None = None,
    response: dict | None = None,
    omitted_keys: tuple[str, ...] = (),
) -> ExchangeCapture:
    return ExchangeCapture(
        run_tag=run_tag,
        seq=seq,
        created_at=created_at,
        provider="anthropic",
        model=model,
        endpoint=None,
        request={} if request is None else request,
        response={} if response is None else response,
        status=status,
        usage_json=usage_json,
        omitted_keys=omitted_keys,
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


# --- Merged from test_console_cost_modal.py (task-10 retired that ---------
# --- standalone modal; ``_format_row`` lives here now, task-8's port) -----


def test_format_row_shows_audio_and_transcription_when_present() -> None:
    """task-2390: ``_format_row`` surfaces realtime audio-token and
    transcription-duration costs -- ``ConsoleCostRow`` already folds them
    into ``cost_usd`` (a single dollar figure), and this pin requires the
    breakdown not silently hide them inside that undecomposable total."""
    row = ConsoleCostRow(
        index=0,
        role="assistant",
        model="gpt-realtime",
        uncached_input=15,
        cache_read=0,
        cache_write=0,
        output=28,
        cost_usd=0.006844,
        estimated=False,
        audio_input=18,
        audio_output=90,
        transcription_seconds=2.5,
    )
    text = ConsoleConversationInspector._format_row(row)
    assert "audio_in:18" in text
    assert "audio_out:90" in text
    assert "transcribe:2.5s" in text


def test_format_row_omits_audio_fields_for_a_non_realtime_row() -> None:
    row = ConsoleCostRow(
        index=0,
        role="user",
        model="claude-sonnet-4-6",
        uncached_input=100,
        cache_read=0,
        cache_write=0,
        output=0,
        cost_usd=0.10,
        estimated=False,
    )
    text = ConsoleConversationInspector._format_row(row)
    assert "audio_in" not in text
    assert "audio_out" not in text
    assert "transcribe" not in text


# --- Exchange tab (task-9) -------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_call_sections_render() -> None:
    """One capture with every optional section populated -- expanding the
    turn, the call, and each individual section must surface every value
    the brief calls out: the system prompt text, a tool's name, the
    response text, and a sampling kwarg -- plus the "omitted by capture
    policy" line for a dropped credential key, which is NOT behind a lazy
    section (it renders as soon as the call itself expands)."""
    cap = _capture(
        "r1",
        0,
        "t",
        "m",
        request={
            "system_message": "SYS PROMPT",
            "messages_payload": [{"role": "user", "content": "hello"}],
            "tools": [{"function": {"name": "get_time"}}],
            "temp": 0.7,
        },
        response={"content": "world", "tool_calls": []},
        omitted_keys=("api_key",),
    )

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        # Multi-widget synchronous mount (omitted line, actions row, 5
        # section headers) can take more than one pump tick to fully
        # realize under load -- poll rather than trust a single pause.
        await _wait_until(pilot, lambda: len(call.query(Collapsible)) >= 5)

        # "Omitted by capture policy" mounts immediately with the call --
        # it is not gated behind any further expansion.
        call_statics = [str(s.renderable) for s in call.query(Static)]
        assert any("Omitted by capture policy: api_key" in t for t in call_statics)

        section_ids = {
            "system": "console-inspector-exchange-section-0-0-system",
            "messages": "console-inspector-exchange-section-0-0-messages",
            "tools": "console-inspector-exchange-section-0-0-tools",
            "response": "console-inspector-exchange-section-0-0-response",
            "sampling": "console-inspector-exchange-section-0-0-sampling",
        }
        titles = {
            name: _rendered_title(call.query_one(f"#{section_id}", Collapsible))
            for name, section_id in section_ids.items()
        }
        assert "System prompt" in titles["system"]
        assert "Tools" in titles["tools"]
        assert "Response" in titles["response"]
        assert "Sampling" in titles["sampling"]

        for section_id in section_ids.values():
            call.query_one(f"#{section_id}", Collapsible).collapsed = False
        # system/tools/response/sampling each mount one TextArea; messages
        # mounts per-message Collapsibles instead (not counted here).
        await _wait_until(pilot, lambda: len(call.query(TextArea)) >= 4)

        all_text = "\n".join(ta.text for ta in call.query(TextArea))
        assert "SYS PROMPT" in all_text
        assert "get_time" in all_text
        assert "world" in all_text
        assert "temp" in all_text

        # "Tool calls" is omitted entirely -- this capture's response
        # carries an empty tool_calls list.
        assert not call.query("#console-inspector-exchange-section-0-0-toolcalls")


@pytest.mark.asyncio
async def test_estimates_labeled_and_reported_authoritative() -> None:
    """A capture WITH ``usage_json`` -- the estimated per-piece figure (the
    Response section's own title) must carry the "~"/"est." labels, while
    the reported figure (both the call-level summary line and the
    unprefixed half of the Response title) must NOT -- it is the
    authoritative, provider-reported number."""
    usage = ProviderUsage(
        uncached_input=100, cache_read=0, cache_write=0, output=5,
        provider="anthropic", model="m",
    )
    cap = _capture(
        "r1",
        0,
        "t",
        "m",
        usage.to_json(),
        request={"system_message": "hi", "messages_payload": [], "tools": []},
        response={"content": "hello world", "tool_calls": []},
    )

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        await _wait_until(
            pilot,
            lambda: bool(call.query(Collapsible))
            and any(
                str(s.renderable).startswith("Reported usage")
                for s in call.query(Static)
            ),
        )

        reported_lines = [
            str(s.renderable)
            for s in call.query(Static)
            if str(s.renderable).startswith("Reported usage")
        ]
        assert len(reported_lines) == 1
        reported_line = reported_lines[0]
        assert "~" not in reported_line
        assert "est." not in reported_line
        assert "out:5" in reported_line

        response_title = _rendered_title(
            call.query_one("#console-inspector-exchange-section-0-0-response", Collapsible)
        )
        assert "~" in response_title
        assert "tokens est." in response_title
        assert "reported out:5" in response_title


@pytest.mark.asyncio
async def test_synthetic_fallback_response_is_labeled_not_shown_as_model_output() -> (
    None
):
    """Review finding M3: a capture whose response is locally synthesized
    fallback UI copy (``synthetic_fallback: True``, stamped by the gateway
    when the provider returned nothing usable) must not be presented in the
    Response section as if it were the model's own answer -- the section
    title switches to an explicit label instead of the normal token
    estimate."""
    cap = _capture(
        "r1",
        0,
        "t",
        "m",
        request={"system_message": "", "messages_payload": [], "tools": []},
        response={
            "content": "The provider returned no content.",
            "tool_calls": [],
            "synthetic_fallback": True,
        },
    )

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(Collapsible)))

        response_title = _rendered_title(
            call.query_one("#console-inspector-exchange-section-0-0-response", Collapsible)
        )
        assert "synthesized fallback" in response_title
        assert "~" not in response_title
        assert "tokens est." not in response_title


@pytest.mark.asyncio
async def test_status_badges() -> None:
    """"stopped"/"error" statuses and an ``abandoned=True`` pair each render
    their own distinct badge in the call's title -- all three coexisting on
    one turn's three calls."""
    stopped_cap = _capture("r1", 0, "t0", "m", status="stopped")
    error_cap = _capture("r1", 1, "t1", "m", status="error")
    abandoned_cap = _capture("r1", 2, "t2", "m", status="complete")

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(stopped_cap, False), (error_cap, False), (abandoned_cap, True)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: len(turn.query(Collapsible)) == 3)

        titles = [_rendered_title(c) for c in turn.query(Collapsible)]
        assert any("[stopped]" in t for t in titles)
        assert any("[error]" in t for t in titles)
        assert any("[abandoned regeneration]" in t for t in titles)
        # The abandoned regeneration must NOT also read "stopped"/"error" --
        # it completed normally, just against a superseded generation.
        abandoned_title = next(t for t in titles if "[abandoned regeneration]" in t)
        assert "[complete]" in abandoned_title


@pytest.mark.asyncio
async def test_collapsible_bodies_mount_lazily() -> None:
    """Expanding the call mounts the section headers (cheap Collapsibles)
    but NOT their TextArea bodies -- a section's TextArea only exists once
    THAT section is itself expanded. Proves the three-level lazy chain
    (turn -> call -> section) never front-loads content."""
    cap = _capture(
        "r1",
        0,
        "t",
        "m",
        request={"system_message": "hi", "messages_payload": [], "tools": []},
        response={"content": "hello", "tool_calls": []},
    )

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(Collapsible)))

        # Section headers exist (still collapsed)...
        section = call.query_one(
            "#console-inspector-exchange-section-0-0-system", Collapsible
        )
        assert section.collapsed is True
        # ...but nothing has mounted a TextArea yet, anywhere under this call.
        assert not call.query(TextArea)

        section.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(TextArea)))

        assert len(call.query(TextArea)) == 1


# --- Exchange tab review fixes (task-9 review round) -----------------------


# Finding 2's hardcoded half of the pin -- see the test below for why this
# must NOT be computed from the live import.
_TODAY_CAPTURE_ALLOWLIST_SNAPSHOT = frozenset({
    "api_endpoint", "api_base_url", "system_message", "messages_payload",
    "tools", "model", "streaming", "temp", "topp", "maxp", "topk", "minp",
    "max_tokens", "seed", "presence_penalty", "frequency_penalty",
    "reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort",
    "thinking_budget_tokens", "prompt_caching", "response_format",
    "api_mode", "request_timeout", "request_retries", "request_retry_delay",
    "provider_continuations",
})


@pytest.mark.asyncio
async def test_sampling_section_key_set_is_pinned_to_the_capture_allowlist() -> None:
    """Finding 2 (task-9 review): the Sampling section is built by
    EXCLUDING ``_SAMPLING_EXCLUDED_REQUEST_KEYS`` from ``capture.request``
    rather than allowlisting -- accepted as safe TODAY only because
    ``build_request_capture`` (console_exchange_capture.py) already
    allowlists everything that can land in ``request`` to
    ``CAPTURE_REQUEST_ALLOWLIST``. Nothing enforced that relationship
    before this test: a future key added to that allowlist would silently
    start rendering under "Sampling & routing" with zero changes to this
    file.

    Two-part pin:
      (a) A HARDCODED snapshot of today's allowlist (not derived from the
          live import -- that would make part (b) tautological and never
          fail) must still equal the live ``CAPTURE_REQUEST_ALLOWLIST``.
          If it drifts, THIS assertion fails first, forcing whoever
          touched the allowlist to look at this test and consciously
          decide whether the new/removed key belongs under Sampling.
      (b) Given a capture whose request carries every allowlisted key
          (from the live import, so this half tracks real behavior), the
          Sampling section's rendered key set is exactly the allowlist
          minus the four keys this widget already surfaces elsewhere
          (system prompt, messages, tools, model).
    """
    assert _TODAY_CAPTURE_ALLOWLIST_SNAPSHOT == CAPTURE_REQUEST_ALLOWLIST, (
        "CAPTURE_REQUEST_ALLOWLIST changed -- decide whether the new/removed "
        "key(s) belong under the Exchange tab's 'Sampling & routing' section "
        "(console_conversation_inspector.py's _SAMPLING_EXCLUDED_REQUEST_KEYS), "
        "then update _TODAY_CAPTURE_ALLOWLIST_SNAPSHOT above."
    )

    request = {key: f"value-for-{key}" for key in CAPTURE_REQUEST_ALLOWLIST}
    cap = _capture(
        "r1", 0, "t", "m", request=request,
        response={"content": "x", "tool_calls": []},
    )

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(Collapsible)))

        sampling_section = call.query_one(
            "#console-inspector-exchange-section-0-0-sampling", Collapsible
        )
        sampling_section.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(TextArea)))

        [text_area] = call.query(TextArea)
        rendered_keys = set(json.loads(text_area.text).keys())

        assert rendered_keys == (
            CAPTURE_REQUEST_ALLOWLIST - _SAMPLING_EXCLUDED_REQUEST_KEYS
        )


@pytest.mark.asyncio
async def test_save_button_disabled_and_tooltip_when_ephemeral() -> None:
    """Finding 3 (task-9 review): pins the per-call Save-to-File gate.
    This repo has a documented history of gates discovered inert -- an
    ephemeral inspector's Save button must be disabled AND carry the
    blocked-reason tooltip; Copy JSON (never writes to disk) stays
    enabled."""
    cap = _capture("r1", 0, "t", "m")

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(
            exchanges_loader=loader, initial_tab=TAB_EXCHANGE, ephemeral=True
        )
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(Button)))

        save_button = call.query_one("#console-inspector-exchange-save-0-0", Button)
        assert save_button.disabled is True
        assert save_button.tooltip is not None
        assert "temporary chat" in str(save_button.tooltip)

        copy_button = call.query_one("#console-inspector-exchange-copy-0-0", Button)
        assert copy_button.disabled is False


@pytest.mark.asyncio
async def test_save_exchange_capture_direct_call_still_blocked_when_ephemeral(
    monkeypatch,
) -> None:
    """Review finding M7: ``_save_exchange_capture`` used to write
    unconditionally -- the Save button's own ``disabled=`` state
    (asserted by the test above) was the ONLY enforcement of the ephemeral
    save-block. A direct call bypassing the button (e.g. a future caller)
    must still be blocked. Patches this module's own ``Path`` name to
    raise if ``Path.home()`` is ever reached -- proving the method returns
    before touching the filesystem at all."""
    cap = _capture("r1", 0, "t", "m")

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(
            exchanges_loader=loader, initial_tab=TAB_EXCHANGE, ephemeral=True
        )
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(Button)))

        import tldw_chatbook.Widgets.Console.console_conversation_inspector as inspector_module

        class _BoomPath:
            @staticmethod
            def home():
                raise AssertionError(
                    "Path.home() must not be reached when save is blocked"
                )

        monkeypatch.setattr(inspector_module, "Path", _BoomPath)

        modal._save_exchange_capture("0-0")  # must not raise, must not write


@pytest.mark.asyncio
async def test_save_exchange_capture_rejected_destination_does_not_write(
    monkeypatch,
) -> None:
    """Qodo PR #1883 finding 2: ``_save_exchange_capture`` (like its Next
    Send-tab sibling ``_save_json``) now runs its Downloads-bound
    destination through ``path_validation.validate_path`` before
    ``mkdir``/``write_text``. When that check rejects the destination,
    the capture must not be written -- ``mkdir``/``write_text`` below
    raise ``AssertionError`` if ever reached, proving the short-circuit
    -- and the toast must surface only the failure class + path, never
    the raw ``path_validation`` exception body."""
    cap = _capture("r1", 0, "t", "m")

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(Button)))

        import tldw_chatbook.Widgets.Console.console_conversation_inspector as inspector_module

        class _RejectedSaveGuardPath:
            """``home()``/``__truediv__`` succeed (so path construction
            reaches the validation call); ``mkdir``/``write_text`` must
            never be reached once ``validate_path`` (patched below)
            rejects."""

            def __str__(self) -> str:
                return "/guard/Downloads/rejected.json"

            @classmethod
            def home(cls) -> "_RejectedSaveGuardPath":
                return cls()

            def __truediv__(self, other: str) -> "_RejectedSaveGuardPath":
                return self

            def mkdir(self, **kwargs: object) -> None:
                raise AssertionError(
                    "must not mkdir when destination is rejected"
                )

            def write_text(self, *args: object, **kwargs: object) -> None:
                raise AssertionError(
                    "must not write when destination is rejected"
                )

        monkeypatch.setattr(inspector_module, "Path", _RejectedSaveGuardPath)

        sentinel = "SENTINEL-VALIDATE-boom-must-not-leak-71ab"

        def _reject(*_args: object, **_kwargs: object) -> None:
            raise ValueError(sentinel)

        monkeypatch.setattr(inspector_module, "validate_path", _reject)

        notifications: list[tuple[str, str | None]] = []
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, *a, **k: notifications.append(
                (str(message), k.get("severity"))
            ),
        )

        modal._save_exchange_capture("0-0")  # must not raise, must not write

        assert notifications, "expected a rejection toast"
        message, severity = notifications[-1]
        assert severity == "error"
        assert sentinel not in message
        assert "ValueError" in message
        assert "/guard/Downloads/rejected.json" in message


# ---------------------------------------------------------------------------
# C1 (CRITICAL): the Exchange tab's per-call Copy JSON / Save to File must
# never carry an automatically-injected project-instruction body. The fix
# lives one layer down, in ``build_request_capture`` (``console_exchange_
# capture.py``) -- these two tests exercise the REAL function to build the
# capture's ``request``, then drive the REAL ``_copy_exchange_capture``/
# ``_save_exchange_capture`` methods end-to-end, proving the redaction
# survives being loaded into the Inspector and serialized back out, not
# just that the standalone unit test passes in isolation.
# ---------------------------------------------------------------------------

_EXCHANGE_EXPORT_SENTINEL = (
    "SENTINEL-EXCHANGE-EXPORT: automatic project instruction body must "
    "never reach the Exchange tab's Copy JSON or Save to File output."
)


def _capture_with_project_instruction_row() -> ExchangeCapture:
    request, omitted = build_request_capture(
        {
            "model": "gpt-4",
            "messages_payload": [
                {"role": "user", "content": "ordinary message"},
                {
                    "role": "user",
                    "content": _EXCHANGE_EXPORT_SENTINEL,
                    EPHEMERAL_ORIGIN_KEY: "project_instructions",
                },
            ],
        }
    )
    return _capture("r1", 0, "t", "gpt-4", request=request, omitted_keys=omitted)


async def _expand_first_exchange_call(pilot, modal) -> None:
    turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
    turn.collapsed = False
    await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

    call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
    call.collapsed = False
    await _wait_until(pilot, lambda: bool(call.query(Button)))


@pytest.mark.asyncio
async def test_exchange_copy_json_omits_project_instruction_body(monkeypatch) -> None:
    cap = _capture_with_project_instruction_row()

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    fake_copy = SimpleNamespace(copy=Mock())
    monkeypatch.setitem(sys.modules, "pyperclip", fake_copy)

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        await _expand_first_exchange_call(pilot, modal)

        await pilot.click("#console-inspector-exchange-copy-0-0")
        await pilot.pause()

    fake_copy.copy.assert_called_once()
    exported = fake_copy.copy.call_args.args[0]
    assert _EXCHANGE_EXPORT_SENTINEL not in exported
    assert "ordinary message" in exported
    assert "omitted by capture policy" in exported


@pytest.mark.asyncio
async def test_exchange_save_to_file_omits_project_instruction_body(
    tmp_path, monkeypatch
) -> None:
    cap = _capture_with_project_instruction_row()

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    import tldw_chatbook.Widgets.Console.console_conversation_inspector as inspector_module

    class FakePath:
        """See ``test_console_context_modal.py``'s ``FakePath`` -- real
        ``path_validation.validate_path`` needs ``os.PathLike`` support
        (``__fspath__``), so a bare stand-in raises ``TypeError`` there and
        would make this save look "rejected" instead of exercising the
        happy path this test is pinning."""

        def __init__(self, *parts: str | Path) -> None:
            self._path = tmp_path.joinpath(*parts)

        @classmethod
        def home(cls) -> "FakePath":
            return cls(tmp_path)

        def __truediv__(self, other: str) -> "FakePath":
            return FakePath(self._path, other)

        def __fspath__(self) -> str:
            return str(self._path)

        def __getattr__(self, name: str):
            return getattr(self._path, name)

    monkeypatch.setattr(inspector_module, "Path", FakePath)

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        await _expand_first_exchange_call(pilot, modal)

        await pilot.click("#console-inspector-exchange-save-0-0")
        await pilot.pause()

    saved_files = list((tmp_path / "Downloads").glob("chatbook_exchange_*.json"))
    assert len(saved_files) == 1
    saved = saved_files[0].read_text(encoding="utf-8")
    assert _EXCHANGE_EXPORT_SENTINEL not in saved
    assert "ordinary message" in saved
    assert "omitted by capture policy" in saved


@pytest.mark.asyncio
async def test_per_message_collapsible_mounts_its_json_body() -> None:
    """Finding 4 (task-9 review): the fourth lazy level (per-message,
    nested inside the Messages section) was never exercised by any test --
    ``_mount_exchange_message_body`` and its dispatch branch were dead
    coverage. Expand it and assert the message's own content actually
    renders (and, before that, that it genuinely was not mounted yet)."""
    cap = _capture(
        "r1", 0, "t", "m",
        request={"messages_payload": [{"role": "user", "content": "hello"}]},
    )

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)
        call.collapsed = False
        await _wait_until(pilot, lambda: bool(call.query(Collapsible)))

        messages_section = call.query_one(
            "#console-inspector-exchange-section-0-0-messages", Collapsible
        )
        messages_section.collapsed = False
        await _wait_until(pilot, lambda: bool(messages_section.query(Collapsible)))

        message_collapsible = messages_section.query_one(
            "#console-inspector-exchange-message-0-0-0", Collapsible
        )
        assert not messages_section.query(TextArea)  # still lazy pre-expand

        message_collapsible.collapsed = False
        await _wait_until(pilot, lambda: bool(messages_section.query(TextArea)))

        [text_area] = messages_section.query(TextArea)
        assert "hello" in text_area.text


@pytest.mark.asyncio
async def test_call_level_mount_failure_does_not_mark_it_loaded_and_retries() -> None:
    """Finding 5 (task-9 review): before the fix, a call's id was added to
    ``_loaded_exchange_call_keys`` BEFORE ``_mount_exchange_call_body`` ran
    -- a failed mount (e.g. the capture not yet cached) permanently
    blocked any retry, unlike the turn level's own discard-on-failure
    contract. Simulates that failure by emptying the capture cache right
    before the call first expands, then restoring it and re-expanding --
    the call must mount successfully on the second attempt."""
    cap = _capture("r1", 0, "t", "m")

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        return [(cap, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: bool(turn.query(Collapsible)))

        call = turn.query_one("#console-inspector-exchange-call-0-0", Collapsible)

        # Simulate a mount failure: the capture cache is empty at the
        # moment the call first expands.
        saved_capture = modal._exchange_capture_by_call_key.pop("0-0")
        call.collapsed = False
        await pilot.pause()

        assert "0-0" not in modal._loaded_exchange_call_keys
        assert not call.query(Collapsible)  # nothing mounted -- no sections

        # Restore the capture and retry via collapse/re-expand.
        modal._exchange_capture_by_call_key["0-0"] = saved_capture
        call.collapsed = True
        await pilot.pause()
        call.collapsed = False
        await pilot.pause()

        assert "0-0" in modal._loaded_exchange_call_keys
        assert call.query(Collapsible)  # sections mounted this time


@pytest.mark.asyncio
async def test_exchange_calls_ordered_by_created_at_not_arrival_or_run_tag() -> None:
    """Finding 6 (task-9 review): nothing pinned the Exchange tab's OWN
    ``(created_at, seq)`` re-sort in ``_load_exchange_turn`` --
    ``test_status_badges`` feeds already-chronological ``created_at``
    values, so it cannot discriminate a missing sort. Three captures whose
    run_tag ordering, loader arrival order, AND model-name alphabetical
    order are all inverted against their ``created_at`` order -- only a
    genuine chronological sort produces the right rendered order."""
    early = _capture("run-c", 0, "2026-08-17T09:00:00Z", "model-early")
    middle = _capture("run-b", 0, "2026-08-17T10:00:00Z", "model-middle")
    late = _capture("run-a", 0, "2026-08-17T11:00:00Z", "model-late")

    async def loader(_native_message_id: str) -> list[tuple[ExchangeCapture, bool]]:
        # Handed back in REVERSE chronological order; run_tag ("run-a" <
        # "run-b" < "run-c") and model name also both sort the OPPOSITE of
        # created_at -- only (created_at, seq) can produce the right order.
        return [(late, False), (middle, False), (early, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=loader, initial_tab=TAB_EXCHANGE)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        turn = modal.query_one("#console-inspector-exchange-turn-0", Collapsible)
        turn.collapsed = False
        await _wait_until(pilot, lambda: len(turn.query(Collapsible)) == 3)

        titles = [_rendered_title(c) for c in turn.query(Collapsible)]
        early_index = next(i for i, t in enumerate(titles) if "model-early" in t)
        middle_index = next(i for i, t in enumerate(titles) if "model-middle" in t)
        late_index = next(i for i, t in enumerate(titles) if "model-late" in t)
        assert early_index < middle_index < late_index, (
            f"expected chronological order (early, middle, late), got {titles!r}"
        )


# --- Next Send worker isolation (task-10 review finding 2) -----------------


@pytest.mark.asyncio
async def test_default_group_worker_error_does_not_toast_next_send_or_clear_its_spinner(
    monkeypatch,
) -> None:
    """task-10 review finding 2a: this screen runs the Costs tab's
    ``_load_turn_captures`` and the Exchange tab's ``_load_exchange_turn``
    workers in Textual's "default" group (neither passes ``group=``, same
    as ``run_worker``'s own default). Before the fix,
    ``on_worker_state_changed`` was UNFILTERED, so ANY worker owned by
    this screen reaching ``WorkerState.ERROR`` -- not just the Next Send
    snapshot load -- toasted "Failed to refresh context." and cleared
    ``next_send_loading``, even though the failure had nothing to do with
    Next Send.

    Simulates a Costs/Exchange-shaped failure directly: a throwaway
    coroutine in the SAME "default" group, ``exit_on_error=False`` so the
    simulated failure surfaces as ``WorkerState.ERROR`` without crashing
    the harness (mirrors how a real, uncaught mount-path exception in
    ``_load_turn_captures``/``_load_exchange_turn`` would be reported --
    neither of those call sites passes ``exit_on_error=False`` themselves,
    but Textual still sets the ERROR state before acting on that flag)."""
    never_ready = asyncio.Event()

    async def _blocking_snapshot() -> ConsoleContextSnapshot:
        await never_ready.wait()
        return ConsoleContextSnapshot(current_messages=[], next_send_payload={})

    app = InspectorHarness(**_default_kwargs(snapshot_factory=_blocking_snapshot))

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert modal.next_send_loading, "expected the snapshot load still in flight"

        notifications: list[str] = []
        monkeypatch.setattr(
            modal, "notify", lambda message, *a, **k: notifications.append(message)
        )

        async def _boom() -> None:
            raise RuntimeError("simulated Costs/Exchange mount-path failure")

        worker = modal.run_worker(
            _boom(), group="default", exclusive=False, exit_on_error=False
        )
        await _wait_until(pilot, lambda: worker.state == WorkerState.ERROR)
        await pilot.pause()

        assert modal.next_send_loading, (
            "an unrelated default-group worker's error must not clear the "
            "Next Send tab's own spinner"
        )
        assert not notifications, (
            f"expected no toast from an unrelated worker's error, got {notifications!r}"
        )

        never_ready.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_next_send_refresh_does_not_cancel_an_in_flight_costs_capture_load() -> None:
    """task-10 review finding 2b: before the fix, the snapshot worker's
    ``exclusive=True`` lived in the SAME "default" group as the Costs
    tab's ``_load_turn_captures`` worker (``exclusive`` cancels every
    OTHER worker in its OWN group) -- refreshing Next Send (Refresh
    button / "r") while a Costs row's capture load was still in flight
    would cancel it. Since ``_loaded_row_indices`` already marks that row
    loaded BEFORE the worker starts (``_on_row_toggled``), a cancelled
    load left the row permanently empty, with no retry short of
    reopening the whole modal. Now on its own worker group, a Next Send
    refresh must leave an in-flight Costs capture load free to complete."""
    still_loading = asyncio.Event()
    capture = _capture("run-1", 0, "2026-08-17T09:00:00Z", "gpt-4")

    async def slow_loader(
        _native_message_id: str,
    ) -> list[tuple[ExchangeCapture, bool]]:
        await still_loading.wait()
        return [(capture, False)]

    app = InspectorHarness(
        **_default_kwargs(exchanges_loader=slow_loader, initial_tab=TAB_NEXT_SEND)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen

        row = modal.query_one("#console-inspector-cost-row-0", Collapsible)
        row.collapsed = False
        await pilot.pause()  # let the Costs worker start and block on the event

        await pilot.click("#console-inspector-next-send-refresh")
        await pilot.pause()

        still_loading.set()

        def _has_call_line() -> bool:
            return any(
                "call 0" in str(static.renderable) for static in row.query(Static)
            )

        await _wait_until(pilot, _has_call_line)


@pytest.mark.asyncio
async def test_refresh_shows_the_refreshed_estimate_in_the_same_refresh() -> None:
    """Review finding M15: no test anywhere passed ``estimate_factory=``,
    so task-10 review finding 6's fix (``_load_snapshot`` re-estimates
    BEFORE reassigning ``self.snapshot``, so the header shows the NEW
    estimate in the same refresh rather than one refresh stale) had zero
    coverage. Two distinguishable estimates: the first from ``on_mount``'s
    initial load, the second from a Refresh click.

    ``snapshot_factory`` returns a distinguishable payload on every call --
    the default ``_noop_snapshot`` returns a snapshot that dataclass-equals
    the reactive's own default (both empty), and Textual's ``reactive``
    skips the watcher (so ``_update_view`` never runs) when a reassignment
    doesn't change the value -- that would mask this exact bug rather than
    exercise it.
    """
    calls = {"estimate": 0, "snapshot": 0}

    def estimate_factory() -> int:
        calls["estimate"] += 1
        return 111 if calls["estimate"] == 1 else 222

    async def snapshot_factory() -> ConsoleContextSnapshot:
        calls["snapshot"] += 1
        return ConsoleContextSnapshot(
            current_messages=[], next_send_payload={"call": calls["snapshot"]}
        )

    app = InspectorHarness(
        **_default_kwargs(
            estimate_factory=estimate_factory,
            snapshot_factory=snapshot_factory,
            initial_tab=TAB_NEXT_SEND,
        )
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        header = modal.query_one("#console-inspector-next-send-header", Static)
        await _wait_until(pilot, lambda: "~111 tokens" in str(header.renderable))

        await pilot.click("#console-inspector-next-send-refresh")
        await _wait_until(pilot, lambda: "~222 tokens" in str(header.renderable))
        assert "~111" not in str(header.renderable)


# ---------------------------------------------------------------------------
# I1: ``_focus_initial_control`` runs from ``on_mount`` AND from
# ``_load_snapshot``'s tail on EVERY tab (the Next Send prefetch itself
# starts unconditionally in ``on_mount``, regardless of ``initial_tab``) --
# not just while the Next Send tab is active. Both call sites must now do
# nothing unless the Next Send tab is the ACTIVE one at the moment the
# callback actually fires.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opening_on_costs_tab_does_not_focus_close() -> None:
    """Before the fix, Close was the one selector NOT gated on
    ``next_send_active`` -- opening the cost-chip entry point (starts on
    the Costs tab) still fell through to focusing it, so Enter dismissed
    the modal and arrow-key tab switching was no longer immediate."""
    app = InspectorHarness(**_default_kwargs(initial_tab=TAB_COSTS))

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert modal.query_one("#console-inspector-tabs").active == TAB_COSTS
        close_button = modal.query_one(f"#{CLOSE_BUTTON_ID}", Button)
        assert app.focused is not close_button


@pytest.mark.asyncio
async def test_late_snapshot_completion_does_not_steal_focus_from_costs_tab() -> None:
    """The real ``snapshot_factory`` is slow (resolves the provider, reads
    ``AGENTS.md`` off-thread) -- a user already drilled into a Costs-tab
    row must not have focus yanked to Close (or anywhere else) when that
    background Next Send load completes after the fact."""
    still_loading = asyncio.Event()

    async def slow_snapshot() -> ConsoleContextSnapshot:
        await still_loading.wait()
        return ConsoleContextSnapshot(current_messages=[], next_send_payload={})

    app = InspectorHarness(
        **_default_kwargs(snapshot_factory=slow_snapshot, initial_tab=TAB_COSTS)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert modal.query_one("#console-inspector-tabs").active == TAB_COSTS

        row_title = modal.query_one(
            "#console-inspector-cost-row-0", Collapsible
        ).query_one(CollapsibleTitle)
        row_title.focus()
        await pilot.pause()
        assert app.focused is row_title

        # Let the still-in-flight Next Send snapshot (started unconditionally
        # from on_mount) resolve and run its focus-scheduling tail.
        still_loading.set()
        await pilot.pause()
        await pilot.pause()

        close_button = modal.query_one(f"#{CLOSE_BUTTON_ID}", Button)
        assert app.focused is not close_button
        assert app.focused is row_title
