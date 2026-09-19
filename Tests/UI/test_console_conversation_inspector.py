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

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from loguru import logger
from textual.app import ComposeResult
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    Static,
)
from textual.widgets._collapsible import CollapsibleTitle

import tldw_chatbook.Widgets.Console.console_conversation_inspector as inspector_module
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Chat.console_chat_controller import (
    CapturePolicySnapshot,
    CapturePurgeAvailability,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import (
    ConsoleCostRow,
    ConsoleCostRowTotals,
    build_cost_rows,
)
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    CapturePolicyResolution,
    CapturePolicySource,
    ExchangeCapture,
    build_request_capture,
    compact_safe_history_rows,
    history_elision_marker,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Utils.log_sanitizer import content_fingerprint
from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
    CapturePolicyBindings,
)
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    CLOSE_BUTTON_ID,
    TAB_COSTS,
    TAB_EXCHANGE,
    TAB_NEXT_SEND,
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


def _capture_policy_bindings_for_inspector() -> CapturePolicyBindings:
    snapshot = CapturePolicySnapshot(
        session_id="session-at-open",
        conversation_id="conversation-at-open",
        conversation_title="Immutable titled chat",
        enabled=True,
        next_detail=CaptureDetail.FULL,
        conversation_detail=CaptureDetail.SAFE,
        global_detail=CaptureDetail.SAFE,
        effective=CapturePolicyResolution(
            True,
            CaptureDetail.FULL,
            CapturePolicySource.NEXT_SEND,
            (),
        ),
        policy_revision=1,
        config_generation=2,
        capture_revision=3,
        active_run_detail=None,
        queued_consumer=False,
        save_pending=False,
        error_code=None,
    )

    async def unused_async(*_args, **_kwargs):
        raise AssertionError("not used")

    return CapturePolicyBindings(
        target_session_id=snapshot.session_id,
        target_conversation_id=snapshot.conversation_id,
        read=lambda: snapshot,
        apply_next=lambda *_args: (_ for _ in ()).throw(AssertionError("not used")),
        apply_conversation=unused_async,
        apply_global=lambda *_args: (_ for _ in ()).throw(AssertionError("not used")),
        count_full=unused_async,
        purge_full=unused_async,
        capture_revision=lambda: snapshot.capture_revision,
        purge_availability=lambda: CapturePurgeAvailability(True, None),
    )


@pytest.mark.asyncio
async def test_compact_capture_status_names_title_and_armed_next_send() -> None:
    app = InspectorHarness(
        **_default_kwargs(
            capture_policy_bindings=_capture_policy_bindings_for_inspector(),
        )
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        status = app.screen.query_one("#console-inspector-policy-status", Static)
        text = str(status.render())

        assert "Immutable titled chat" in text
        assert "Next eligible send: Full (armed)" in text
        assert text.count("\n") == 1
        assert status.region.height == 2


def _default_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "conversation_title": "Test chat",
        "target_profile_key": "test-profile",
        "target_is_current": lambda: True,
        "initial_tab": TAB_COSTS,
        "rows": [_row()],
        "totals": _totals(),
        "turns": [_turn()],
        "exchanges_loader": _empty_exchanges_loader,
        "snapshot_factory": _noop_snapshot,
    }
    kwargs.update(overrides)
    return kwargs


class InspectorHarness(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[Path]] = list(APP_STYLESHEETS)

    def __init__(self, **modal_kwargs: object) -> None:
        super().__init__()
        self._modal_kwargs = modal_kwargs

    def compose(self) -> ComposeResult:
        yield Static("background")

    def on_mount(self) -> None:
        self.push_screen(ConsoleConversationInspector(**self._modal_kwargs))


@pytest.mark.asyncio
async def test_rejected_export_logs_safe_path_and_preserves_notification(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "task-19864-private-inspector-export.json"
    raw_exception = f"TASK-19864 export rejected for {path}"

    def reject_path(*_args: object, **_kwargs: object) -> None:
        raise ValueError(raw_exception)

    monkeypatch.setattr(inspector_module, "validate_path", reject_path)
    app = InspectorHarness(**_default_kwargs())
    notifications: list[tuple[str, str | None]] = []
    records: list[str] = []
    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        monkeypatch.setattr(
            modal,
            "notify",
            lambda message, *args, severity=None, **kwargs: notifications.append(
                (message, severity)
            ),
        )
        sink_id = logger.add(lambda message: records.append(str(message)))
        try:
            assert modal._validated_export_destination(path) is None
        finally:
            logger.remove(sink_id)

    assert notifications == [(f"Save failed (ValueError): {path}", "error")]
    rendered = "".join(records)
    assert "Rejected export destination" in rendered
    assert f"path_sha256={content_fingerprint(str(path))}" in rendered
    assert "exception_type=ValueError" in rendered
    assert str(path) not in rendered
    assert path.name not in rendered
    assert raw_exception not in rendered


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
    capture_detail: CaptureDetail = CaptureDetail.SAFE,
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
        capture_detail=capture_detail,
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


# --- Exchange tab review fixes (task-9 review round) -----------------------


# Finding 2's hardcoded half of the pin -- see the test below for why this
# must NOT be computed from the live import.
_TODAY_CAPTURE_ALLOWLIST_SNAPSHOT = frozenset(
    {
        "api_endpoint",
        "api_base_url",
        "system_message",
        "messages_payload",
        "tools",
        "model",
        "streaming",
        "temp",
        "topp",
        "maxp",
        "topk",
        "minp",
        "max_tokens",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
        "prompt_caching",
        "response_format",
        "api_mode",
        "request_timeout",
        "request_retries",
        "request_retry_delay",
        "provider_continuations",
    }
)


# ---------------------------------------------------------------------------
# The governed Export action retains the redacted captured request and is
# the only disclosure path offered by the Exchange tab.
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


# --- Next Send worker isolation (task-10 review finding 2) -----------------


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
async def test_inspector_adopts_first_revision_after_opening_during_quiescence() -> (
    None
):
    revision = SimpleNamespace(value=None)
    app = InspectorHarness(
        **_default_kwargs(capture_revision_provider=lambda: revision.value)
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        assert modal._capture_revision_at_open is None

        revision.value = 7

        assert modal._capture_revision_is_current() is True
        assert modal._capture_revision_at_open == 7


def test_messages_section_title_states_original_and_elided_counts() -> None:
    """task-23026 / ADR-096: a Safe capture's stored list is COMPACTED, so
    the physical row count alone would under-state what the call actually
    sent. When the aggregate history-elision marker is present, the
    Messages title must surface its original/omitted counts; without a
    marker the plain physical count remains."""
    rows = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"row {i}"}
        for i in range(17)
    ]
    compacted, _ = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    assert history_elision_marker(compacted) is not None  # sanity

    inspector = object.__new__(ConsoleConversationInspector)
    compacted_capture = _capture(
        "run-1", 0, "t", "m", request={"messages_payload": compacted}
    )
    title = inspector._messages_summary(compacted_capture)
    assert f"{len(rows)} sent" in title
    marker = history_elision_marker(compacted)
    assert f"{marker['omitted_rows']} elided by capture policy" in title

    plain_capture = _capture(
        "run-1", 0, "t", "m", request={"messages_payload": rows[-3:]}
    )
    plain_title = inspector._messages_summary(plain_capture)
    assert plain_title == "Messages (3)"


# --- task-25836: payload-based Next Send header token estimate --------------


async def _payload_snapshot() -> ConsoleContextSnapshot:
    return ConsoleContextSnapshot(
        current_messages=[],
        next_send_payload={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )


@pytest.mark.asyncio
async def test_next_send_payload_estimate_replaces_header_count_once_loaded():
    """task-25836: after the snapshot loads, the Next Send header's "~N
    tokens" count must reflect the assembled next-send payload (system +
    tools + staged evidence included), not the draft-only factory value."""
    app = InspectorHarness(
        **_default_kwargs(
            snapshot_factory=_payload_snapshot,
            estimate_factory=lambda: 7,
            token_estimate=7,
            payload_estimate=lambda snapshot: 4242,
            initial_tab=TAB_NEXT_SEND,
        )
    )

    async with app.run_test(size=(120, 44)) as pilot:
        modal = app.screen
        header = modal.query_one("#console-inspector-next-send-header", Static)
        await _wait_until(pilot, lambda: "~4,242 tokens" in str(header.renderable))


@pytest.mark.asyncio
async def test_next_send_payload_estimate_none_falls_back_to_factory():
    """A payload estimate of ``None`` (nothing estimable, e.g. an
    assembly-error payload) falls back to the draft-only factory rather
    than dropping the count."""
    app = InspectorHarness(
        **_default_kwargs(
            snapshot_factory=_payload_snapshot,
            estimate_factory=lambda: 9,
            token_estimate=7,
            payload_estimate=lambda snapshot: None,
            initial_tab=TAB_NEXT_SEND,
        )
    )

    async with app.run_test(size=(120, 44)) as pilot:
        modal = app.screen
        header = modal.query_one("#console-inspector-next-send-header", Static)
        await _wait_until(pilot, lambda: "~9 tokens" in str(header.renderable))


# --- TASK-32336: human role names in the Current Context viewer ---------------


@pytest.mark.asyncio
async def test_usage_entry_does_not_prepare_hidden_context():
    calls = []

    async def snapshot_factory():
        calls.append("prepared")
        return ConsoleContextSnapshot(current_messages=[], next_send_payload={})

    app = InspectorHarness(
        **_default_kwargs(snapshot_factory=snapshot_factory, initial_tab=TAB_COSTS)
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert calls == []
