"""Original Agent overview/fleet display values, detached from native owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..Agents.agent_models import TERMINAL_RUN_STATUSES
from .console_execution_projection_values import (
    _boolean,
    _fail,
    _integer,
    _record,
    _text,
    _tuple,
)
from .console_glyphs import STATUS_GLYPHS
from .console_inspector_state import ConsoleInspectorSectionState, InspectorSectionRow

if TYPE_CHECKING:
    from ..Agents.fleet_coordinator import FleetHandle
    from .console_agent_bridge import AgentLiveSnapshot


@dataclass(frozen=True, slots=True)
class ConsoleAgentRailInputs:
    """Actual owner captures; None live snapshot means the bridge is absent.

    Setup/bound facts come from the original standard-bridge checks. Historical
    input must be independently settled whenever original precedence reads it.
    Live-run snapshots align with handles, including observed absent snapshots.
    The owner settles stale drill scope and guards session/visit currentness.
    """

    conversation_id: str
    live_snapshot: AgentLiveSnapshot | None
    historical_snapshot: AgentLiveSnapshot | None
    handles: tuple[FleetHandle, ...]
    live_run_snapshots: tuple[AgentLiveSnapshot | None, ...]
    setup_established: bool
    primary_run_bound: bool
    fleet_running: int
    fleet_pending: int
    pause_reason: str | None
    wake_has_pending: bool
    has_history_picker: bool
    full_log_available: bool
    drill_run_id: str | None
    observed_at: float


@dataclass(frozen=True, slots=True)
class ConsoleAgentRailProjection:
    """Display records only; no operation or cancellation authority."""

    overview_lines: tuple[str, str, str]
    fleet_summary: str
    fleet_section: ConsoleInspectorSectionState
    cancel_all_visible: bool
    fleet_token_total: int
    full_log_available: bool


def _historical(inputs: ConsoleAgentRailInputs) -> AgentLiveSnapshot:
    if inputs.historical_snapshot is None:
        raise ValueError("Agent rail historical snapshot is unavailable")
    return inputs.historical_snapshot


def _fleet_rows(inputs: ConsoleAgentRailInputs) -> tuple[InspectorSectionRow, ...]:
    from ..UI.Console_Modules.agent import (
        _fleet_row_from_handle,
        _fleet_row_from_summary,
    )

    if inputs.live_snapshot is None or not inputs.conversation_id:
        return ()
    if inputs.handles:
        return tuple(
            _fleet_row_from_handle(
                handle,
                now=inputs.observed_at,
                live_snapshot=live if handle.run_id else None,
            )
            for handle, live in zip(
                inputs.handles, inputs.live_run_snapshots, strict=True
            )
        )
    live = inputs.live_snapshot
    established = (live.status == "setup" and inputs.setup_established) or (
        live.status == "running" and inputs.primary_run_bound
    )
    snapshot = live if established else _historical(inputs)
    return tuple(
        _fleet_row_from_summary(summary, index)
        for index, summary in enumerate(snapshot.subagents)
    )


def build_agent_rail_projection(
    inputs: ConsoleAgentRailInputs,
) -> ConsoleAgentRailProjection:
    """Apply the original overview and fleet presentation to actual captures.

    Args:
        inputs: Settled values from the bridge/controller/history owner.

    Returns:
        Original overview text and fleet records, without native references.

    Raises:
        ValueError: A drill is active, required history is missing, or captured
            handle/snapshot sequences do not align.
    """
    if inputs.drill_run_id:
        raise ValueError("Agent rail drill requires its separate parent checkpoint")
    if len(inputs.handles) != len(inputs.live_run_snapshots):
        raise ValueError("Agent rail handle and live snapshot captures do not align")

    snapshot = inputs.live_snapshot
    if snapshot is None:
        overview = ("Agent: unavailable", "", "")
    else:
        if snapshot.status == "idle":
            snapshot = _historical(inputs)
        status = f"Agent: {snapshot.status}"
        if snapshot.status == "running":
            status = f"Agent: running · step {snapshot.step}"
        overview = (
            status,
            "\n".join(f"· {step.text}" for step in snapshot.steps),
            "\n".join(
                f"{STATUS_GLYPHS.get(child.status, '●')} {child.text}"
                for child in snapshot.subagents
            ),
        )

    rows = () if inputs.pause_reason == "history_unavailable" else _fleet_rows(inputs)
    if inputs.pause_reason:
        from ..UI.Console_Modules.agent import _automatic_work_pause_notice

        section = ConsoleInspectorSectionState(
            rows=rows,
            summary="Paused",
            notice=_automatic_work_pause_notice(
                inputs.pause_reason,
                results_saved=(
                    inputs.pause_reason != "history_unavailable"
                    or inputs.wake_has_pending
                ),
            ),
        )
    elif not rows:
        section = ConsoleInspectorSectionState(
            rows=(),
            summary=(
                "Run history"
                if inputs.live_snapshot is not None
                and inputs.conversation_id
                and inputs.has_history_picker
                else ""
            ),
        )
    else:
        working = sum(row.status not in TERMINAL_RUN_STATUSES for row in rows)
        summary = f"{working} working, {len(rows) - working} done"
        if len(rows) <= 4:
            glyphs = "".join(STATUS_GLYPHS.get(row.status, "●") for row in rows)
            summary = f"{glyphs} {summary}"
        section = ConsoleInspectorSectionState(rows=rows, summary=summary)

    handles = (
        inputs.handles
        if inputs.live_snapshot is not None and inputs.conversation_id
        else ()
    )
    return ConsoleAgentRailProjection(
        overview_lines=overview,
        fleet_summary=(
            f"{inputs.fleet_running} other agents running, "
            f"{inputs.fleet_pending} waiting for approval."
            if inputs.fleet_running + inputs.fleet_pending > 0
            else ""
        ),
        fleet_section=section,
        cancel_all_visible=any(
            handle.status not in TERMINAL_RUN_STATUSES for handle in handles
        ),
        fleet_token_total=sum(handle.total_tokens for handle in handles),
        full_log_available=inputs.full_log_available,
    )


_ROW = {
    "row_id": _text,
    "primary_text": _text,
    "secondary_text": _text,
    "status": _text,
    "clickable": _boolean,
    "cancellable": _boolean,
    "indent": _integer,
    "wrap_secondary": _boolean,
}


def _row(value: object, decode: bool):
    return _record(value, InspectorSectionRow, _ROW, decode)


_SECTION = {"rows": _tuple(_row), "summary": _text, "notice": _text}


def _section(value: object, decode: bool):
    return _record(value, ConsoleInspectorSectionState, _SECTION, decode)


def _overview(value: object, decode: bool):
    result = _tuple(_text)(value, decode)
    if len(result) != 3:
        _fail("expected three Agent overview lines")
    return result


_PROJECTION = {
    "overview_lines": _overview,
    "fleet_summary": _text,
    "fleet_section": _section,
    "cancel_all_visible": _boolean,
    "fleet_token_total": _integer,
    "full_log_available": _boolean,
}


def encode_agent_rail_projection(projection: ConsoleAgentRailProjection) -> dict:
    """Detach the exact original row fields; no handle or clock can cross."""
    return _record(projection, ConsoleAgentRailProjection, _PROJECTION, False)


def decode_agent_rail_projection(payload: dict) -> ConsoleAgentRailProjection:
    """Restore original records from a closed, exact-primitive schema."""
    return _record(payload, ConsoleAgentRailProjection, _PROJECTION, True)
