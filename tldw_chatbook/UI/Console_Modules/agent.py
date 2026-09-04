"""Console agent controller.

Extracted out of `ChatScreen` (wave-4 console decomposition, task 3): the
agent runtime's screen-side cluster -- the lazily-built `ConsoleAgentBridge`
handle, the Agent rail section's text derivation, the sub-agent drill-in
cycle, the "View full log" target/probe/loader, the fleet auto-open
override, the conversation-browser's `[N Sub-Agents]` badge-count cache, and
the resume-time TOOL-marker re-derivation.

The binding rule this file follows is stated canonically in
`ConsoleDictationController.__init__`'s docstring (`dictation.py`); the
construction itself lives in `Console_Modules/wiring.py` (wave-4 task 1),
not in `ChatScreen.__init__`.

Moved from `ChatScreen`, thirteen methods byte-for-byte plus one split:

- `_ensure_console_agent_bridge`, `_console_agent_section_lines`,
  `_console_agent_fleet_summary_line`, `_console_agent_full_log_run_id`,
  `_console_agent_full_log_available`, `_open_console_agent_run_log_viewer`,
  `_load_console_agent_run_log`, `_show_console_agent_run_log_modal`,
  `_toggle_console_agent_drilldown_from_subagents_click`,
  `_apply_fleet_agent_section_auto_open`,
  `_console_subagent_counts_refresh_needed`,
  `_console_subagent_counts_for_rows`, `_inject_resume_agent_markers`.
- `_console_agent_section_payload` -- the non-DOM half of
  `_sync_console_agent_section`. See that method's own docstring below.

Four bodies are NOT byte-for-byte, each for a reason the binding rule
already names:

1. `_load_console_agent_run_log` lost its `@work(thread=True)` decorator.
   Textual's decorator asserts `isinstance(self, DOMNode)` (`textual/
   _work_decorator.py`), and a controller is deliberately not one, so the
   dispatch moved to its caller as an explicit `run_worker(..., thread=
   True, group="default", name="_load_console_agent_run_log")`. Those are
   the exact values `@work(thread=True)` itself passes -- the pre-move
   worker group name is preserved, per the binding rule.
2. ...and its `self.app.call_from_thread(...)` became
   `self._screen.app.call_from_thread(...)`, reaching the canonical owning
   `App` bridge without adding a controller-local alias.
3. `_show_console_agent_run_log_modal`'s `self.app.push_screen(...)`
   became `self.push_screen(...)`, same reason (and the same shape
   `ConsoleMessageController` already uses).
4. `_console_subagent_counts_refresh_needed` imports its two module-level
   constants (`CONSOLE_ACTIVE_RUN_STATUSES`, `CONSOLE_SUBAGENT_COUNTS_
   CACHE_TTL_SECONDS`) inside the method body. They still live in
   `chat_screen.py` -- `Tests/UI/test_console_native_chat_flow.py` imports
   the first from there -- and a module-level import here would be a cycle
   that fails for real: `chat_screen.py` imports `Console_Modules.wiring`
   at line 89, which imports this module, and both constants are defined
   at line 587+, so a cold `import chat_screen` would find them missing.

**Zero DOM.** No `query_one`/`query`/`mount` traffic reaches through
`screen` here, matching every other controller. That is what keeps
`_sync_console_agent_section` on `ChatScreen`: its second half is nine
`query_one` calls against the mounted rail. Only its first half -- deriving
the seven-field payload, the part that reads the bridge, the run store, the
rail state and the full-log probe -- moved, as `_console_agent_section_
payload`. The screen's own method is now the equality guard plus the DOM
writes, and `_console_agent_section_last` (that guard's memo, and nothing
else's) stayed with it.

`ChatScreen` keeps exactly ONE one-line delegation, under its original
private name: `_ensure_console_agent_bridge`. Three test files replace it
on the screen instance (`Tests/Chat/test_change_turn_tracking.py` x2,
`Tests/Chat/test_console_agent_swap.py`) to steer five screen-level
consumers that are not part of this cluster -- the chat controller's
construction and its core-state sync, the conversation browser's badge
counts, and the Change Review opener -- so the screen must keep answering
to that name for those patches to keep working.

Every other moved method left no residue: their screen-side callers
(`compose_content`, `_toggle_console_rail_section`, `_with_console_
conversation_browser_state`, `on_click`, `on_button_pressed`) now reach
`self._agent.<name>()` directly, the same shape those methods already use
for `self._workspace`/`self._session`, and the ~30 test call sites were
repointed with them. That is deliberate: `chat_screen.py` is under a
method-count ratchet (`Tests/Architecture/test_screen_size_ratchet.py`),
and a delegation table of six pure forwards would have spent the whole of
this extraction's method-count gain on re-exporting names with one caller
each -- 599 methods against a 598 ceiling, versus 591 with them gone.

One narrowing worth stating plainly: a test that replaces `screen.
_ensure_console_agent_bridge` on the instance still steers every SCREEN-side
consumer (they call the delegation), but no longer steers this controller's
own internal bridge lookups, which call this class's method directly. Every
such site in the suite today drives screen-owned code
(`Tests/Chat/test_change_turn_tracking.py`, `Tests/Chat/test_console_agent_
swap.py`), so nothing regressed; a future test that needs to stub the
bridge for the rail itself should set `screen._console_agent_bridge`
instead, which is what `Tests/UI/test_console_agent_rail.py` already does
(and which still works, via the read-write proxy on the screen).

`ChatScreen` keeps three read-write proxy properties for the cluster state
its own non-agent code still touches -- `_console_agent_bridge` (read by
`_build_console_inspector_state`), `_console_agent_drilldown_run_id` (read
by `compose_content`, cleared by `on_button_pressed`'s Back branch and by
`ConsoleSessionController`/`ConsoleWorkspaceController` through their own
same-named proxies) and `_agent_section_user_dismissed_while_busy` (written
by `_toggle_console_rail_section`). Each is read-WRITE: all three were
plain assignable attributes at baseline, and the suite assigns two of them
20+ times. The cluster's other six attributes have no proxy at all.

`_console_agent_runtime_enabled` is deliberately NOT part of this cluster,
despite matching on name. It is a four-line `[console] agent_runtime`
config read whose only two callers are the screen's own chat-controller
construction and core-state sync; it touches no agent state, and its
identical twin two lines below it in `chat_screen.py`
(`_console_native_tool_calls_enabled`, same shape, same consumers) is not
name-matched and would have been split away from it. See the task-3 report
for the full per-method verdict table.

PR2b Task 4 (supervisor fleet, spec `Docs/superpowers/specs/2026-08-08-
supervisor-agent-fleet-design.md` section 7) retired one method from the
list above and added its replacement: `_toggle_console_agent_drilldown_
from_subagents_click` (the "click the joined sub-agents line to cycle to
the next run" affordance) is GONE, replaced by `_drill_into_console_agent_
subagent(row_id)` -- a specific row (posted via `ConsoleInspectorSection.
RowActivated`, caught by the screen) now resolves directly to its own run,
rather than stepping through every run one click at a time. The three new
row-building methods (`_console_agent_fleet_rows`,
`_console_agent_fleet_section_state`, and the resolver
`_console_agent_drilldown_target_run_id`) feed the `ConsoleInspectorSection`
component (Task 3) that now renders inside the Agent rail section's body,
replacing the old single joined-string Static. `_console_agent_section_
lines` itself is UNCHANGED in shape (still a 3-tuple; still directly unit
tested by `Tests/UI/test_console_agent_rail.py`/`test_console_agent_
controller.py`) -- only its hard `[:60]` slice was retired (the component
now owns width truncation via CSS ellipsis) -- but its third element (the
old joined sub-agents string) is no longer painted into the DOM by
`_console_agent_section_payload`; the fleet rows are derived independently,
straight from `ConsoleAgentBridge.fleet_snapshot`/`historical_snapshot`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Iterable, TYPE_CHECKING

import re
import time

from loguru import logger
from textual.message_pump import NoActiveAppError

from ...Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    STEP_TOOL_CALL,
    TERMINAL_RUN_STATUSES,
)
from ...Chat.cost_display import format_token_count
from ...Chat.console_chat_models import ConsoleMessageRole
from ...Widgets.Console.console_agent_steering_bar import (
    STEERING_STATE_HIDDEN,
    ConsoleAgentSteeringState,
)
from ...Widgets.Console.console_inspector_section import (
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)
from ...Widgets.Console.console_run_log_modal import ConsoleRunLogModal
from ...Widgets.Console.console_transcript import CONSOLE_GENERATING_PLACEHOLDER

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...Agents.fleet_coordinator import FleetHandle
    from ...Chat.console_agent_bridge import SubAgentSummary
    from ...Chat.console_chat_models import ConsoleChatMessage
    from ...Chat.console_rail_state import ConsoleRailState
    from ..Screens.chat_screen import ChatScreen

#: Status glyph, shared by the text-only overview builder
#: (`_console_agent_section_lines`) and the fleet row/summary builders
#: below -- one vocabulary, one place (agent.py:354/:457-463 in the PR2b
#: seam map). Unknown/future statuses fall back to "●" wherever this is
#: consulted via `.get(status, "●")`.
_AGENT_STATUS_GLYPHS: Dict[str, str] = {
    "done": "✓",
    "running": "●",
    "stuck": "⚠",
    "error": "✗",
    "cancelled": "✗",
}

#: TASK-31429: the rail's Agent status line ("Agent: running · step 3",
#: "Sub-agent · done") carries its run status as the first word after the
#: prefix; that word selects the line's `$ds-status-*` colour class.
_AGENT_STATUS_LINE_RE = re.compile(r"^(?:Agent:|Sub-agent ·)\s*([a-z-]+)")


def console_agent_status_state(status_line: str) -> str:
    """Return the run status a rail status line carries, or "" to stay uncoloured.

    Only the statuses in ``_AGENT_STATUS_GLYPHS`` are colour-worthy; "idle",
    "unavailable", and anything unrecognised return "".
    """
    match = _AGENT_STATUS_LINE_RE.match(status_line or "")
    status = match.group(1) if match else ""
    return status if status in _AGENT_STATUS_GLYPHS else ""


def apply_console_agent_status_state(widget: Any, status_line: str) -> None:
    """Swap ``widget``'s ``console-agent-section-status-<state>`` class to match.

    Exactly one state class (or none) is present after the call; the base
    classes are left alone.
    """
    state = console_agent_status_state(status_line)
    for status in _AGENT_STATUS_GLYPHS:
        widget.set_class(status == state, f"console-agent-section-status-{status}")


#: The ``ConsoleInspectorSection.section_id`` the fleet mini-section is
#: constructed with (``left_rail.py``'s ``compose()``) and the id
#: ``chat_screen.py``'s ``RowActivated`` handler matches against -- one
#: constant so the two sides can never drift apart.
CONSOLE_AGENT_FLEET_SECTION_ID = "agent-fleet"

#: The "Cancel all agents" button's DOM id (PR3b Task 5) -- constructed
#: in ``left_rail.py``'s ``compose()``, matched by ``chat_screen.py``'s
#: ``@on`` selector, and written by its ``_sync_console_agent_section``
#: apply; one constant, same drift rule as the section id above. The UI
#: suite (``test_console_agent_cancel_all.py``) deliberately pins the
#: LITERAL instead, so a silent rename here still fails a test.
CONSOLE_AGENT_CANCEL_ALL_ID = "console-agent-cancel-all"

#: Leading glyph for the turn-activity line's tool state. Same glyph the
#: transcript's completed TOOL markers use (``format_agent_step_marker``),
#: so "what is running now" and "what ran" read as one vocabulary.
CONSOLE_TURN_ACTIVITY_TOOL_GLYPH = "⚙"
#: The turn-activity line's between-tools state: the loop has handed the
#: last tool result back to the model and is waiting on the next round.
#: There is no step kind for "a model call started" (``STEP_MODEL`` is
#: emitted AFTER the round returns, carrying its text), so this is derived
#: from "the last primary step is not a tool call", not from an event.
CONSOLE_TURN_ACTIVITY_THINKING = "Thinking…"
#: Separator between the state and its elapsed segment.
CONSOLE_TURN_ACTIVITY_SEPARATOR = " · "

__all__ = [
    "ConsoleAgentController",
    "CONSOLE_AGENT_FLEET_SECTION_ID",
    "CONSOLE_TURN_ACTIVITY_SEPARATOR",
    "CONSOLE_TURN_ACTIVITY_THINKING",
    "CONSOLE_TURN_ACTIVITY_TOOL_GLYPH",
    "console_turn_activity_text",
]


def _format_fleet_elapsed(seconds: float | None) -> str:
    """Format an already-computed duration as ``"Ns"``/``"Nm Ss"``/``"<1s"``.

    Mirrors ``Library.library_ingest_state._format_elapsed``'s exact
    grammar, but takes a precomputed duration instead of two raw
    endpoints -- the one fleet row source that can compute elapsed at all
    (live ``FleetHandle``s) uses ``time.monotonic()`` floats, so the
    "compute a duration" step happens once, at the call site, before
    reaching this shared formatter.

    Args:
        seconds: The duration to format, or ``None``/negative when there is
            no usable base (mirrors ``_format_elapsed``'s own "no usable
            base -> claiming a duration would be a lie" reasoning).

    Returns:
        ``""`` when ``seconds`` is ``None`` or negative -- the caller omits
        the segment; ``"<1s"`` under one second; otherwise ``"Ns"`` under a
        minute, or ``"Nm Ss"`` at or above a minute.
    """
    if seconds is None or seconds < 0:
        return ""
    if seconds < 1:
        return "<1s"
    total_seconds = int(round(seconds))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes}m {secs}s"


def console_turn_activity_text(snapshot: Any, *, now: float) -> str:
    """Return the live activity line for one in-flight Console turn.

    **Live-only, by construction.** There is no resumed counterpart and
    there must never be one: this line reports what an agent is doing
    RIGHT NOW, and a finished turn's transcript already carries the TOOL
    markers that say what it did. (Same "live rendering has no resumed
    twin" note ``format_todo_marker`` carries -- unlike
    ``format_agent_step_marker``, which is deliberately shared by the live
    and resume paths so both render byte-identical text.)

    The four states, derived from the primary agent's most recent step:

    ===========================  ==========================================
    situation                    line
    ===========================  ==========================================
    a tool is running            ``⚙ <tool> · <elapsed>``
    between tools / after one    ``Thinking… · <elapsed>``
    running, no primary step     ``Generating…`` (today's copy, unchanged)
    turn ended (any non-running) ``""`` -- the caller renders nothing
    ===========================  ==========================================

    Sub-agent steps are skipped, not merely deprioritised: a child's work
    belongs to the Agent rail's fleet rows, never to the primary assistant
    row. This is a real path, not a defensive one -- ``ConsoleAgentBridge.
    on_step`` routes a SUB-AGENT step whose run id is empty into the
    PRIMARY run's own live feed (its documented "no run attributed"
    fallback), so ``snapshot.steps[-1]`` can genuinely belong to a child.

    Quiet catalog tools (``find_tools``/``load_tools``) DO appear here.
    ``_QUIET_STEP_TOOLS`` keeps them out of the permanent, append-only TOOL
    markers, where a discovery round would be lasting clutter; this line is
    ephemeral and exists precisely so no working moment looks frozen, and
    suppressing them would restore a silent gap for the whole round.

    ``STEP_MODEL``'s summary is never shown. It carries the raw model turn
    text -- mid-turn that is the tool-call fence itself -- so the state, not
    the summary, is what the user sees.

    Args:
        snapshot: The conversation's ``AgentLiveSnapshot`` (duck-typed:
            ``status`` plus a ``steps`` sequence).
        now: ``time.monotonic()`` reading for this poll tick, injected so
            the elapsed segment is testable without sleeping.

    Returns:
        The line to render, or ``""`` when nothing is live.
    """
    if getattr(snapshot, "status", "idle") != "running":
        return ""
    step = next(
        (
            candidate
            for candidate in reversed(tuple(getattr(snapshot, "steps", ()) or ()))
            if getattr(candidate, "agent_kind", "") == AGENT_KIND_PRIMARY
        ),
        None,
    )
    if step is None:
        # Pre-first-token: the model has not come back once yet, so there is
        # no step to name and no honest base to time from.
        return CONSOLE_GENERATING_PLACEHOLDER
    if step.kind == STEP_TOOL_CALL:
        # `AgentLiveStep.text` for a tool-call step IS the tool name:
        # `agent_runtime` adds every STEP_TOOL_CALL with `tool_name=` and
        # neither `summary` nor `result`, and `_summarize`'s precedence
        # (summary or result or tool_name or kind) therefore lands on it.
        label = f"{CONSOLE_TURN_ACTIVITY_TOOL_GLYPH} {step.text}"
    else:
        label = CONSOLE_TURN_ACTIVITY_THINKING
    started_at = getattr(step, "started_at", None)
    elapsed = (
        _format_fleet_elapsed(max(0.0, now - started_at))
        if started_at is not None
        else ""
    )
    return f"{label}{CONSOLE_TURN_ACTIVITY_SEPARATOR}{elapsed}" if elapsed else label


def _fleet_row_from_handle(handle: "FleetHandle", *, now: float) -> InspectorSectionRow:
    """Build one fleet row from a LIVE ``FleetCoordinator`` handle.

    ``row_id`` is the handle's own ``handle_id`` -- stable across the
    handle's whole life (reserved once, never reassigned), unlike its
    ``run_id`` (empty until ``attach_run`` fires a moment later). Using a
    value that can CHANGE mid-flight as the row's structural identity would
    make ``ConsoleInspectorSection``'s structural key see a different row
    the instant the run attaches, forcing an avoidable recompose right when
    the row becomes genuinely interesting.

    ``clickable`` requires a non-empty ``run_id`` -- there is nothing to
    drill into yet for a handle whose run hasn't attached.

    ``cancellable`` (PR2b Task 5) is true only while ``status`` is still
    live (not in ``TERMINAL_RUN_STATUSES``) -- a finished/errored/cancelled
    child has nothing left to cooperatively stop, and offering the gesture
    for a stale row would just make ``AgentService.cancel_subagent`` no-op
    silently on press.

    The secondary line's trailing token segment (PR2b Task 5) reads
    ``handle.total_tokens`` -- 0, and so omitted, until ``FleetCoordinator.
    finish()`` records the child's real ``RunOutcome.total_tokens`` spend;
    a still-running child's spend is not final, so nothing is shown for it
    rather than a partial, growing-then-frozen number.
    """
    status = handle.status or "running"
    glyph = _AGENT_STATUS_GLYPHS.get(status, "●")
    name = handle.agent or "sub-agent"
    primary = f"{glyph} {name}"
    if handle.started_at:
        end = handle.finished_at if handle.finished_at is not None else now
        elapsed = _format_fleet_elapsed(max(0.0, end - handle.started_at))
        if elapsed:
            primary = f"{primary} · {elapsed}"
    secondary = (handle.error or handle.result or handle.task or "").strip()
    if handle.total_tokens:
        token_segment = f"{format_token_count(handle.total_tokens)} tok"
        secondary = f"{secondary} · {token_segment}" if secondary else token_segment
    # PR3b Task 3 (spec §6 latency honesty): a posted steering entry is
    # QUEUED until the child's next drain boundary consumes it, and the
    # row says so. `queued_steering` is computed onto every coordinator
    # copy from the mailbox itself (never stored on the live handle), so
    # this figure can never disagree with what the drain would deliver;
    # it defaults 0 for the historical/fallback row sources, which never
    # reach this builder anyway. Terminal rows drop the segment (Qodo
    # audit minor batch): a finished/errored/cancelled child never drains
    # its mailbox again, so "steering queued (N)" on such a row is a
    # delivery promise the app can no longer keep -- the mailbox copy
    # still reports the count, which is exactly why the gate lives here.
    queued_steering = int(getattr(handle, "queued_steering", 0) or 0)
    if queued_steering and status not in TERMINAL_RUN_STATUSES:
        steering_segment = f"steering queued ({queued_steering})"
        secondary = (
            f"{secondary} · {steering_segment}" if secondary else steering_segment
        )
    return InspectorSectionRow(
        row_id=handle.handle_id,
        primary_text=primary,
        secondary_text=secondary,
        status=status,
        clickable=bool(handle.run_id),
        cancellable=status not in TERMINAL_RUN_STATUSES,
    )


def _fleet_row_from_summary(
    summary: "SubAgentSummary", index: int
) -> InspectorSectionRow:
    """Build one fleet row from a HISTORICAL/resumed ``SubAgentSummary``.

    No elapsed segment: unlike a live ``FleetHandle``,
    ``AgentLiveSnapshot.subagents`` carries no timestamps (only
    ``text``/``status``/``run_id``/``handle_id`` -- see
    ``SubAgentSummary``'s own docstring), so there is nothing honest to
    compute a duration from here.

    ``row_id`` prefers ``run_id`` (populated for every REAL resumed row by
    ``ConsoleAgentBridge._derive_historical_snapshot``), falling back to
    ``handle_id`` and then a synthetic, index-based id only for a bare test
    double that constructs a ``SubAgentSummary`` with neither -- keeps the
    row's structural identity non-empty and unique even then.
    """
    status = summary.status or "running"
    glyph = _AGENT_STATUS_GLYPHS.get(status, "●")
    row_id = summary.run_id or summary.handle_id or f"idx-{index}"
    return InspectorSectionRow(
        row_id=row_id,
        primary_text=f"{glyph} {summary.text}".strip(),
        secondary_text="",
        status=status,
        clickable=bool(summary.run_id),
    )


def _fleet_row_from_record(record: dict) -> InspectorSectionRow:
    """Build one fleet row from a raw ``AgentRunsDB`` run dict.

    Last-resort fallback for a bridge stub that implements only
    ``subagent_runs`` (no ``historical_snapshot``) -- see
    ``_console_agent_fleet_rows``'s docstring for the full precedence.
    ``row_id`` is the record's own permanent id, so this is always
    clickable when it has one.
    """
    status = str(record.get("status") or "running")
    glyph = _AGENT_STATUS_GLYPHS.get(status, "●")
    name = str(record.get("task") or "sub-agent")
    row_id = str(record.get("id") or "")
    return InspectorSectionRow(
        row_id=row_id,
        primary_text=f"{glyph} {name}",
        secondary_text="",
        status=status,
        clickable=bool(row_id),
    )


class ConsoleAgentController:
    """Owns the Console agent runtime's screen-side state and behaviour.

    See this module's docstring for the map of what moved, what stayed, and
    why.
    """

    def __init__(
        self,
        screen: "ChatScreen",
        *,
        app_instance: Any,
        chat_store_accessor: Callable[[], Any],
        provider_gateway_accessor: Callable[[], Any],
        native_tool_calls_enabled_accessor: Callable[[], Any],
        current_rail_conversation_id: Callable[[], str | None],
        current_rail_state_accessor: Callable[[], "ConsoleRailState"],
        chat_controller_accessor: Callable[[], Any],
        sync_native_console_chat_ui_accessor: Callable[[], Any],
    ) -> None:
        """Wire the agent controller. Built in `Console_Modules/wiring.py`.

        The binding rule is stated canonically in
        `ConsoleDictationController.__init__`'s docstring. As it applies
        here:

        1. **Framework services** (`run_worker`, `push_screen`) are live-read
           from the screen via `@property` below, never snapshotted.
        2. **`app_instance`** is the one justified snapshot: it never
           changes identity over this controller's life, and
           `_ensure_console_agent_bridge` only ever `getattr`s off it.
        3. Every **app-level dependency** is a named keyword-only callable,
           wired at the call site as a late-binding lambda -- never a bound
           method, which would freeze the current target and stop observing
           a later `monkeypatch.setattr`. The pre-existing suite patches
           `_current_console_rail_conversation_id` on `screen._character`
           and `_current_console_rail_state` on the screen (both in
           `Tests/UI/test_console_agent_rail.py`), so this is load-bearing,
           not ceremony.

        Args:
            screen: The Console screen this controller belongs to. Held for
                the framework-service properties only; no DOM is reached
                through it.
            app_instance: The `TldwCli` app. Snapshot (see above).
            chat_store_accessor: `ChatScreen._ensure_console_chat_store`,
                called to hand the bridge its store.
            provider_gateway_accessor:
                `ChatScreen._ensure_console_provider_gateway`, same.
            native_tool_calls_enabled_accessor: Returns the screen's
                `_console_native_tool_calls_enabled` **method object** --
                the bridge takes it as a callable and calls it later, so
                this accessor must not call it here.
            current_rail_conversation_id: Returns the conversation id the
                rail is currently showing, or `None`.
            current_rail_state_accessor: `ChatScreen._current_console_rail_
                state`, for the Agent section's own open/collapsed state.
            chat_controller_accessor: Reads the screen's
                `_console_chat_controller` attribute (`getattr`-safe: the
                fleet line is reachable on a screen that has never built
                one).
            sync_native_console_chat_ui_accessor: Returns the screen's
                `_sync_native_console_chat_ui` **method object**, handed
                straight to `run_worker` by the drill-in toggle.
        """
        self._screen = screen
        self.app_instance = app_instance
        self._chat_store_accessor = chat_store_accessor
        self._provider_gateway_accessor = provider_gateway_accessor
        self._native_tool_calls_enabled_accessor = native_tool_calls_enabled_accessor
        self._current_rail_conversation_id = current_rail_conversation_id
        self._current_rail_state_accessor = current_rail_state_accessor
        self._chat_controller_accessor = chat_controller_accessor
        self._sync_native_console_chat_ui_accessor = (
            sync_native_console_chat_ui_accessor
        )

        # -- Moved state (was `ChatScreen.__init__`) ------------------------
        #: The lazily-built `ConsoleAgentBridge` is a PROPERTY over the
        #: app-owned `ConsoleRuntime` (task-15860 lifetime landing) -- it has
        #: no `__init__` slot, because a fresh screen's `None` would shadow
        #: the surviving runtime's live bridge. Still read-write through the
        #: screen's proxy of the same name
        #: (`Tests/UI/test_console_agent_rail.py` replaces it on the screen
        #: 10 times); the write now lands on the runtime.
        #: The sub-agent run currently drilled into, and the conversation
        #: that drill-in is scoped to. Both proxied read-write on the screen.
        self._console_agent_drilldown_run_id: str | None = None
        self._console_agent_drilldown_conversation_id: str | None = None
        #: `_console_agent_full_log_available`'s probe cache -- the resolved
        #: target run id and the answer for it. No reader outside this
        #: cluster, so no screen proxy.
        self._console_agent_full_log_cache_run_id: str | None = None
        self._console_agent_full_log_cache_available: bool = False
        #: The batched `[N Sub-Agents]` badge-count cache and its two
        #: invalidation keys. Also cluster-private.
        self._console_subagent_counts_cache: Dict[str, int] = {}
        self._console_subagent_counts_cache_row_ids: frozenset = frozenset()
        self._console_subagent_counts_cache_at: float = 0.0
        #: TASK-915: sticky suppression of the fleet force-open for the rest
        #: of THIS busy window. Written by the screen's own `_toggle_console_
        #: rail_section`, so it is proxied read-write there.
        self._agent_section_user_dismissed_while_busy = False

    # -- Framework services (live-read via `@property`) --------------------

    @property
    def run_worker(self) -> Any:
        """`Screen.run_worker`, bound. See `__init__`'s docstring for why
        this is a property rather than a value snapshotted once."""
        return self._screen.run_worker

    @property
    def push_screen(self) -> Any:
        """`Screen.app.push_screen`, bound. See `__init__`'s docstring."""
        return self._screen.app.push_screen

    # -- Named constructor dependencies -------------------------------------
    #
    # Each property below is a thin wrapper around a stored callable, kept
    # under the SAME name the original `ChatScreen` method/attribute used --
    # which is what lets the method bodies further down be byte-for-byte
    # copies of the pre-extraction source.

    @property
    def _ensure_console_chat_store(self) -> Any:
        """`ChatScreen._ensure_console_chat_store`, reached by name."""
        return self._chat_store_accessor

    @property
    def _ensure_console_provider_gateway(self) -> Any:
        """`ChatScreen._ensure_console_provider_gateway`, reached by name."""
        return self._provider_gateway_accessor

    @property
    def _console_native_tool_calls_enabled(self) -> Any:
        """The screen's `_console_native_tool_calls_enabled` method object.

        Handed to `ConsoleAgentBridge` as a callable it stores and calls
        later, so this returns the method rather than its result -- the
        same "bare-attribute READ, not a call" shape
        `ConsolePromptsController` documents for
        `open_console_provider_recovery_accessor`.
        """
        return self._native_tool_calls_enabled_accessor()

    @property
    def _current_console_rail_conversation_id(self) -> Any:
        """The character controller's rail-conversation accessor, by name."""
        return self._current_rail_conversation_id

    @property
    def _current_console_rail_state(self) -> Any:
        """`ChatScreen._current_console_rail_state`, by name."""
        return self._current_rail_state_accessor

    @property
    def _console_chat_controller(self) -> Any:
        """The screen's current `ConsoleChatController`, or `None`."""
        return self._chat_controller_accessor()

    @property
    def _sync_native_console_chat_ui(self) -> Any:
        """The screen's `_sync_native_console_chat_ui` method object.

        Handed straight to `run_worker` (not called here) by the drill-in
        toggle, exactly as the pre-move body did.
        """
        return self._sync_native_console_chat_ui_accessor()

    # -- Moved methods -------------------------------------------------------

    def _console_runtime(self) -> Any:
        """Return the app-owned `ConsoleRuntime` (task-15860).

        Resolved through the SCREEN's own memoised accessor whenever it has
        one, so this controller and its screen can never end up holding two
        different runtimes.
        """
        screen = self._screen
        resolver = getattr(screen, "_console_runtime", None)
        if callable(resolver):
            return resolver()
        from tldw_chatbook.Chat.console_runtime import ensure_console_runtime

        return ensure_console_runtime(self.app_instance, view=screen)

    @property
    def _console_agent_bridge(self) -> Any:
        """The runtime's Console agent bridge, or `None`."""
        return self._console_runtime().agent_bridge

    @_console_agent_bridge.setter
    def _console_agent_bridge(self, value: Any) -> None:
        self._console_runtime().set_agent_bridge(value)

    def _ensure_console_agent_bridge(self) -> Any:
        """Return the native Console agent bridge, creating it lazily.

        Returns ``None`` (no agent runtime) when there is no durable
        ChaChaNotes DB to key the sibling ``AgentRunsDB`` file off of (e.g. an
        in-memory test harness) -- callers use the provider-direct Console
        stream in that case regardless of the config gate.

        task-15860 Task 1 (pure ownership move): the bridge, its
        ``AgentRunsDB`` and the ``register_fleet_attention`` fan-out
        registration that must sit next to construction are now built by
        the app-owned ``ConsoleRuntime`` (``Chat/console_runtime.py``).
        This method keeps its name -- three test files replace it on the
        screen instance, see this module's docstring -- and its caching.
        The store and gateway go over as CALLABLES so the runtime can keep
        the original ordering: the durable-DB probe still runs before
        either of them is touched.
        """
        if self._console_agent_bridge is not None:
            return self._console_agent_bridge
        self._console_runtime().ensure_agent_bridge(
            store_factory=self._ensure_console_chat_store,
            provider_gateway_factory=self._ensure_console_provider_gateway,
            skills_service=getattr(self.app_instance, "skills_scope_service", None),
            native_tools_enabled_factory=(
                lambda: self._console_native_tool_calls_enabled
            ),
        )
        return self._console_agent_bridge

    def console_turn_activity(self) -> str:
        """The viewed session's live turn-activity line, or ``""``.

        Read once per 0.2s Console poll tick and handed to
        ``ConsoleTranscript.apply_turn_activity``. **No new timer**: during
        an agent turn the viewed run's status is in
        ``CONSOLE_ACTIVE_RUN_STATUSES``, which is exactly the condition
        ``_start_console_transcript_sync_timer`` keeps its 0.2s tick alive
        for, so the line already repaints for free while a turn is in
        flight -- and never repaints when nothing is (task-15664 AC#2).

        That same status check is the FIRST gate here, deliberately. The
        bridge's published snapshot is per-conversation and only a terminal
        publish clears it, so a run that died without one would otherwise
        leave ``status="running"`` behind and let this line tick forever on
        an idle transcript. ``run_state`` is the read-only facade for the
        VIEWED session only, so it is also what keeps another session's
        in-flight turn from writing onto this one's row.

        Every bridge attribute is reached through ``getattr`` for the same
        reason the rail's own reads are: several suites drive this cluster
        with a bare bridge double that implements only part of the surface.

        Returns:
            The rendered line, or ``""`` when no live turn owns this view.
        """
        # Imported in-body for the cycle this module's docstring documents:
        # `chat_screen` imports `Console_Modules.wiring`, which imports this
        # module, and the constant is defined further down that file.
        from ..Screens.chat_screen import CONSOLE_ACTIVE_RUN_STATUSES

        controller = self._console_chat_controller
        run_state = getattr(controller, "run_state", None) if controller else None
        if run_state is None or run_state.status not in CONSOLE_ACTIVE_RUN_STATUSES:
            return ""
        bridge = self._console_agent_bridge
        if bridge is None:
            return ""
        conversation_id = self._current_console_rail_conversation_id() or ""
        read_snapshot = getattr(bridge, "live_snapshot", None)
        if read_snapshot is None:
            return ""
        return console_turn_activity_text(
            read_snapshot(conversation_id), now=time.monotonic()
        )

    def _console_agent_section_lines(self) -> tuple[str, str, str]:
        """Return the Agent rail's (status, steps, sub-agents) line text.

        Reads the live in-memory run snapshot (or, when drilled into one
        sub-agent, that run's durable record) via the Console agent bridge --
        the same bridge whose ``AgentRunsDB`` backs resume re-derivation, so
        this always reflects the latest known state without any extra event
        plumbing (the 0.2s Console poll re-calls this on every tick).

        Finding B: none of this text is escaped -- every string returned
        here is rendered into a ``markup=False`` Static (see the compose
        block below), so escaping would be a second guard stacked on top
        of ``markup=False`` and would render literal backslashes (e.g.
        ``fetch [docs]`` -> ``fetch \\[docs]``). Contrast with the
        conversation-browser badge label (``format_console_conversation_
        row_label``), which renders through ``Text.from_markup`` and must
        stay escaped.

        Finding C: a drill-in is scoped to the conversation active when
        the user drilled in. Every call here re-checks that scope --
        catching any switch path that doesn't itself clear the drill-down
        -- and falls back to the overview on a mismatch rather than show
        a foreign conversation's sub-agent detail.

        Gate Finding 2 (agent-runtime live gate): the top-level overview
        line used to read only ``bridge.live_snapshot`` -- an in-memory,
        per-process cache that starts empty every new bridge instance, so
        it showed "Agent: idle" for a resumed conversation right after an
        app restart even though the drill-in and the conversation-row
        badge both correctly re-derived from ``AgentRunsDB``. An idle live
        snapshot now falls back to ``bridge.historical_snapshot`` (cached
        by the bridge itself, so this does not add a DB hit per 0.2s poll
        tick) -- a live/in-process run always reports non-"idle" and keeps
        precedence over the fallback.
        """
        bridge = self._ensure_console_agent_bridge()
        conversation_id = self._current_console_rail_conversation_id() or ""
        if bridge is None:
            return ("Agent: unavailable", "", "")
        if conversation_id != self._console_agent_drilldown_conversation_id:
            # The active conversation/session changed since the drill-in
            # (tab switch, Ctrl+K switcher, saved-conversation resume,
            # workspace switch, ...) -- this self-heals even for a switch
            # path that doesn't explicitly clear the drill-down itself.
            self._console_agent_drilldown_run_id = None
            self._console_agent_drilldown_conversation_id = conversation_id
        drill = self._console_agent_drilldown_run_id
        if drill:
            record = bridge.subagent_run(drill)
            if record is not None and record.get("conversation_id") == conversation_id:
                # Finding A (review round 2): this used to slice the raw
                # `summary`/`result` field to a hardcoded 80 characters,
                # bypassing `_summarize_persisted_step` entirely -- so a
                # drilled-in sub-agent's step text neither respected the
                # user-configurable display cap (TASK-870) nor got the
                # word-boundary truncation affordance every other render
                # path shares. Route through the same helper the top-level
                # overview's persisted/resumed steps already use (see
                # `ConsoleAgentBridge._derive_historical_snapshot`).
                from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

                steps = "\n".join(
                    f"{s.get('kind')}: "
                    f"{ConsoleAgentBridge._summarize_persisted_step(s)}"
                    for s in record.get("steps", [])
                )
                if not steps:
                    # PR3a-1 Task 6b (audit F1): a run's steps reach
                    # `AgentRunsDB` in ONE write, when the run ends
                    # (`AgentService._persist`), so this record is
                    # step-less for the whole time a child is actually
                    # working -- and a fleet child now keeps working after
                    # the turn that spawned it returned, so that is no
                    # longer a sub-second window. The bridge's own per-run
                    # live slot is the only source for it; read it here
                    # rather than render an empty drill-in for a child
                    # visibly listed as running. Only when the record has
                    # nothing: the DB row is COMPLETE once written, while
                    # the live slot keeps only the last few steps.
                    # `getattr` tolerates the bare bridge doubles several
                    # tests in this module use.
                    live_run_snapshot = getattr(bridge, "live_run_snapshot", None)
                    live_run = (
                        live_run_snapshot(conversation_id, drill)
                        if live_run_snapshot is not None
                        else None
                    )
                    if live_run is not None:
                        steps = "\n".join(f"{s.kind}: {s.text}" for s in live_run.steps)
                # PR3b Task 4: a resumed sub-agent (send_to_agent to a
                # finished child starts a NEW run seeded with its retained
                # transcript) carries its lineage in the header. The run
                # row's `resumed_from_run_id` flows here for free (the
                # bridge reads SELECT * row dicts); `.get(...) or ""`
                # keeps pre-v11 rows (key absent or NULL) byte-identical.
                resumed_from = str(record.get("resumed_from_run_id") or "")
                header = f"Sub-agent · {record.get('status')}"
                if resumed_from:
                    header += f" · resumed from {resumed_from}"
                return (
                    f"{header} (Back)",
                    steps,
                    str(record.get("task") or ""),
                )
            # The drilled-into run vanished, or (defensive re-check) its
            # recorded conversation_id no longer matches the one now
            # active -- fall back to the live snapshot instead of showing
            # a stale/foreign drill-in view.
            self._console_agent_drilldown_run_id = None
        snapshot = bridge.live_snapshot(conversation_id)
        if snapshot.status == "idle":
            # Finding 2 (Plan-B agent-runtime gate): this bridge instance
            # has never run this conversation in-process -- most likely a
            # resumed conversation right after an app restart, since
            # ``live_snapshot`` is an in-memory-only, per-process cache
            # that starts empty every new instance. Fall back to
            # AgentRunsDB so the summary reflects history immediately
            # instead of showing "Agent: idle" until the next live run.
            # A live run already in progress/finished in this process
            # always reports a non-"idle" status above and keeps
            # precedence -- this fallback is only ever consulted when
            # there is nothing live to show. ``getattr`` tolerates a bare
            # test double that only implements ``live_snapshot``.
            historical = getattr(bridge, "historical_snapshot", None)
            if historical is not None:
                snapshot = historical(conversation_id)
        status = f"Agent: {snapshot.status}"
        if snapshot.status == "running":
            status = f"Agent: running · step {snapshot.step}"
        # Finding A (review round 2): `s.text` is already truncated to the
        # user-configurable cap by whichever bridge helper built this
        # snapshot (`_summarize` for a live step, `_summarize_persisted_
        # step` for a resumed/historical one) -- re-slicing to a hardcoded
        # 80 here silently overrode any configured value above 80 with no
        # visible effect, defeating the whole point of that setting.
        steps = "\n".join(f"· {s.text}" for s in snapshot.steps)
        # PR2b Task 4: the hard `[:60]` slice this used to apply is retired
        # -- the joined string this builds is no longer painted into any
        # DOM Static (the mounted rail now renders per-row widgets via
        # `_console_agent_fleet_section_state`, which owns its own CSS
        # ellipsis truncation instead of a fixed character cap). This
        # method's return shape stays a 3-tuple for its existing direct
        # callers/tests (`Tests/UI/test_console_agent_rail.py`/
        # `test_console_agent_controller.py`), just un-truncated.
        subagents = "\n".join(
            f"{_AGENT_STATUS_GLYPHS.get(s.status, '●')} {s.text}"
            for s in snapshot.subagents
        )
        return (status, steps, subagents)

    def _console_agent_fleet_summary_line(self) -> str:
        """Return the Agent rail's fleet summary line (parallel-agents spec §6).

        Sourced from ``ConsoleChatController.fleet_summary_counts`` (other
        running / other pending-approval sessions, relative to the active
        one). Copy is VERBATIM per spec §6 -- no singular/plural grammar
        handling, so ``"1 other agents running, ..."`` is intentional, not a
        bug. Returns ``""`` when both counts are zero; the caller hides the
        fleet Static in that case (absent, not present-but-blank) so it
        never crowds the rail with an empty line.
        """
        controller = getattr(self, "_console_chat_controller", None)
        if controller is None:
            return ""
        running, pending = controller.fleet_summary_counts()
        if running + pending <= 0:
            return ""
        return f"{running} other agents running, {pending} waiting for approval."

    def _console_agent_full_log_run_id(self) -> str | None:
        """Return the run id the "View full log" affordance should target.

        TASK-870: mirrors ``_console_agent_section_lines``'s own
        drill-vs-overview precedence -- the drilled-into sub-agent run
        (when drilled in and still valid for the active conversation, the
        same check that method uses), else the conversation's latest
        primary run, which is what the top-level overview is summarizing.

        Returns:
            The relevant run id, or ``None`` when there is nothing to
            target -- no bridge, no active conversation, a stale drill-in
            left over from a conversation switch, or a conversation that
            has never run an agent. Callers must still confirm
            ``ConsoleAgentBridge.run_log_available`` before showing the
            affordance for whatever id this returns -- a valid run id does
            not imply a log was ever written for it.
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return None
        conversation_id = self._current_console_rail_conversation_id() or ""
        if not conversation_id:
            return None
        drill = self._console_agent_drilldown_run_id
        if drill:
            record = bridge.subagent_run(drill)
            if record is not None and record.get("conversation_id") == conversation_id:
                return drill
            return None
        # getattr tolerates a bare test double that only implements the
        # older bridge surface (subagent_run/subagent_runs/live_snapshot) --
        # same idiom _console_agent_section_lines already uses for
        # historical_snapshot, immediately below this method in this file.
        latest_primary_run_id = getattr(bridge, "latest_primary_run_id", None)
        if latest_primary_run_id is None:
            return None
        return latest_primary_run_id(conversation_id)

    def _console_agent_full_log_available(self, *, allow_probe: bool = True) -> bool:
        """Whether the "View full log" affordance should be shown right now.

        TASK-870 (AC#6/#7): ``True`` only when ``_console_agent_full_log_
        run_id`` resolves to a run AND that run actually has an on-disk log
        -- absent (button hidden) for every other case, including a bridge
        or filesystem lookup that raises, so a resolution failure can never
        surface as a dangling or erroring button.

        Finding D (review round 2): the underlying check costs a SQLite
        lookup (``_console_agent_full_log_run_id``, to resolve the target
        run id) plus a filesystem probe (``bridge.run_log_available``, to
        confirm a log directory/segment exists -- for a drilled-in
        sub-agent this can also mean parsing its primary's whole log, see
        finding B) -- paying that unconditionally on every 0.2s rail tick
        is real, avoidable I/O for a value that is overwhelmingly the same
        tick to tick. The filesystem/DB probe result is cached keyed by
        the resolved run id and only redone when that id changes; when
        ``allow_probe`` is ``False`` (the periodic sync passes this while
        the Agent section is collapsed -- see
        ``_sync_console_agent_section``), this returns the last cached
        answer WITHOUT even resolving the current run id, so a collapsed
        section's steady-state tick touches neither disk nor the DB.

        Args:
            allow_probe: Whether a cache miss may fall through to the
                SQLite/filesystem lookup. Callers that need a fresh,
                authoritative answer (the one-shot compose-time render,
                and the "open the viewer" press-time re-check) should
                leave this ``True``; the periodic rail sync passes
                ``section_open`` here.

        Returns:
            Whether the affordance should be visible for the current
            target run -- possibly a stale cached value when
            ``allow_probe`` is ``False`` and the target has since changed
            unobserved (self-corrects the moment the section reopens).
        """
        if not allow_probe:
            return self._console_agent_full_log_cache_available
        run_id = self._console_agent_full_log_run_id()
        if run_id == self._console_agent_full_log_cache_run_id:
            return self._console_agent_full_log_cache_available
        if not run_id:
            available = False
        else:
            bridge = self._ensure_console_agent_bridge()
            if bridge is None:
                available = False
            else:
                try:
                    available = bool(bridge.run_log_available(run_id))
                except Exception:
                    logger.opt(exception=True).warning(
                        "console agent rail: run_log_available check failed for "
                        f"run_id={run_id}; hiding the View full log affordance"
                    )
                    available = False
        self._console_agent_full_log_cache_run_id = run_id
        self._console_agent_full_log_cache_available = available
        return available

    def _open_console_agent_run_log_viewer(self) -> None:
        """Kick off loading the full run log for whatever "View full log" targets.

        TASK-870 (AC#6): re-resolves the target run id at press time
        (rather than trusting a value cached from the last 0.2s sync) so a
        drill-in change between sync ticks can never open the wrong run's
        log. No-ops quietly if there is no current target.

        Finding C (review round 2): the actual filesystem read + record
        parse + formatting now happens off the UI thread (see
        ``_load_console_agent_run_log``) -- a run's segments can total
        many megabytes (4MB per segment, no cap on segment count), and
        doing that synchronously on the Textual event loop could freeze
        the whole app for the duration of the read.
        """
        run_id = self._console_agent_full_log_run_id()
        if not run_id:
            return
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return
        # Was `self._load_console_agent_run_log(bridge, run_id)` while
        # that method carried `@work(thread=True)`. Same worker, same
        # group ("default") and name -- see the loader's own docstring.
        self.run_worker(
            partial(self._load_console_agent_run_log, bridge, run_id),
            thread=True,
            group="default",
            name="_load_console_agent_run_log",
        )

    def _load_console_agent_run_log(self, bridge: Any, run_id: str) -> None:
        """Load, filter, and format one run's full log off the UI thread.

        Finding C: the worker half of ``_open_console_agent_run_log_
        viewer`` -- everything here is filesystem/CPU work (no widget
        access), so it is safe to run in a real thread. The modal is only
        ever pushed back on the UI thread, via ``call_from_thread``.

        Wave-4 task 3: this carried ``@work(thread=True)`` before the move.
        Textual's decorator asserts its target is a ``DOMNode`` and a
        controller is not one, so the caller now dispatches it through
        ``run_worker`` with the same thread/group/name ``@work`` supplied.

        Args:
            bridge: The already-resolved Console agent bridge (resolved on
                the UI thread by the caller -- lazy bridge construction
                touches ``self.app_instance``/config and should not run
                off-thread).
            run_id: The run id to load, as resolved by the caller at press
                time.
        """
        try:
            if not bridge.run_log_available(run_id):
                return
            log_text = bridge.load_run_log_text(run_id)
        except Exception:
            logger.opt(exception=True).warning(
                f"console agent rail: failed to load run log for run_id={run_id}"
            )
            return
        if not log_text:
            return
        self._screen.app.call_from_thread(
            self._show_console_agent_run_log_modal, run_id, log_text
        )

    def _show_console_agent_run_log_modal(self, run_id: str, log_text: str) -> None:
        """Push the full-log modal. UI-thread only -- see ``_load_console_agent_run_log``.

        Args:
            run_id: The run id the loaded text belongs to.
            log_text: The fully rendered, untruncated log text.
        """
        self.push_screen(ConsoleRunLogModal(run_id=run_id, log_text=log_text))

    def _console_agent_section_payload(
        self,
    ) -> tuple[
        str,
        str,
        ConsoleInspectorSectionState,
        str,
        bool,
        bool,
        bool,
        ConsoleAgentSteeringState,
        bool,
    ]:
        """Derive everything the mounted Agent rail should be showing.

        TASK-251: equality-guarded against the last successfully-applied
        payload -- the 0.2s tick called this unconditionally, forcing three
        ``Static.update()`` calls plus a style write per tick even when
        nothing agent-related had changed.

        Fix round 2 (parallel-agents spec §6 live-smoke finding): also
        tracks and applies the Agent section's own open/collapsed state
        (header chevron + body ``display``). Compose-time already applies
        ``_apply_fleet_agent_section_auto_open`` once at mount, but that is
        a one-shot snapshot -- a background session's run starting or
        ending *after* mount (the overwhelmingly common case) must reopen
        or release the section on this same periodic sync, or the fleet
        line would only ever be reachable for whichever fleet state
        happened to exist at the moment the screen was first composed.

        TASK-870: also tracks the "View full log" affordance's visibility
        (``_console_agent_full_log_available``) -- present only while a run
        log actually exists for whatever run the section is currently
        showing (AC#6/#7).

        Wave-4 task 3: this is the non-DOM half of the screen's
        ``_sync_console_agent_section``, which kept the equality guard
        itself (``_console_agent_section_last``, compared against what this
        returns) and the nine ``query_one`` writes -- controllers own no
        DOM. Every paragraph above still describes what is derived here;
        only the "applies"/"tracks" half of each is now the screen's. The
        ordering constraint the third paragraph records -- probe the
        full-log affordance only once ``section_open`` is known -- lives
        here too, which is why the derivation moved as one piece rather
        than as seven accessors the screen would have had to sequence.

        PR2b Task 4: the third element used to be the joined, ``[:60]``-
        sliced sub-agents STRING that painted straight into a single
        Static. It is now a ``ConsoleInspectorSectionState`` (rows +
        header summary) for the ``ConsoleInspectorSection`` component that
        replaced that Static -- derived independently, by
        ``_console_agent_fleet_section_state``, from the SAME live/
        historical bridge sources, not from ``_console_agent_section_
        lines``'s own (still-3-tuple, still separately tested) text. Both
        this method's callers already compared the whole payload with
        ``==``; ``ConsoleInspectorSectionState`` is a frozen dataclass of
        immutable fields, so the equality guard keeps working unchanged.

        Returns:
            ``(status_line, steps_text, fleet_section_state, fleet_line,
            back_visible, section_open, full_log_visible, steering_state,
            cancel_all_visible)`` -- the exact tuple the screen compares
            against its last applied payload. ``steering_state`` (PR3b
            Task 3) is the drill-in steering bar's
            ``ConsoleAgentSteeringState`` -- a frozen dataclass, so the
            payload's ``==`` equality guard keeps working unchanged, the
            same argument PR2b Task 4 recorded for
            ``ConsoleInspectorSectionState``. ``cancel_all_visible``
            (PR3b Task 5) is the "Cancel all agents" affordance's
            visibility -- extending THIS payload rather than adding a
            second equality guard, per Task 3's landing note.
        """
        status_line, steps_text, _subagents_text = self._console_agent_section_lines()
        fleet_section_state = self._console_agent_fleet_section_state()
        fleet_line = self._console_agent_fleet_summary_line()
        back_visible = bool(self._console_agent_drilldown_run_id)
        try:
            section_open = self._current_console_rail_state().agent_open
        except (AttributeError, NoActiveAppError):
            # A bare/unmounted screen (several tests construct
            # `ChatScreen(app)` directly with no active Textual app
            # context -- e.g. `test_console_provider_selection_carries_
            # active_session_system_prompt`) has no real rail width to
            # derive responsive state from: `_console_rail_available_
            # columns` reads `self.size`, which raises here rather than
            # returning `None` (`Screen.size` needs `self.app`). Same
            # guard idiom `_provider_readiness_app_config` already uses
            # for this exact failure mode. Fall back to the persisted
            # default (collapsed) -- the Statics-only updates below are
            # still worth applying even when the section's own open state
            # cannot be derived.
            section_open = False
        # Finding D: never probe disk/DB for the full-log affordance while
        # the section is collapsed -- `section_open` must be known first.
        full_log_visible = self._console_agent_full_log_available(
            allow_probe=section_open
        )
        return (
            status_line,
            steps_text,
            fleet_section_state,
            fleet_line,
            back_visible,
            section_open,
            full_log_visible,
            # Derived AFTER `_console_agent_section_lines` above, which
            # self-heals a stale drill-in scope first (Finding C) -- though
            # the derivation re-checks the scope itself defensively.
            self._console_agent_steering_state(),
            self._console_agent_cancel_all_visible(),
        )

    # -- PR2b Task 4: the fleet mini-section (states 1/2, spec §7) ---------

    def _console_agent_fleet_rows(self) -> tuple[InspectorSectionRow, ...]:
        """Build one row per sub-agent for the ``ConsoleInspectorSection``.

        Three sources, tried in order, matching the same live-over-
        historical precedence ``_console_agent_section_lines`` already
        uses:

        1. ``bridge.fleet_snapshot(conversation_id)`` (PR2b Task 1): the
           REAL, live ``FleetCoordinator`` handles for a run in flight in
           THIS process -- real per-child status, plus ``started_at``/
           ``finished_at`` (monotonic floats), which is the only source
           that can render an "elapsed" segment.
        2. ``bridge.historical_snapshot(conversation_id).subagents``: the
           durable, DB-re-derived fallback for a resumed conversation (no
           live coordinator this process has ever seen) -- cached by the
           bridge itself, so this costs no extra DB round trip beyond what
           ``_console_agent_section_lines`` already pays each tick.
        3. ``bridge.subagent_runs(conversation_id)``: a last-resort raw-
           record fallback for a bridge stub that implements only the
           oldest surface (no ``historical_snapshot``) -- several test
           doubles in ``Tests/UI/test_console_agent_rail.py`` predate
           Task 1-3 and only carry this method.

        ``getattr`` guards every optional method the same way
        ``_console_agent_full_log_run_id`` already does for
        ``latest_primary_run_id`` -- a bare test double implementing only
        part of the bridge surface must degrade to "no rows", never raise.

        Returns:
            Rows in source order (fleet-reservation order, or
            ``AgentRunsDB.list_runs``' newest-first order for the two
            fallback tiers) -- empty when there is no bridge, no active
            conversation, or no sub-agent has ever run for it.
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return ()
        conversation_id = self._current_console_rail_conversation_id() or ""
        if not conversation_id:
            return ()
        fleet_snapshot = getattr(bridge, "fleet_snapshot", None)
        handles = fleet_snapshot(conversation_id) if fleet_snapshot is not None else []
        if handles:
            now = time.monotonic()
            return tuple(_fleet_row_from_handle(handle, now=now) for handle in handles)
        historical_snapshot = getattr(bridge, "historical_snapshot", None)
        if historical_snapshot is not None:
            return tuple(
                _fleet_row_from_summary(summary, index)
                for index, summary in enumerate(
                    historical_snapshot(conversation_id).subagents
                )
            )
        subagent_runs = getattr(bridge, "subagent_runs", None)
        if subagent_runs is None:
            return ()
        return tuple(
            _fleet_row_from_record(record) for record in subagent_runs(conversation_id)
        )

    def _console_agent_fleet_token_total(self) -> int:
        """Sum the active conversation's LIVE fleet's measured token spend.

        PR2b Task 5 (cost rollup): the aggregate the Console cost ticker
        reaches for -- feeds ``build_cost_snapshot``'s ``fleet_tokens``
        keyword (see ``chat_screen.py``'s ``_build_console_cost_state``).
        Sums ``FleetHandle.total_tokens`` directly off
        ``bridge.fleet_snapshot(conversation_id)`` -- the SAME live source
        ``_console_agent_fleet_rows`` reads for its live tier, so the
        aggregate here and each row's own token segment can never disagree.
        A still-running handle's ``total_tokens`` is 0 (see
        ``_fleet_row_from_handle``'s docstring), so it naturally contributes
        nothing until it finishes -- no separate "only count terminal rows"
        filter is needed.

        Returns 0 -- never raises -- when there is no bridge, no active
        conversation, or (the common historical/resumed case) no LIVE
        fleet for it: this deliberately does NOT fall back to the
        historical/DB-derived tiers `_console_agent_fleet_rows` also reads,
        since per-child spend is not persisted there (see `FleetHandle.
        total_tokens`'s docstring) -- there is nothing honest to sum for a
        resumed conversation this process has never run.
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return 0
        conversation_id = self._current_console_rail_conversation_id() or ""
        if not conversation_id:
            return 0
        fleet_snapshot = getattr(bridge, "fleet_snapshot", None)
        if fleet_snapshot is None:
            return 0
        return sum(handle.total_tokens for handle in fleet_snapshot(conversation_id))

    def _cancel_console_agent_fleet_row(self, row_id: str) -> bool:
        """Cooperatively cancel a LIVE fleet row's child (PR2b Task 5).

        ``row_id`` is the row's structural identity as built by
        ``_fleet_row_from_handle`` -- the ``FleetCoordinator`` handle id --
        so this reaches ``ConsoleAgentBridge.cancel_subagent`` with NO
        resolution step, unlike drill-in's ``_console_agent_drilldown_
        target_run_id`` (which must also accept a historical row's run id).
        A historical/fallback row is never cancellable (``_fleet_row_from_
        summary``/``_fleet_row_from_record`` leave ``cancellable`` at its
        ``False`` default), so in practice ``row_id`` reaching this method
        is always a live handle id.

        Routes through ``ConsoleAgentBridge.cancel_subagent`` ->
        ``AgentService.cancel_subagent`` -> the SAME ``_cancel_fleet_
        handles`` cooperative-cancel + approval-revoke path
        ``_settle_fleet`` already uses at end of turn (PR 2a's guarantee
        that cancelling a child revokes its pending approval cards) -- no
        second cancellation mechanism.

        Returns:
            Whether the handle was live and the cancel request was
            actually issued -- ``False`` for no bridge, no active
            conversation, or an unknown/already-terminal handle.
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return False
        conversation_id = self._current_console_rail_conversation_id() or ""
        if not conversation_id or not row_id:
            return False
        cancel_subagent = getattr(bridge, "cancel_subagent", None)
        if cancel_subagent is None:
            return False
        return bool(cancel_subagent(conversation_id, row_id))

    def _console_agent_cancel_all_visible(self) -> bool:
        """Whether the "Cancel all agents" affordance should be offered.

        PR3b Task 5: visible exactly while the conversation has at least
        one LIVE child -- read from ``bridge.fleet_snapshot``, the SAME
        live source the fleet rows and the steering-bar visibility read,
        so the three surfaces can never disagree about whether live work
        exists. A fleet of finished children (rows still on screen) hides
        it: offering a kill switch for work that already ended would be
        a lie.

        Returns:
            ``True`` only with a bridge, an active conversation, and a
            live (non-terminal) handle in its fleet snapshot; ``False``
            -- never raises -- otherwise.
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return False
        conversation_id = self._current_console_rail_conversation_id() or ""
        if not conversation_id:
            return False
        fleet_snapshot = getattr(bridge, "fleet_snapshot", None)
        if fleet_snapshot is None:
            return False
        return any(
            handle.status not in TERMINAL_RUN_STATUSES
            for handle in fleet_snapshot(conversation_id)
        )

    def _cancel_all_console_agents(self) -> int:
        """Cancel every live child of the active conversation (PR3b Task 5).

        The exact shape of ``_cancel_console_agent_fleet_row`` above, for
        the same reasons: ``getattr`` guards degrade a bare test double
        to "nothing cancelled", never a raise, and no enumeration happens
        here -- ``ConsoleAgentBridge.cancel_all_subagents`` owns the
        owner walk and delegates each handle to the existing per-handle
        revocation path (no second mechanism).

        Returns:
            The number of children actually cancelled -- 0 for no
            bridge, no active conversation, or an idle fleet.
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return 0
        conversation_id = self._current_console_rail_conversation_id() or ""
        if not conversation_id:
            return 0
        cancel_all = getattr(bridge, "cancel_all_subagents", None)
        if cancel_all is None:
            return 0
        return int(cancel_all(conversation_id))

    def _console_agent_steering_state(self) -> ConsoleAgentSteeringState:
        """Derive the drill-in steering bar's state (PR3b Task 3).

        Visible ONLY while drilled into a LIVE child (spec §1's owner
        pin: the panel watches/steers, never launches) -- never a
        finished/historical child, never the overview. The drill-in
        target (``_console_agent_drilldown_run_id``, a RUN id) is matched
        against the live fleet snapshot in both vocabularies, the same
        pair ``_console_agent_drilldown_target_run_id`` accepts; the
        state then carries the matched handle's HANDLE id as the steering
        target -- the mailbox's own key, the panel rows' identity, and
        the vocabulary ``ConsoleAgentBridge.steer_subagent`` resolves
        first.

        ``queued`` is the handle copy's ``queued_steering`` -- computed
        from the coordinator's mailbox at snapshot time (PR3b Task 1), so
        the bar's "steering queued (N)" line can never disagree with what
        the child's next drain would actually deliver.

        Returns:
            The bar's ``ConsoleAgentSteeringState``; the hidden state --
            never raises -- when not drilled in, the drill-in is scoped to
            another conversation, there is no bridge/fleet surface, the
            target cannot be found among live handles, or it has gone
            terminal.
        """
        drill = self._console_agent_drilldown_run_id
        if not drill:
            return STEERING_STATE_HIDDEN
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return STEERING_STATE_HIDDEN
        conversation_id = self._current_console_rail_conversation_id() or ""
        if (
            not conversation_id
            or conversation_id != self._console_agent_drilldown_conversation_id
        ):
            return STEERING_STATE_HIDDEN
        fleet_snapshot = getattr(bridge, "fleet_snapshot", None)
        if fleet_snapshot is None:
            return STEERING_STATE_HIDDEN
        for handle in fleet_snapshot(conversation_id):
            if drill not in (handle.run_id, handle.handle_id):
                continue
            if handle.status in TERMINAL_RUN_STATUSES:
                return STEERING_STATE_HIDDEN
            return ConsoleAgentSteeringState(
                visible=True,
                target_id=handle.handle_id,
                queued=int(getattr(handle, "queued_steering", 0) or 0),
            )
        return STEERING_STATE_HIDDEN

    def _steer_console_agent_drilldown_child(self, target_id: str, text: str) -> bool:
        """Route one validated steering submit to the bridge (PR3b Task 3).

        The exact shape of ``_cancel_console_agent_fleet_row`` above, for
        the same reasons: ``getattr`` guards degrade a bare test double to
        "no", never a raise, and no resolution happens here --
        ``ConsoleAgentBridge.steer_subagent`` owns both id vocabularies
        and the boundary validation (the bar already refused
        empty/oversize drafts with its own copy before posting).

        Args:
            target_id: The steering target the bar's submit message
                carried (the drilled-in child's handle id).
            text: The stripped steering text.

        Returns:
            Whether the entry was actually queued -- ``False`` for no
            bridge, no active conversation, an empty target, or a bridge
            refusal (unknown/terminal target, invalid text).
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return False
        conversation_id = self._current_console_rail_conversation_id() or ""
        if not conversation_id or not target_id:
            return False
        steer_subagent = getattr(bridge, "steer_subagent", None)
        if steer_subagent is None:
            return False
        return bool(steer_subagent(conversation_id, target_id, str(text or "")))

    def _console_agent_fleet_section_state(self) -> ConsoleInspectorSectionState:
        """Build the fleet mini-section's rows + header summary (states 1/2).

        Returns an empty state (no rows, no summary) while drilled into a
        specific sub-agent: the drilled-in status/steps Statics already
        show that one child's own detail (state 3, unchanged -- see
        ``_console_agent_section_lines``), so the aggregate fleet list
        would be redundant right beside it. ``_sync_console_agent_section``
        hides the mounted section entirely whenever this returns no rows,
        via the same visibility toggle the Back/View-full-log buttons
        already use.

        The header summary is a glyph cluster (one glyph per row, in row
        order) plus ``"N working, M done"`` (spec §7 state 1) -- "working"
        is ``status not in TERMINAL_RUN_STATUSES`` (i.e. still
        ``"running"``; every other status this codebase's fleet vocabulary
        uses -- ``done``/``error``/``stuck``/``cancelled`` -- is terminal
        per ``SubAgentSummary.status``'s own docstring), "done" is
        everything else. Returns an empty state when there are zero rows
        (never a hollow "0 working, 0 done" summary).
        """
        if self._console_agent_drilldown_run_id:
            return ConsoleInspectorSectionState(rows=(), summary="")
        rows = self._console_agent_fleet_rows()
        if not rows:
            return ConsoleInspectorSectionState(rows=(), summary="")
        working = sum(1 for row in rows if row.status not in TERMINAL_RUN_STATUSES)
        done = len(rows) - working
        glyphs = "".join(_AGENT_STATUS_GLYPHS.get(row.status, "●") for row in rows)
        summary = f"{glyphs} {working} working, {done} done"
        return ConsoleInspectorSectionState(rows=rows, summary=summary)

    def _console_agent_drilldown_target_run_id(self, row_id: str) -> str | None:
        """Resolve a clicked fleet row's id to the run id to drill into.

        ``row_id`` may be either a live ``FleetCoordinator`` handle id (the
        identity ``_fleet_row_from_handle`` stamps on a LIVE row -- stable
        across a handle's life even before ``attach_run`` gives it a real
        run id) or an ``AgentRunsDB`` run id directly (every historical/
        fallback row's identity). Both are tried, matching whichever
        source actually built the row -- the caller does not have to know
        which one that was.

        Returns:
            The run id to drill into, or ``None`` when ``row_id`` cannot be
            resolved to a run belonging to the ACTIVE conversation (a stale
            click after the fleet already cleared, a handle whose run
            hasn't attached yet, or a foreign/unknown id).
        """
        if not row_id:
            return None
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return None
        conversation_id = self._current_console_rail_conversation_id() or ""
        fleet_snapshot = getattr(bridge, "fleet_snapshot", None)
        if fleet_snapshot is not None:
            for handle in fleet_snapshot(conversation_id):
                if handle.handle_id == row_id:
                    return handle.run_id or None
        record = bridge.subagent_run(row_id)
        if record is not None and record.get("conversation_id") == conversation_id:
            return row_id
        return None

    def _drill_into_console_agent_subagent(self, row_id: str) -> None:
        """Drill into ONE specific sub-agent row's transcript (TASK-4).

        Replaces the old cycling toggle (``_toggle_console_agent_drilldown_
        from_subagents_click``, which stepped through every sub-agent run
        one click at a time, newest first, then back to the overview): the
        row the user actually clicked now resolves directly to its own run
        via ``_console_agent_drilldown_target_run_id``, whichever source
        built it. The dedicated Back button (unchanged) always returns to
        the overview directly.

        No-ops when the row cannot be resolved to a run belonging to the
        active conversation -- the rail simply stays on whatever it was
        already showing, the same defensive posture the old cycling method
        had for an empty run list.
        """
        target_run_id = self._console_agent_drilldown_target_run_id(row_id)
        if not target_run_id:
            return
        conversation_id = self._current_console_rail_conversation_id() or ""
        self._console_agent_drilldown_run_id = target_run_id
        self._console_agent_drilldown_conversation_id = conversation_id
        self.run_worker(
            self._sync_native_console_chat_ui,
            exclusive=True,
            group="console-sync",
        )

    def _apply_fleet_agent_section_auto_open(
        self, rail_state: ConsoleRailState
    ) -> ConsoleRailState:
        """Force the Agent rail section open while the fleet has anything to report.

        Parallel-agents spec §6, fix round 2 (live-smoke finding): the Agent
        section's persisted preference defaults collapsed (``agent_open=
        False``, see ``ConsoleRailPreferences``) and nothing previously
        reopened it, so its BODY -- the status/step/sub-agent detail for
        the viewed session's own run -- stayed ``display: none`` even
        while another session was running or parked on an approval.
        Scrolling the rail only ever reached the still-collapsed header;
        that detail was unreachable regardless of scroll position.

        TASK-1140 (UAT F1, fix round 1): ``#console-agent-fleet-summary``
        itself no longer lives inside this section's body -- it is now a
        pinned, non-scrolling sibling of the rail's own header (see the
        compose block a few hundred lines up), painted unconditionally
        whenever the fleet has anything to report, independent of this
        section's open/collapsed state. This method's force-open still
        matters for the Agent section's OWN contextual detail (status/
        steps/sub-agents for whichever conversation is viewed), just not
        for fleet-line visibility anymore.

        Mirrors ``_apply_pending_launch_inspector_auto_open``: an ephemeral
        override applied to the RENDERED rail state only, never written back
        to the persisted preference, so a user's own explicit collapse still
        takes effect the moment the fleet goes quiet (``fleet_summary_
        counts()`` returns to ``(0, 0)``). Uses ``_console_agent_fleet_
        summary_line()`` -- the exact same non-empty-string signal the
        pinned fleet Static's own ``display`` toggles on -- so "must render
        the section open" and "has a line to show" can never disagree.

        TASK-915: fix round 3 (live-smoke finding). Round 2's force was a
        one-shot-per-rendered-state override, but the 0.2s sync tick
        recomputes it every time the agent-section payload changes (e.g. a
        second background run starting/finishing) -- so a manual collapse
        while the fleet was busy held only until the NEXT such change, then
        got silently re-forced open. `_toggle_console_rail_section` now sets
        `_agent_section_user_dismissed_while_busy` when the user closes this
        section with a non-empty fleet line; honoured here so the force
        stays suppressed for the rest of THIS busy window. Still never
        touches the persisted preference -- only the transient flag and the
        returned dataclass change.
        """
        fleet_line = self._console_agent_fleet_summary_line()
        if not fleet_line:
            # Fleet is quiet: release any sticky dismissal so the NEXT busy
            # window auto-opens again (TASK-915 AC2).
            self._agent_section_user_dismissed_while_busy = False
            return rail_state
        if rail_state.agent_open:
            return rail_state
        if self._agent_section_user_dismissed_while_busy:
            return rail_state
        return replace(rail_state, agent_open=True)

    def _console_subagent_counts_refresh_needed(self, row_ids: frozenset) -> bool:
        """Decide whether the sub-agent badge-count cache needs a DB round trip.

        Finding A: refreshing on every 0.2s poll tick would re-issue the
        batched count query up to 5x/second even when nothing sub-agent
        related changed. This gates the refresh to three cheap-to-check
        conditions instead of refreshing unconditionally:

        1. The visible conversation row set changed (a rebuild) -- new
           rows may need counts we have never cached.
        2. A run is actively streaming/validating/retrying for this
           screen -- a just-spawned sub-agent's count should show up
           promptly rather than wait out the full TTL.
        3. The cache has aged past ``CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS``
           -- a fallback bound covering counts that changed from a
           different Console session/tab or a resumed run, where neither
           of the above two signals fires on this screen.

        Args:
            row_ids: The conversation ids of the currently visible browser
                rows (deduplicated, blanks excluded).

        Returns:
            ``True`` when the cache should be rebuilt from the DB.
        """
        # Imported in-body, not at module scope: both constants still
        # live in `chat_screen.py`, which imports this module's package at
        # import time and defines them well after that -- see this module's
        # docstring.
        from ..Screens.chat_screen import (
            CONSOLE_ACTIVE_RUN_STATUSES,
            CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS,
        )

        if row_ids != self._console_subagent_counts_cache_row_ids:
            return True
        controller = self._console_chat_controller
        if (
            controller is not None
            and controller.run_state.status in CONSOLE_ACTIVE_RUN_STATUSES
        ):
            return True
        age = time.monotonic() - self._console_subagent_counts_cache_at
        return age >= CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS

    def _console_subagent_counts_for_rows(
        self,
        bridge: Any | None,
        rows: Iterable[Any],
    ) -> Dict[str, int]:
        """Return ``conversation_id -> sub-agent count`` for browser rows.

        Finding A: previously called ``bridge.subagent_count(cid)`` once
        per row (a fresh sqlite connection per call) on every poll tick --
        up to ~75 queries/tick. Replaced with one batched
        ``bridge.subagent_counts(...)`` call, gated by
        ``_console_subagent_counts_refresh_needed`` so it isn't reissued
        unconditionally every tick either.

        Args:
            bridge: The Console agent bridge, or ``None`` when the agent
                runtime is unavailable (e.g. in-memory test harness).
            rows: The conversation-browser input rows currently visible.

        Returns:
            Mapping of ``conversation_id -> count``; conversations with
            zero sub-agent runs are simply absent (see
            ``AgentRunsDB.count_subagents_by_conversation``).
        """
        if bridge is None:
            return {}
        row_ids = frozenset(
            cid for row in rows if (cid := getattr(row, "conversation_id", None))
        )
        if self._console_subagent_counts_refresh_needed(row_ids):
            self._console_subagent_counts_cache = (
                bridge.subagent_counts(list(row_ids)) if row_ids else {}
            )
            self._console_subagent_counts_cache_row_ids = row_ids
            self._console_subagent_counts_cache_at = time.monotonic()
        return self._console_subagent_counts_cache

    def _inject_resume_agent_markers(
        self,
        messages: list[ConsoleChatMessage],
        conversation_id: str,
    ) -> list[ConsoleChatMessage]:
        """Re-derive and interleave TOOL markers from ``AgentRunsDB`` on resume.

        Plan-B final-review Medium-1: the rail already re-derives from
        ``AgentRunsDB`` on resume (``_console_agent_section_lines`` ->
        ``bridge.historical_snapshot``, and the ``[N Sub-Agents]`` badge);
        the inline transcript TOOL markers did not, since
        ``_console_messages_from_conversation_tree`` only ever reads
        persisted ChaChaNotes rows, where markers never land
        (``ConsoleAgentBridge._append_marker`` uses ``persist=False`` so
        agent activity survives a restart without being written into the
        conversation itself).

        Task 3: ``resume_marker_messages`` now pairs each run's block with
        the ``assistant_message_id`` of the reply it produced, and
        ``inject_resume_agent_markers`` anchors placement to that id
        against ``messages``'s own ``persisted_message_id``s -- a run
        whose reply isn't on the active path (edited/regenerated onto
        another branch) has its block hidden rather than misattributed to
        a different reply; a legacy/null-id run keeps the prior ordinal
        placement. ``messages`` is the caller's already-active-path
        transcript (``_console_messages_from_conversation_tree`` walks the
        active thread only), so this composes correctly with branching
        without any extra filtering here. See ``inject_resume_agent_
        markers`` for the full placement/idempotency contract, and
        ``resume_marker_messages`` for how each run's marker block and
        anchor id are derived.

        Returns ``messages`` unchanged when there is no durable agent
        bridge available (e.g. an in-memory test harness, matching
        ``_ensure_console_agent_bridge``'s own fallback).
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return messages
        from tldw_chatbook.Chat.console_agent_bridge import inject_resume_agent_markers
        from tldw_chatbook.Chat.thinking_blocks import ThinkingEnvelope

        thinking_rounds_by_owner = {
            message.persisted_message_id or message.id: frozenset(
                block.round_ordinal for block in message.thinking.blocks
            )
            for message in messages
            if message.role is ConsoleMessageRole.ASSISTANT
            and isinstance(message.thinking, ThinkingEnvelope)
        }

        # bridge.resume_marker_messages returns the (anchor_id, block) pairs
        # inject_resume_agent_markers now expects directly -- no reshaping
        # needed here, just passed straight through.
        return inject_resume_agent_markers(
            messages,
            bridge.resume_marker_messages(
                conversation_id,
                thinking_round_ordinals_by_assistant_message_id=(
                    thinking_rounds_by_owner
                ),
            ),
        )
