"""``MCPToolProvider`` — the task-201 bridge from the agent worker thread to
main-loop-bound MCP execution.

This is the concurrency-critical seam of the Phase 5 chat bridge: the agent
runtime (``Agents/agent_runtime.py``/``agent_service.py``) drives tool calls
from a *worker thread* (``asyncio.to_thread``), but MCP client sessions and
the rest of the control-plane's I/O are bound to Textual's *main* event
loop. ``invoke()`` — the only method the runtime calls per tool call — must
never touch Textual, never raise, and never hang unbounded.

Threading decision (binding, documented here since Task 2/4's seams leave it
to this module to decide): only the async execute path
(:meth:`UnifiedMCPControlPlaneService.execute_hub_tool`, which ultimately
touches an ``MCPClient`` session) is submitted to the main loop via
``asyncio.run_coroutine_threadsafe``. The sync, store-backed methods
(``gate_tool_test``, ``record_tool_decision``, ``is_session_approved``,
``approve_for_session``, ``set_tool_state``, ``get_kill_switch``,
``effective_tool_states``) do small, atomic file I/O with no event-loop
affinity, so this provider calls them *directly* from whichever thread it is
currently running on (worker thread for ``invoke()``/``pending_gate_for()``,
main loop for ``compose_catalog()``) rather than paying a second
cross-thread round trip for each one.

``compose_catalog()`` is the one method that itself performs async I/O
(:meth:`UnifiedMCPControlPlaneService.local_external_catalog`) — it is
documented to run ON the main loop at registration time (T6 awaits it
directly, before spawning the worker thread), so it is declared ``async def``
and does not need any cross-thread submission of its own.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_chatbook.MCP.hub_tool_catalog import (
    HubTool, builtin_tools_from_inventory, local_tools_from_record,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.MCP.redaction import redact_mapping
from tldw_chatbook.MCP.tool_naming import dedupe_names, llm_tool_name

from .agent_models import ToolCatalogEntry, ToolResult, ToolSchema

SOURCE = "mcp"

# Model-facing refusal copy -- exact strings per the Phase 5 plan (spec §11 /
# Global Constraints); T5's approval-card timeout path and T7/T8's audit
# canvas both key off these exact decision/error shapes, so they must not
# drift from what is logged via `record_tool_decision`.
DENY_REFUSAL = "blocked by MCP permissions (set to Off)"
TIMEOUT_REFUSAL = "user did not approve within the time limit; do not retry"
KILL_SWITCH_REFUSAL = "blocked — MCP tools are switched off"
NON_TEXT_PLACEHOLDER = "[image result — not yet supported]"

# `.result(timeout=...)` slack added on top of the configured per-call tool
# timeout, so a well-behaved `execute_hub_tool` call that itself times out
# (raising `RuntimeError` inside the coroutine) always has room to report
# that failure back before this provider's own wait gives up.
_RESULT_WAIT_SLACK_SECONDS = 5.0

_MAX_RESULT_CHARS = 4000
_MAX_ERROR_CHARS = 300
_NON_TEXT_CONTENT_TYPES = frozenset({"image", "blob"})

_FAIL_CLOSED_STATE = EffectiveToolState(state="ask", origin="global_default")


@dataclass(frozen=True)
class MCPPendingCall:
    """One tool call awaiting human approval, surfaced to the batch-approval UI."""

    llm_name: str
    server_key: str
    tool_name: str
    server_label: str
    arguments: dict
    reason: str  # ask|config_changed|risk_floored


def _has_non_text_content(value: Any) -> bool:
    """Defensively sniff a result payload for image/blob content entries.

    Walks nested Mappings/sequences looking for any dict whose ``type`` key
    is ``"image"`` or ``"blob"`` (the MCP content-item shape) — the model
    cannot consume binary payloads, so results containing them are replaced
    with :data:`NON_TEXT_PLACEHOLDER` instead of being JSON-dumped.
    """
    if isinstance(value, Mapping):
        if value.get("type") in _NON_TEXT_CONTENT_TYPES:
            return True
        return any(_has_non_text_content(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_text_content(v) for v in value)
    return False


def _pending_reason(state: EffectiveToolState) -> str:
    if state.config_changed:
        return "config_changed"
    if state.risk_floored:
        return "risk_floored"
    return "ask"


class MCPToolProvider:
    """``ToolProvider``: local + builtin MCP tools, gated per call.

    Server-source tools are out of scope (Phase 6) — this provider is built
    entirely from :meth:`UnifiedMCPControlPlaneService.local_external_catalog`
    (local external profiles) and the built-in server's inventory, mirroring
    the data path ``mcp_workbench.MCPWorkbench._collect_hub_tools`` uses for
    its ``source == "local"`` branch, but driven through the service from
    this non-UI module.

    All Protocol methods (``list_catalog``/``load_schema``/``invoke``) are
    SYNC, matching ``Agents/tool_catalog.ToolProvider``. ``invoke()`` runs on
    the agent worker thread; see the module docstring for the threading
    decision it and ``pending_gate_for()`` rely on.
    """

    def __init__(
        self,
        *,
        service: Any,
        main_loop: asyncio.AbstractEventLoop,
        approval_callback: Callable[[list[MCPPendingCall]], dict[str, str]] | None = None,
    ) -> None:
        """Build an uncomposed provider; call `compose_catalog()` before use.

        Args:
            service: The `UnifiedMCPControlPlaneService`-shaped object
                this provider drives (see the module docstring for which
                of its methods run cross-thread vs. directly).
            main_loop: The running Textual main event loop `invoke()`'s
                `_execute` submits `execute_hub_tool` coroutines onto.
            approval_callback: `invoke()`'s single-call fallback gate for
                an `"ask"`-state tool with no batch-review stamp (e.g. no
                `review_tool_calls` hook was wired for this run). `None`
                fails closed to deny.
        """
        self._service = service
        self._main_loop = main_loop
        self._approval_callback = approval_callback
        self._catalog: list[ToolCatalogEntry] = []
        # llm_name -> (HubTool, EffectiveToolState as resolved at composition
        # time). Built ONCE by compose_catalog() so list_catalog()/
        # load_schema()/invoke() never re-derive or re-list the catalog per
        # lookup (task-201's don't-re-list-per-lookup note).
        self._entry_by_llm_name: dict[str, tuple[HubTool, EffectiveToolState]] = {}
        self._not_connected_count = 0
        # Per-turn verdict stamps set by T6's batch-review closure via
        # apply_batch_decisions(). PEEKED (not popped) by invoke() via
        # stamped_decision() so every call sharing an llm_name within the
        # SAME turn sees the same verdict (Finding F1) -- apply_batch_
        # decisions() itself is what clears the previous turn's stamps
        # (REPLACE semantics, called every turn including with `{}` by
        # build_mcp_review_hook), not a per-read pop.
        self._stamped_decisions: dict[str, str] = {}

    # -- composition (main loop, once per registration) -------------------

    async def compose_catalog(self) -> None:
        """Build the eligible tool catalog. MAIN LOOP, called once at registration.

        Kill switch on -> empty catalog (the provider is effectively inert;
        T6 is expected not to even register it in that case, but this stays
        safe either way). Otherwise: derive `HubTool`s from local external
        profiles (`service.local_external_catalog()`) plus the built-in
        server's inventory, drop any whose resolved state is `"deny"` (one
        batched `effective_tool_states()` call, never a per-tool resolve),
        assign LLM-facing names via T1 (`llm_tool_name` computed for every
        tool FIRST, then ONE `dedupe_names` pass over the whole list -- the
        binding T1 handoff note: incremental dedupe breaks global
        uniqueness), and cache both the `ToolCatalogEntry` list and the
        `{llm_name: (HubTool, EffectiveToolState)}` lookup table.
        """
        # Clear stale stamped decisions from prior catalogs to prevent
        # auto-approval of tools not in the new catalog (Finding 3).
        self._stamped_decisions.clear()

        if self._service.get_kill_switch():
            self._catalog = []
            self._entry_by_llm_name = {}
            self._not_connected_count = 0
            return

        hub_tools: list[HubTool] = []
        records = await self._service.local_external_catalog()
        for record in records:
            hub_tools.extend(local_tools_from_record(record))

        local_service = getattr(self._service, "local_service", None)
        get_inventory = getattr(local_service, "get_inventory", None)
        if callable(get_inventory):
            try:
                inventory = get_inventory()
            except Exception as exc:  # noqa: BLE001 -- never abort composition
                logger.warning(f"MCPToolProvider: built-in inventory read failed: {exc}")
                inventory = None
            if isinstance(inventory, Mapping):
                hub_tools.extend(builtin_tools_from_inventory(inventory))

        effective = self._service.effective_tool_states(hub_tools)
        eligible = [
            tool for tool in hub_tools
            if effective.get((tool.server_key, tool.name), _FAIL_CLOSED_STATE).state != "deny"
        ]

        # Distinct servers (not tools) among the eligible, non-denied set
        # that are currently disconnected -- matches T6's "N servers
        # enabled, not connected" inspector affordance. Built-in tools are
        # never stale, so they never contribute here.
        self._not_connected_count = len({
            tool.server_key for tool in eligible if tool.stale
        })

        names = [llm_tool_name(tool.server_key, tool.name) for tool in eligible]
        deduped_names = dedupe_names(names)

        catalog: list[ToolCatalogEntry] = []
        entry_by_llm_name: dict[str, tuple[HubTool, EffectiveToolState]] = {}
        for tool, llm_name in zip(eligible, deduped_names):
            state = effective.get((tool.server_key, tool.name), _FAIL_CLOSED_STATE)
            catalog.append(ToolCatalogEntry(
                id=llm_name, name=llm_name,
                one_line_description=tool.description, source=SOURCE,
            ))
            entry_by_llm_name[llm_name] = (tool, state)

        self._catalog = catalog
        self._entry_by_llm_name = entry_by_llm_name

    @property
    def not_connected_count(self) -> int:
        """Distinct eligible local servers currently disconnected (T6 affordance)."""
        return self._not_connected_count

    # -- ToolProvider protocol (sync; cache reads only) --------------------

    def list_catalog(self) -> list[ToolCatalogEntry]:
        """Return this run's composed tool catalog.

        Returns:
            The `ToolCatalogEntry` list built by the most recent
            `compose_catalog()` call -- empty before composition has run,
            or when the kill switch was on at that time.
        """
        return list(self._catalog)

    def load_schema(self, tool_id: str) -> ToolSchema:
        """Resolve one catalog entry's invocation schema.

        Args:
            tool_id: An LLM-facing tool id from `list_catalog()` (a
                `ToolCatalogEntry.id`/`.name`), as produced by
                `MCP.tool_naming.llm_tool_name`/`dedupe_names` at
                composition time.

        Returns:
            The tool's `ToolSchema`, with `parameters` defaulted to an
            empty JSON-object schema (`{"type": "object", "properties":
            {}}`) when the underlying `HubTool` declared none.

        Raises:
            KeyError: `tool_id` is not present in the catalog built by
                the most recent `compose_catalog()` call.
        """
        entry = self._entry_by_llm_name.get(tool_id)
        if entry is None:
            raise KeyError(f"Unknown MCP tool id: {tool_id}")
        tool, _state = entry
        parameters = tool.input_schema if tool.input_schema is not None else {
            "type": "object", "properties": {},
        }
        return ToolSchema(
            id=tool_id, name=tool_id, description=tool.description,
            parameters=parameters,
        )

    # -- per-turn verdict stamps (set/read same-thread by T6's closure) ----

    def apply_batch_decisions(self, decisions: dict[str, str]) -> None:
        """Replace this turn's verdict stamps with `decisions`.

        REPLACE, not merge: any stamp left over from a prior turn is
        always cleared first (Finding F1) -- `build_mcp_review_hook` calls
        this exactly once per turn that has any tool calls at all,
        including with `{}` when none of them needed gating, specifically
        so a stale stamp can never survive into a later turn and be
        misread by `invoke()`'s `stamped_decision()` peek as this turn's
        verdict. Also cleared wholesale by `compose_catalog()` at
        registration time (a different, coarser-grained clear for stale
        catalogs, not a substitute for this per-turn one).

        Args:
            decisions: This turn's `{llm_name: verdict}` map, as returned
                by the batch-approval round trip
                (`ConsoleChatController.request_mcp_approvals`). Falsy
                clears without setting anything new.
        """
        self._stamped_decisions = dict(decisions or {})

    def stamped_decision(self, llm_name: str) -> str | None:
        """Peek at this turn's stamped verdict for `llm_name`, if any.

        Non-destructive on purpose (Finding F1): multiple calls to the
        same tool within one turn must ALL observe the identical stamped
        verdict, so reading here never removes it -- only
        `apply_batch_decisions` (called once per turn) ever clears a
        stamp.

        Args:
            llm_name: The LLM-facing tool id to look up.

        Returns:
            The stamped verdict string for `llm_name` this turn, or
            `None` if it has no stamp.
        """
        return self._stamped_decisions.get(llm_name)

    # -- gate resolution for the batch-review hook (worker thread) --------

    def pending_gate_for(self, llm_name: str, args: dict) -> MCPPendingCall | None:
        """Resolve one call's gate; return a pending descriptor iff it needs asking.

        Direct (not main-loop-submitted) call to `gate_tool_test` -- see the
        module docstring's threading decision. `None` for allow/deny: the
        caller (T6's closure) lets those flow through to `invoke()`'s own
        gate rather than duplicating that logic here. Also `None` when an
        "ask" tool already has a live session approval (Finding I1):
        `gate_tool_test` resolves from the permission store only and never
        consults session approvals, so without this check every ask-state
        tool would re-prompt on every turn even after "Approve for
        session" -- `invoke()`'s own fresh gate resolves the same tool via
        its `is_session_approved` short-circuit, so skipping the prompt
        here is exactly as correct as asking and having the user approve
        it again.

        Args:
            llm_name: The LLM-facing tool id to resolve, as reported on
                the incoming `ToolCall.name`.
            args: The tool call's raw arguments, copied verbatim into the
                returned `MCPPendingCall.arguments` (never mutated).

        Returns:
            An `MCPPendingCall` describing what needs asking, or `None`
            when `llm_name` is unknown to this provider, its resolved
            state doesn't need asking (`"allow"`/`"deny"`), or it already
            has a live session approval.
        """
        entry = self._entry_by_llm_name.get(llm_name)
        if entry is None:
            return None
        tool, _cached_state = entry
        try:
            state = self._service.gate_tool_test(tool)
        except Exception as exc:  # noqa: BLE001 -- fail closed to "let invoke handle it"
            logger.warning(
                f"MCPToolProvider: gate_tool_test failed for {tool.server_key}/{tool.name}: {exc}"
            )
            return None
        if state.state != "ask":
            return None
        if self._is_session_approved_safe(tool):
            return None
        return MCPPendingCall(
            llm_name=llm_name, server_key=tool.server_key, tool_name=tool.name,
            server_label=tool.server_label, arguments=dict(args or {}),
            reason=_pending_reason(state),
        )

    # -- invocation (WORKER THREAD) ----------------------------------------

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        """Execute one tool call. WORKER THREAD. Never raises, never hangs unbounded.

        Order: the kill switch is checked FIRST (Minor 5) and wins over
        everything else, including an already-stamped verdict from this
        turn's batch review. Absent a kill-switch refusal, a stamped
        verdict from `stamped_decision()` (set by T6's batch-review
        closure earlier this turn, PEEKED not popped -- Finding F1, so
        every call sharing this turn's `tool_id` sees the same verdict)
        wins outright; absent a stamp, this resolves a fresh gate itself
        (direct `gate_tool_test` call -- see module docstring), a live
        session approval short-circuits an `"ask"` state to execute
        (decision="approved"), and otherwise an `"ask"` verdict falls back
        to `self._approval_callback` as a single-call list (no callback ->
        fail closed to deny).

        Args:
            tool_id: The LLM-facing tool id to invoke (a
                `ToolCatalogEntry.id`/`.name`).
            args: The tool call's raw arguments.

        Returns:
            A `ToolResult`: `ok=True` with the formatted, redacted,
            length-capped result content on success; `ok=False` with a
            length-capped, always-non-empty `error` on refusal or
            failure. Never raises.
        """
        entry = self._entry_by_llm_name.get(tool_id)
        if entry is None:
            return ToolResult(ok=False, error=f"Unknown MCP tool: {tool_id}"[:_MAX_ERROR_CHARS])
        tool, _cached_state = entry
        call_args = dict(args or {})

        # Cheap hardening (Minor 5): a kill switch flipped after
        # compose_catalog() (or between a T6 batch-review stamp and this
        # dispatch) must still block execution -- checked before the
        # stamped-verdict short-circuit below so even an earlier-this-turn
        # approval cannot bypass it.
        if self._kill_switch_engaged():
            self._record_decision_safe(tool, decision="denied")
            return ToolResult(ok=False, error=KILL_SWITCH_REFUSAL)

        stamped = self.stamped_decision(tool_id)
        if stamped is not None:
            return self._apply_verdict(stamped, tool, call_args)

        try:
            state = self._service.gate_tool_test(tool)
        except Exception as exc:  # noqa: BLE001 -- invoke() must never raise
            return ToolResult(ok=False, error=str(exc)[:_MAX_ERROR_CHARS])

        if state.state == "deny":
            self._record_decision_safe(tool, decision="denied")
            return ToolResult(ok=False, error=DENY_REFUSAL)

        if state.state == "allow":
            return self._execute(tool, call_args, decision="allowed")

        if self._is_session_approved_safe(tool):
            # Finding I1: a live session approval is a DIFFERENT decision
            # than a permanent "allow" state -- keep the audit vocabulary
            # (and the model-facing execution record) distinct so Findings
            # mode can tell "server default was allow" apart from "the
            # user approved this session".
            return self._execute(tool, call_args, decision="approved")

        # state == "ask"
        if self._approval_callback is None:
            self._record_decision_safe(tool, decision="denied")
            return ToolResult(ok=False, error=DENY_REFUSAL)

        pending = MCPPendingCall(
            llm_name=tool_id, server_key=tool.server_key, tool_name=tool.name,
            server_label=tool.server_label, arguments=call_args,
            reason=_pending_reason(state),
        )
        try:
            decisions = self._approval_callback([pending])
        except Exception as exc:  # noqa: BLE001 -- invoke() must never raise
            return ToolResult(ok=False, error=str(exc)[:_MAX_ERROR_CHARS])
        verdict = (decisions or {}).get(tool_id, "deny")
        return self._apply_verdict(verdict, tool, call_args)

    # -- internals ----------------------------------------------------------

    def _kill_switch_engaged(self) -> bool:
        """Best-effort, never-raise read of the service's kill switch.

        Guarded (``getattr``/``try``) rather than a direct call: unlike
        ``compose_catalog`` (main-loop, once at registration -- a raise
        there is acceptable to surface), ``invoke()`` must never raise, and
        a fake/test double may not define this method at all.
        """
        getter = getattr(self._service, "get_kill_switch", None)
        if not callable(getter):
            return False
        try:
            return bool(getter())
        except Exception as exc:  # noqa: BLE001 -- a read failure must not block execution
            logger.warning(f"MCPToolProvider: get_kill_switch failed during invoke: {exc}")
            return False

    def _is_session_approved_safe(self, tool: HubTool) -> bool:
        try:
            return bool(self._service.is_session_approved(tool.server_key, tool.name))
        except Exception as exc:  # noqa: BLE001 -- a read failure must not deny silently-wrongly
            logger.warning(
                f"MCPToolProvider: is_session_approved failed for {tool.server_key}/{tool.name}: {exc}"
            )
            return False

    def _apply_verdict(self, verdict: str, tool: HubTool, args: dict) -> ToolResult:
        """Apply one verdict's side effects (if any), then execute or refuse.

        `"approve_once"` has no side effect; `"approve_session"` writes the
        in-memory session cache (T2's `approve_for_session`); `"always_allow"`
        persists a tool-level allow keyed to this call's live `HubTool` (the
        rug-pull guard's definition hash); `"timeout"`/`"deny"`/anything
        unrecognized fail closed to the exact model-facing refusal copy.
        """
        if verdict == "approve_once":
            return self._execute(tool, args, decision="approved")
        if verdict == "approve_session":
            self._safe_side_effect(
                lambda: self._service.approve_for_session(tool.server_key, tool.name),
                tool, what="approve_for_session",
            )
            return self._execute(tool, args, decision="approved")
        if verdict == "always_allow":
            self._safe_side_effect(
                lambda: self._service.set_tool_state(
                    tool.server_key, tool.name, "allow", tool=tool),
                tool, what="set_tool_state",
            )
            return self._execute(tool, args, decision="approved")
        if verdict == "timeout":
            self._record_decision_safe(tool, decision="denied-timeout")
            return ToolResult(ok=False, error=TIMEOUT_REFUSAL)
        # "deny" and any unrecognized verdict fail closed the same way.
        self._record_decision_safe(tool, decision="denied")
        return ToolResult(ok=False, error=DENY_REFUSAL)

    def _safe_side_effect(self, fn: Callable[[], None], tool: HubTool, *, what: str) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- a persistence failure must not block execution
            logger.warning(
                f"MCPToolProvider: {what} failed for {tool.server_key}/{tool.name}: {exc}"
            )

    def _record_decision_safe(self, tool: HubTool, *, decision: str, error: str | None = None) -> None:
        try:
            self._service.record_tool_decision(
                tool.server_key, tool.name, decision=decision, initiator="agent", error=error,
            )
        except Exception as exc:  # noqa: BLE001 -- best-effort audit trail only
            logger.warning(
                f"MCPToolProvider: record_tool_decision failed for {tool.server_key}/{tool.name}: {exc}"
            )

    def _execute(self, tool: HubTool, args: dict, *, decision: str) -> ToolResult:
        """Run the tool via the main loop. NEVER raises, NEVER hangs unbounded.

        Submits `service.execute_hub_tool(...)` to `self._main_loop` via
        `asyncio.run_coroutine_threadsafe` (the only I/O this provider
        cross-thread-submits -- see the module docstring) and bounds the
        wait at the configured per-call timeout plus a fixed slack. Every
        exception along this path (the submit itself on a dead/closed loop,
        `concurrent.futures.TimeoutError`, or any exception the coroutine
        raised) is caught here and converted to a truncated error
        `ToolResult` -- this method must never propagate.

        Args:
            tool: The resolved `HubTool` to execute.
            args: The call's arguments, passed through unchanged.
            decision: The audit decision string this call was authorized
                under (e.g. `"allowed"`/`"approved"`), forwarded to
                `execute_hub_tool` and, on a bridge failure (Finding F2),
                to the best-effort audit record below.

        Returns:
            A `ToolResult`: `ok=True` with the formatted result on
            success; `ok=False` with a non-empty, length-capped `error`
            on any failure. Never raises.
        """
        future: asyncio.Future | None = None
        try:
            timeout = self._service._tool_call_timeout() + _RESULT_WAIT_SLACK_SECONDS
            future = asyncio.run_coroutine_threadsafe(
                self._service.execute_hub_tool(
                    tool.server_key, tool.name, args,
                    initiator="agent", decision=decision,
                ),
                self._main_loop,
            )
            raw_result = future.result(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- the never-raise/never-hang contract
            # Finding F3: `future` may still be unbound here if
            # `run_coroutine_threadsafe` itself raised (e.g. a dead/closed
            # loop) -- guard before cancelling rather than assuming the
            # submit succeeded.
            if future is not None:
                # Finding 2: best-effort cancel lingering future on timeout/cancellation.
                try:
                    future.cancel()
                except Exception:
                    pass
            # Finding 1: TimeoutError/CancelledError have empty str(), so guarantee
            # non-empty error via (str(exc) or repr(exc)) so the model receives actual info.
            error = (str(exc) or repr(exc))[:_MAX_ERROR_CHARS]
            # Finding F2: a bridge failure here (submit-time raise, or the
            # bounded wait timing/erroring out) previously left NO audit
            # trail at all -- best-effort record it now so Findings mode
            # can still see the call happened and why it failed.
            self._record_decision_safe(
                tool, decision=decision,
                error=f"bridge execution failed: {(str(exc) or repr(exc))[:200]}",
            )
            return ToolResult(ok=False, error=error)
        return self._format_result(raw_result)

    def _format_result(self, raw_result: Any) -> ToolResult:
        try:
            if isinstance(raw_result, Mapping):
                if _has_non_text_content(raw_result):
                    return ToolResult(ok=True, content=NON_TEXT_PLACEHOLDER)
                content = json.dumps(redact_mapping(raw_result), default=str)
            else:
                # Defensive only: execute_hub_tool's real contract always
                # returns a dict: a non-Mapping raw result would come from a
                # nonconforming fake/future backend, not production.
                content = str(raw_result)
            return ToolResult(ok=True, content=content[:_MAX_RESULT_CHARS])
        except Exception as exc:  # noqa: BLE001 -- formatting must not turn success into a raise
            return ToolResult(ok=False, error=str(exc)[:_MAX_ERROR_CHARS])
