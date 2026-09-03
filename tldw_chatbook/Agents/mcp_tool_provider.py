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

PR2a Task 8: with the fleet, this ONE provider instance's ``invoke()`` can
now be called from several worker threads at once (a parent run and its
live children). ``invoke()`` serializes every call to this provider
instance behind ``self._invoke_lock`` -- see its docstring (and
``_invoke_locked``'s) for exactly what that protects and what would let a
future task remove it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

# NOTE (boot budget, ADR-097): `persona_policy` is imported lazily -- the
# `PersonaToolPolicy` reference is annotation-only (future annotations
# above) and `persona_floor_state` is used at the invoke-time gate below --
# so the module stays off the UI-ready census path.
if TYPE_CHECKING:
    from tldw_chatbook.Agents.persona_policy import PersonaToolPolicy
from tldw_chatbook.Agents.builtin_tool_gate import DENIAL_POLICY
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import TOOL_DESCRIPTION_CAPTURE_CAP
from tldw_chatbook.MCP.execution_log import APPROVED_SESSION_DECISION
from tldw_chatbook.MCP.hub_tool_catalog import (
    HubTool,
    builtin_tools_from_inventory,
    local_tools_from_record,
    schema_argument_names,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.MCP.redaction import redact_mapping
from tldw_chatbook.MCP.tool_naming import dedupe_names, llm_tool_name

from .agent_models import ToolCatalogEntry, ToolResult, ToolSchema
from .run_context import current_run_id
from .tool_catalog import ToolExecutionPolicy

SOURCE = "mcp"

# Model-facing refusal copy -- exact strings per the Phase 5 plan (spec §11 /
# Global Constraints); T5's approval-card timeout path and T7/T8's audit
# canvas both key off these exact decision/error shapes, so they must not
# drift from what is logged via `record_tool_decision`.
DENY_REFUSAL = "blocked by MCP permissions (set to Off)"

#: TASK-294: an EXPLICIT user "Deny" on the approval card. Distinct from
#: `DENY_REFUSAL`, which correctly describes the permanent permissions-Off
#: state -- blaming a user's per-call "no" on configuration is misleading
#: provenance (a model reading it retries never; a user reading the
#: transcript goes hunting for a setting they never flipped). Wording
#: matches the builtin gate's and the review hook's user-denial copy.
USER_DENY_REFUSAL = f"tool call denied by the user. {DENIAL_POLICY}"

#: TASK-294: a verdict that is MISSING or unrecognized after the approval
#: round trip. Fails closed like a deny, but blames nobody: the user never
#: decided, and the permissions were not Off.
UNRESOLVED_REFUSAL = "tool call not approved (no decision recorded)"
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
    options: tuple[str, ...] = ()
    #: Per-call verdict key. The provider's tool-call id when the model gave
    #: one (native tool-calling), else "" -- the fence path builds ToolCalls
    #: with no id. The runtime looks up `call_id` FIRST and falls back to
    #: `llm_name`, so an empty value here simply keeps the old shared-verdict
    #: behavior for that call rather than breaking it.
    call_id: str = ""
    #: TASK-1231/F3 AC2: True when this is a builtin file tool
    #: (read_file/list_directory/write_file) whose path argument will be
    #: rejected by `allowed_file_roots`/`validate_path_multi` regardless of
    #: the user's decision -- computed at card-build time by
    #: `console_chat_controller.build_tool_review_hook` via
    #: `Tools.file_operation_tools.path_precheck_failed`. This is a WARNING
    #: only: the user can still approve (and the call will then fail with
    #: the same recovery-route error `validate_path_multi` raises at
    #: dispatch) -- it must never be used to auto-deny. Always `False` for
    #: MCP rows and every non-file builtin tool.
    path_precheck_failed: bool = False
    #: Optional complete command for approval surfaces that must not use the
    #: generic compact argument summary. Raw shell is currently the only
    #: producer; ordinary rows leave this empty.
    full_command: str = ""
    #: Optional plain-text danger copy rendered beside the complete command.
    warning: str = ""
    #: Optional plain-text explanation of a broader approval scope.
    scope_notice: str = ""
    #: Code-owned action effects supplied by local descriptors. Existing
    #: MCP and builtin callers intentionally retain the empty default.
    effects: tuple[str, ...] = ()
    #: Runtime ownership after approved execution starts.  Unknown/external
    #: rows retain the bounded default; only an exact code-owned enum opts in.
    execution_policy: ToolExecutionPolicy = ToolExecutionPolicy.BOUNDED_ABANDONABLE
    #: ADR-090: the model's advisory rationale for this call (advisory
    #: display only -- never gates, never persists).
    rationale: str = ""
    #: ADR-090: the tool definition's description, for the external
    #: summarizer prompt; "" when the owner had none at hand.
    description: str = ""


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
        approval_callback: Callable[[list[MCPPendingCall]], dict[str, str]]
        | None = None,
        builtin_raw_name_exclusions: Any = None,
        profile_id_provider: Callable[[], str] | None = None,
        persona_policy_provider: Callable[[], "PersonaToolPolicy | None"] | None = None,
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
            builtin_raw_name_exclusions: task-1337 (plan Task 8): optional
                iterable of raw tool names dropped during `compose_catalog()`
                when (and only when) they arrive from the
                `builtin:tldw_chatbook` source -- the Console uses this to
                keep its own Library provider from being bypassed by the
                built-in MCP copies. Same-named tools on local/server
                sources are unaffected. Stored as an immutable frozenset;
                `None` (default) preserves current behavior for every
                non-Console caller.
            profile_id_provider: Workspace assistant defaults (Task 6):
                callable returning the permission profile id this
                provider's catalog resolution and always-allow persist
                path run under (see `_profile_kwargs`). Read at CALL time,
                not construction time, so a Console session can switch
                the active workspace binding without rebuilding the
                provider. `None` (default) resolves the `"default"`
                profile -- byte-identical to the pre-profiles behavior.
            persona_policy_provider: Workspace assistant defaults
                (final review): callable returning the run's parsed
                persona tool policy (or `None`). When present, the
                invoke-time fresh gates (`pending_gate_for` and
                `invoke`'s own gate) pass the resolved state through
                `persona_floor_state`, so a persona
                `require_confirmation` rule floors the tool to "ask"
                even under a profile/persisted allow grant. `None`
                (default) is byte-identical to the pre-feature behavior.
        """
        self._service = service
        self._main_loop = main_loop
        self._approval_callback = approval_callback
        self._builtin_raw_name_exclusions = frozenset(builtin_raw_name_exclusions or ())
        # Read fresh on every catalog compose / persist -- never cached, so
        # the active workspace profile can change over this provider's
        # lifetime (Task 7's Console closure supplies the callable).
        self._profile_id = profile_id_provider or (lambda: "default")
        # Persona require_confirmation floor (final review): read fresh per
        # gate resolution; None keeps every pre-feature call identical.
        self._persona_policy_provider = persona_policy_provider
        self._catalog: list[ToolCatalogEntry] = []
        # llm_name -> (HubTool, EffectiveToolState as resolved at composition
        # time). Built ONCE by compose_catalog() so list_catalog()/
        # load_schema()/invoke() never re-derive or re-list the catalog per
        # lookup (task-201's don't-re-list-per-lookup note).
        self._entry_by_llm_name: dict[str, tuple[HubTool, EffectiveToolState]] = {}
        self._not_connected_count = 0
        self._init_decision_state()

    def _init_decision_state(self) -> None:
        """Initialize the per-turn verdict stamps, their lock, and the
        provider-wide execution lock.

        Called from ``__init__``. Factored out so a test can build a bare
        instance (``MCPToolProvider.__new__``) and exercise the stamp
        contract without a service or a running event loop -- see
        ``Tests/Agents/test_gate_run_scoping.py``.

        Stamps are set by the batch-review closure via
        ``apply_batch_decisions()`` and PEEKED (not popped) by ``invoke()``
        via ``stamped_decision()``, so every call sharing an llm_name
        within the SAME turn of the SAME run sees the same verdict
        (Finding F1) -- ``apply_batch_decisions()`` itself is what clears
        that run's previous turn (REPLACE-within-the-run semantics, called
        every turn including with ``{}`` by ``build_mcp_review_hook``),
        not a per-read pop.

        PR2a Task 5: the dict is keyed ``(run_id, llm_name)``, not
        ``llm_name``. It is shared by every run in a tree (parent and all
        sub-agents use ONE provider), and the old whole-dict REPLACE meant
        any run's turn destroyed every other run's verdicts.
        """
        self._stamped_decisions: dict[tuple[str, str], str] = {}
        # Lock, not RLock: every critical section below is a short,
        # self-contained mutation of this one dict, and no locked method
        # calls another locked method (`stamp_scope` explicitly does NOT
        # hold the lock across its `yield` -- a nested run must never
        # block on it). A plain Lock makes an accidental future nesting
        # deadlock loudly instead of silently permitting a non-atomic
        # critical section.
        self._decisions_lock = threading.Lock()
        # PR2a Task 8 (provider thread-safety audit): a SEPARATE lock,
        # entirely unrelated to `_decisions_lock` above -- see `invoke()`'s
        # own docstring for what it protects and why. Defined here (not
        # only in `__init__`) for the same bare-instance-test reason
        # `_decisions_lock` is: a double it constructs via
        # `MCPToolProvider.__new__` + `_init_decision_state()` must not
        # `AttributeError` if it goes on to call `invoke()`.
        self._invoke_lock = threading.Lock()

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
        # auto-approval of tools not in the new catalog (Finding 3). Every
        # run's slice, deliberately: this runs at registration time, before
        # any run of the new catalog exists, so there is nothing to keep.
        with self._decisions_lock:
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
                logger.warning(
                    f"MCPToolProvider: built-in inventory read failed: {exc}"
                )
                inventory = None
            if isinstance(inventory, Mapping):
                builtin_tools = builtin_tools_from_inventory(inventory)
                if self._builtin_raw_name_exclusions:
                    # task-1337 (plan Task 8): drop the Console-shadowed raw
                    # names from the built-in source ONLY -- same-named tools
                    # on local/server sources are unaffected (the raw-name
                    # match alone must never reach them).
                    builtin_tools = [
                        tool
                        for tool in builtin_tools
                        if not (
                            tool.server_key == "builtin:tldw_chatbook"
                            and tool.name in self._builtin_raw_name_exclusions
                        )
                    ]
                hub_tools.extend(builtin_tools)

        effective = self._service.effective_tool_states(
            hub_tools, **self._profile_kwargs()
        )
        eligible = [
            tool
            for tool in hub_tools
            if effective.get((tool.server_key, tool.name), _FAIL_CLOSED_STATE).state
            != "deny"
        ]

        # Distinct servers (not tools) among the eligible, non-denied set
        # that are currently disconnected -- matches T6's "N servers
        # enabled, not connected" inspector affordance. Built-in tools are
        # never stale, so they never contribute here.
        self._not_connected_count = len(
            {tool.server_key for tool in eligible if tool.stale}
        )

        names = [llm_tool_name(tool.server_key, tool.name) for tool in eligible]
        deduped_names = dedupe_names(names)

        catalog: list[ToolCatalogEntry] = []
        entry_by_llm_name: dict[str, tuple[HubTool, EffectiveToolState]] = {}
        for tool, llm_name in zip(eligible, deduped_names):
            state = effective.get((tool.server_key, tool.name), _FAIL_CLOSED_STATE)
            catalog.append(
                ToolCatalogEntry(
                    id=llm_name,
                    name=llm_name,
                    one_line_description=tool.description,
                    source=SOURCE,
                )
            )
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
        parameters = (
            tool.input_schema
            if tool.input_schema is not None
            else {
                "type": "object",
                "properties": {},
            }
        )
        return ToolSchema(
            id=tool_id,
            name=tool_id,
            description=tool.description,
            parameters=parameters,
        )

    # -- per-turn verdict stamps, keyed (run_id, llm_name) -----------------
    #
    # No longer same-thread: PR2a Task 6 runs sub-agents on their own
    # threads against this one shared provider, so every access below goes
    # through `_decisions_lock` and every entry is scoped to the run that
    # wrote it.

    def apply_batch_decisions(self, run_id: str, decisions: dict[str, str]) -> None:
        """Replace `run_id`'s turn verdict stamps with `decisions`.

        REPLACE **within that run's slice**, not merge: any stamp `run_id`
        left over from a PRIOR turn is always cleared first (Finding F1) --
        `build_mcp_review_hook` calls this exactly once per turn that has
        any tool calls at all, including with `{}` when none of them needed
        gating, specifically so a stale stamp can never survive into a
        later turn and be misread by `invoke()`'s `stamped_decision()` peek
        as this turn's verdict. Passing `{}` therefore still CLEARS; that
        is what makes the clear-at-entry (I3) discipline work.

        What it no longer does is clear OTHER runs' slices (PR2a Task 5).
        The whole dict is shared by a parent and every sub-agent it spawns,
        so the old whole-dict replace meant any run's routine clear
        destroyed verdicts a concurrent sibling -- or the parent -- had
        already been granted and not yet consumed. Also cleared wholesale
        by `compose_catalog()` at registration time (a different,
        coarser-grained clear for stale catalogs, not a substitute for this
        per-turn one).

        Args:
            run_id: The run whose turn these decisions belong to. Only
                that run's own `invoke()` can consume them.
            decisions: This turn's `{llm_name: verdict}` map, as returned
                by the batch-approval round trip
                (`ConsoleChatController.request_mcp_approvals`). Falsy
                clears this run's slice without setting anything new.
        """
        with self._decisions_lock:
            self._stamped_decisions = {
                key: value
                for key, value in self._stamped_decisions.items()
                if key[0] != run_id
            }
            for llm_name, verdict in (decisions or {}).items():
                self._stamped_decisions[(run_id, llm_name)] = verdict

    def stamped_decision(self, run_id: str, llm_name: str) -> str | None:
        """Peek at `run_id`'s stamped verdict for `llm_name`, if any.

        Non-destructive on purpose (Finding F1): multiple calls to the
        same tool within one turn must ALL observe the identical stamped
        verdict, so reading here never removes it -- only
        `apply_batch_decisions` (called once per turn) ever clears a
        stamp.

        Args:
            run_id: The run consuming the verdict. A verdict stamped by a
                different run in the same tree is invisible here -- that
                isolation is the point of the per-run key.
            llm_name: The LLM-facing tool id to look up.

        Returns:
            The stamped verdict string for `llm_name` this turn, or
            `None` if it has no stamp.
        """
        with self._decisions_lock:
            return self._stamped_decisions.get((run_id, llm_name))

    @contextlib.contextmanager
    def stamp_scope(self, run_id: str):
        """Snapshot `run_id`'s stamps on enter; RESTORE (not merge) on exit.

        C1 (probe-verified security regression), re-scoped by PR2a Task 5:
        wired as `AgentService`'s `review_state_scope` (see that class's
        own docstring comment) by
        `console_agent_bridge.ConsoleAgentBridge.run_reply`, wrapping every
        NESTED sub-agent run this provider's turn spawns. `spawn_subagent`
        runs the child's entire loop INLINE, synchronously, before the
        parent's own remaining same-batch tool calls are dispatched
        (`agent_service.AgentService._run_one`'s `spawn` closure, called
        from `agent_runtime.run_agent_loop`'s per-call dispatch loop).
        `apply_batch_decisions` used to REPLACE the WHOLE dict every turn
        (Finding F1 -- never merged), so the child's OWN turn(s) silently
        clobbered whatever the PARENT's turn had already stamped for a
        same-named tool before the parent got to consume it -- letting a
        call the user just denied execute anyway (the child happens to
        approve the same tool name), or wiping a genuine parent approval
        (the child's own routine `apply_batch_decisions({})` clear for an
        unrelated, non-MCP tool call).

        **Per-run keying is now the real mechanism**: a child's own turn
        only ever rewrites its own `(child_run_id, ...)` keys, which is
        also the only thing that works once children run CONCURRENTLY --
        snapshot/restore is sound only for a strictly nested (LIFO) inline
        child. This is kept as belt-and-braces for that inline path, and
        because the seam is public (`_combine_state_scopes` composes this
        with the built-in gate's and the local provider's).

        Args:
            run_id: The run whose slice is snapshotted and restored --
                normally the PARENT's, whose verdicts a nested run must
                not disturb.
        """
        with self._decisions_lock:
            snapshot = {
                key: value
                for key, value in self._stamped_decisions.items()
                if key[0] == run_id
            }
        try:
            yield
        finally:
            with self._decisions_lock:
                self._stamped_decisions = {
                    key: value
                    for key, value in self._stamped_decisions.items()
                    if key[0] != run_id
                }
                self._stamped_decisions.update(snapshot)

    # -- gate resolution for the batch-review hook (worker thread) --------

    def pending_gate_for(
        self,
        llm_name: str,
        args: dict,
        call_id: str = "",
        rationale: str = "",
    ) -> MCPPendingCall | None:
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
            call_id: The provider's per-call id, carried onto the row so the
                approval card can offer one decision per TARGET rather than
                one per tool name. Empty when the model's payload had no id
                (`ensure_tool_call_ids` fills those in for the native path);
                an empty id makes the row collapse by name, which shares one
                verdict across every same-name call in the batch.
            rationale: The call's advisory rationale (ADR-090), copied
                verbatim onto the row.

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
            # Task 7 (controller ruling from Task 6's review): the FRESH gate
            # resolves under the ACTIVE workspace profile, never the default
            # one -- a tool set to "ask" in the named profile but "allow" in
            # default must surface its ask here, not fall through to a silent
            # default-profile execution at invoke.
            state = self._persona_floor(
                self._service.gate_tool_test(tool, **self._profile_kwargs()), tool
            )
        except Exception as exc:  # noqa: BLE001 -- fail closed to "let invoke handle it"
            logger.warning(
                f"MCPToolProvider: gate_tool_test failed for {tool.server_key}/{tool.name}: {exc}"
            )
            return None
        if state.state != "ask":
            return None
        if self._is_session_approved_safe(tool):
            return None
        if self._arg_rule_allows_safe(tool, args):
            # TASK-26012: a stored argument-scoped allow quiets exactly this
            # call; non-matching arguments for the same tool still ask.
            return None
        return MCPPendingCall(
            llm_name=llm_name,
            server_key=tool.server_key,
            tool_name=tool.name,
            server_label=tool.server_label,
            arguments=dict(args or {}),
            # TASK-1861: carry the per-call key through, or the card collapses
            # every same-name MCP call into one `xN` row with one verdict --
            # the defect the per-call re-key fixed for built-in tools.
            call_id=call_id,
            rationale=rationale,
            description=str(getattr(tool, "description", "") or "")[
                :TOOL_DESCRIPTION_CAPTURE_CAP
            ],
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
        (decision="approved-session"), and otherwise an `"ask"` verdict
        falls back to `self._approval_callback` as a single-call list (no
        callback -> fail closed to deny).

        PR2a Task 8 (provider thread-safety audit): this whole call runs
        under `self._invoke_lock`, so at most ONE call into this provider
        instance executes at a time, fleet-wide -- see `_invoke_locked`
        (just below) for why.

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
        with self._invoke_lock:
            return self._invoke_locked(tool_id, args)

    def _invoke_locked(self, tool_id: str, args: dict) -> ToolResult:
        """``invoke()``'s actual body -- entered ONLY while holding
        `self._invoke_lock`. Do not call this directly; call `invoke()`.

        Per spec (Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-
        design.md) §5's corrections table: "Tool providers were written
        under one-run-at-a-time dispatch ... Phase-2 thread-safety audit
        (MCP control-plane client, local tools, gated builtins). Unaudited
        provider => per-provider execution lock on invoke (throttle, not
        break). MCP starts locked until proven otherwise." PR2a Task 8's
        audit (see that task's report) found `BuiltinToolProvider` and
        `LocalToolProvider` safe under concurrent `invoke()` by inspection
        (read-only/immutable per-instance state, or state already guarded
        by its own lock) and left them unlocked; THIS provider is the one
        the spec calls out by name, for a reason the audit could not rule
        out by reading Python alone: `_execute()` below hands off to
        `self._service.execute_hub_tool(...)`, which for a LOCAL external
        MCP server ultimately reads/writes that server's own stdio pipe
        through an `MCPClient` session -- a request/response protocol this
        codebase has never exercised with two requests in flight on the
        same session at once (server-source tools are explicitly out of
        scope per this module's own docstring, i.e. genuinely unaudited,
        not merely unlikely to be a problem). Two concurrent fleet agents
        both calling an MCP tool is exactly the scenario the fleet makes
        routine.

        This lock SERIALIZES every call into this ONE provider instance
        across the whole fleet -- a throughput throttle on MCP tool calls
        specifically (a call blocks for up to the provider's own timeout
        plus its result-wait slack while holding the lock), not a break in
        concurrency: builtin, local, and Library tool calls from the same
        fleet turn have their own provider instances and are unaffected,
        and a queued MCP call still eventually runs rather than failing.

        What would let a future task remove this: proving (not assuming)
        that `MCPClient`'s local-server transport safely multiplexes
        concurrent request/response pairs on one session -- e.g. a
        request-id-keyed reader loop in `MCPClient` itself -- or giving
        each concurrently-live run its own subprocess/session instead of
        sharing this provider's one `self._service`. Absent either, this
        lock is the correctness boundary, not merely a performance choice.
        """
        entry = self._entry_by_llm_name.get(tool_id)
        if entry is None:
            return ToolResult(
                ok=False, error=f"Unknown MCP tool: {tool_id}"[:_MAX_ERROR_CHARS]
            )
        tool, _cached_state = entry
        call_args = dict(args or {})

        # Cheap hardening (Minor 5): a kill switch flipped after
        # compose_catalog() (or between a T6 batch-review stamp and this
        # dispatch) must still block execution -- checked before the
        # stamped-verdict short-circuit below so even an earlier-this-turn
        # approval cannot bypass it.
        if self._kill_switch_engaged():
            self._record_decision_safe(tool, decision="denied")
            return ToolResult.blocked(KILL_SWITCH_REFUSAL)

        # PR2a Task 5: only THIS run's own stamp may resolve this call. The
        # `ToolProvider.invoke` Protocol has no run parameter, so the
        # dispatching run id rides `run_context` (bound by `AgentService`
        # around each invocation -- see that module's docstring). Outside
        # any run this is `""`, which matches no stamp a review hook ever
        # writes, so such a call falls through to the fresh gate below --
        # the same path it took before batch review existed.
        stamped = self.stamped_decision(current_run_id(), tool_id)
        if stamped is not None:
            return self._apply_verdict(stamped, tool, call_args)

        try:
            # Task 7 (controller ruling from Task 6's review): same fix as
            # `pending_gate_for` above -- the fresh gate resolves under the
            # ACTIVE workspace profile, so a named-profile "ask" beats a
            # default-profile "allow" here too (an approval round, never a
            # silent execution).
            state = self._persona_floor(
                self._service.gate_tool_test(tool, **self._profile_kwargs()), tool
            )
        except Exception as exc:  # noqa: BLE001 -- invoke() must never raise
            return ToolResult(ok=False, error=str(exc)[:_MAX_ERROR_CHARS])

        if state.state == "deny":
            self._record_decision_safe(tool, decision="denied")
            return ToolResult.blocked(DENY_REFUSAL)

        if state.state == "allow":
            return self._execute(tool, call_args, decision="allowed")

        if self._is_session_approved_safe(tool):
            # Finding I1: a live session approval is a DIFFERENT decision
            # than a permanent "allow" state -- keep the audit vocabulary
            # (and the model-facing execution record) distinct so Findings
            # mode can tell "server default was allow" apart from "the
            # user approved this session".
            return self._execute(tool, call_args, decision=APPROVED_SESSION_DECISION)

        # state == "ask"
        if self._arg_rule_allows_safe(tool, call_args):
            return self._execute(tool, call_args, decision="allowed")
        if self._approval_callback is None:
            self._record_decision_safe(tool, decision="denied")
            return ToolResult.blocked(DENY_REFUSAL)

        pending = MCPPendingCall(
            llm_name=tool_id,
            server_key=tool.server_key,
            tool_name=tool.name,
            server_label=tool.server_label,
            arguments=call_args,
            reason=_pending_reason(state),
        )
        try:
            decisions = self._approval_callback([pending])
        except Exception as exc:  # noqa: BLE001 -- invoke() must never raise
            return ToolResult(ok=False, error=str(exc)[:_MAX_ERROR_CHARS])
        # TASK-294: default to a DISTINCT sentinel, not "deny" -- a missing
        # verdict means nobody decided, and collapsing it into "deny" here
        # is what used to blame the user (or the permissions) for a refusal
        # no one made. `_apply_verdict`'s fall-through maps it to
        # `UNRESOLVED_REFUSAL`; the fail-closed posture is unchanged.
        verdict = (decisions or {}).get(tool_id, "unresolved")
        return self._apply_verdict(verdict, tool, call_args)

    # -- internals ----------------------------------------------------------

    def _persona_floor(
        self, state: EffectiveToolState, tool: HubTool
    ) -> EffectiveToolState:
        """Apply the persona `require_confirmation` floor to a fresh gate
        state; identity when no persona policy provider is wired.

        Narrowing-only (`allow` -> `ask`); a `None` policy or a raise from
        the provider leaves the state untouched rather than blocking the
        call -- the persona floor never widens or invents refusals.
        """
        if self._persona_policy_provider is None:
            return state
        try:
            policy = self._persona_policy_provider()
        except Exception as exc:  # noqa: BLE001 -- a broken provider never blocks invoke
            logger.warning(
                "MCPToolProvider: persona_policy_provider failed for {}; "
                "error_type={}",
                tool.name,
                type(exc).__name__,
            )
            return state
        if policy is None:
            return state
        # Lazy import (boot budget, ADR-097): invoke-time gate only.
        from tldw_chatbook.Agents.persona_policy import persona_floor_state

        return persona_floor_state(state, policy, tool.name)

    def _profile_kwargs(self) -> dict[str, str]:
        """Keyword args threading the active permission profile into the
        service's profile-aware permission seams.

        Workspace assistant defaults (Task 6). Returns ``{}`` when the
        active profile is ``"default"``: the production service treats a
        bare call and ``profile_id="default"`` as byte-identical, but this
        provider's contract is also exercised against signature-exact
        service doubles that predate profiles and reject the keyword
        outright (see ``Tests/Agents/test_mcp_tool_provider.py``'s own
        "no ``**kwargs`` masking" rule), so the default-profile path keeps
        calling exactly as it did before profiles existed. Only a genuinely
        NAMED profile changes the call shape.

        Returns:
            ``{"profile_id": <id>}`` for a named active profile, else
            ``{}`` (omit the keyword entirely).
        """
        profile_id = self._profile_id()
        return {} if profile_id == "default" else {"profile_id": profile_id}

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
            logger.warning(
                f"MCPToolProvider: get_kill_switch failed during invoke: {exc}"
            )
            return False

    def _arg_rule_allows_safe(self, tool: HubTool, args: Mapping[str, Any] | dict) -> bool:
        """TASK-26012: whether a stored argument-scoped rule quiets this call.

        Duck-typed and fail-closed: a service without the capability (or a
        raising one) means no rule matched. The store enforces the rug-pull
        hash and the high-risk floor internally.
        """
        checker = getattr(self._service, "arg_rule_allows_call", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(tool, dict(args or {}), **self._profile_kwargs()))
        except Exception as exc:  # noqa: BLE001 -- a broken rule read never allows
            logger.warning(
                f"MCPToolProvider: arg_rule_allows_call failed for {tool.server_key}/{tool.name}: {exc}"
            )
            return False

    def _is_session_approved_safe(self, tool: HubTool) -> bool:
        try:
            return bool(
                self._service.is_session_approved(
                    tool.server_key, tool.name, **self._profile_kwargs()
                )
            )
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

        Minor 4: a stamped verdict is PEEKED, not popped (Finding F1), so
        TWO calls to the same tool id in one turn both run through here with
        the identical `"approve_session"`/`"always_allow"` verdict -- each
        re-triggers `approve_for_session`/`set_tool_state`. This is
        harmless today, not an apply-once bug: `approve_for_session` just
        re-adds the same key to an in-memory set, and `set_tool_state`
        persists the same last-write-wins `"allow"` state keyed by the same
        live `HubTool` hash -- both idempotent, so a redundant second call
        wastes a little I/O but changes nothing observable. Left as a
        documented redundancy rather than an apply-once guard, which would
        need its own per-turn "already applied" state to track and would
        only ever save that one redundant write.
        """
        if verdict == "approve_once":
            return self._execute(tool, args, decision="approved")
        if verdict == "approve_session":
            already_approved = self._is_session_approved_safe(tool)
            self._safe_side_effect(
                lambda: self._service.approve_for_session(
                    tool.server_key, tool.name, **self._profile_kwargs()
                ),
                tool,
                what="approve_for_session",
            )
            decision = APPROVED_SESSION_DECISION if already_approved else "approved"
            return self._execute(tool, args, decision=decision)
        if verdict == "allow_matching":
            # TASK-26012: persist an allow scoped to EXACTLY the displayed
            # arguments (AC#3) -- never a whole-tool allow. Rug-pull hashing
            # happens service-side against this live HubTool.
            self._safe_side_effect(
                lambda: self._service.add_tool_arg_rule(
                    tool.server_key,
                    tool.name,
                    args=dict(args),
                    tool=tool,
                    **self._profile_kwargs(),
                ),
                tool,
                what="add_tool_arg_rule",
            )
            return self._execute(tool, args, decision="approved")
        if verdict == "always_allow":
            self._safe_side_effect(
                lambda: self._service.set_tool_state(
                    tool.server_key,
                    tool.name,
                    "allow",
                    tool=tool,
                    # Task 6: persist into the ACTIVE workspace profile so
                    # the grant resolves where this provider's catalog
                    # resolves -- not silently into the default profile.
                    **self._profile_kwargs(),
                ),
                tool,
                what="set_tool_state",
            )
            return self._execute(tool, args, decision="approved")
        if verdict == "timeout":
            self._record_decision_safe(tool, decision="denied-timeout")
            return ToolResult.blocked(TIMEOUT_REFUSAL)
        if verdict == "deny":
            # TASK-294: an explicit card "Deny" gets USER provenance -- a
            # person said no to this call; the permissions were not Off.
            self._record_decision_safe(tool, decision="denied")
            return ToolResult.blocked(USER_DENY_REFUSAL)
        # An unrecognized or MISSING verdict fails closed -- but blaming the
        # user here would be the same provenance lie in the other direction:
        # nobody decided anything. Neutral copy, still a refusal -- and the
        # AUDIT record agrees with the transcript (review finding: the first
        # version recorded plain "denied" here, so Decision-filtered audit
        # views reported an explicit denial nobody made). Mirrors the
        # existing "denied-timeout" vocabulary.
        self._record_decision_safe(tool, decision="denied-unresolved")
        return ToolResult.blocked(UNRESOLVED_REFUSAL)

    def _safe_side_effect(
        self, fn: Callable[[], None], tool: HubTool, *, what: str
    ) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- a persistence failure must not block execution
            logger.warning(
                f"MCPToolProvider: {what} failed for {tool.server_key}/{tool.name}: {exc}"
            )

    def _record_decision_safe(
        self, tool: HubTool, *, decision: str, error: str | None = None
    ) -> None:
        try:
            self._service.record_tool_decision(
                tool.server_key,
                tool.name,
                decision=decision,
                initiator="agent",
                error=error,
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
                under (e.g. `"allowed"`/`"approved"`/`"approved-session"`),
                forwarded to `execute_hub_tool` and, on a bridge failure
                this method itself must record (see the discriminator
                comment below), to the best-effort audit record below.

        Returns:
            A `ToolResult`: `ok=True` with the formatted result on
            success; `ok=False` with a non-empty, length-capped `error`
            on any failure. Never raises.
        """
        future: concurrent.futures.Future | None = None
        execution_coroutine = None
        try:
            timeout = self._service._tool_call_timeout() + _RESULT_WAIT_SLACK_SECONDS
            # Task 4 (PR-T3): the same schema `tool.input_schema` the Hub
            # workbench's Test Tool form renders from -- named argument
            # NAMES only, never values, so an agent-initiated run is
            # audited with real provenance instead of the pre-Task-4
            # always-empty `argument_names: []`.
            execution_coroutine = self._service.execute_hub_tool(
                tool.server_key,
                tool.name,
                args,
                initiator="agent",
                decision=decision,
                registered_argument_names=schema_argument_names(tool.input_schema),
            )
            future = asyncio.run_coroutine_threadsafe(
                execution_coroutine,
                self._main_loop,
            )
            raw_result = future.result(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- the never-raise/never-hang contract
            if future is None and execution_coroutine is not None:
                try:
                    execution_coroutine.close()
                except Exception:
                    pass
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
            # C2: record here ONLY when `execute_hub_tool` could NOT have
            # recorded this failure itself. The real service's contract
            # (`UnifiedMCPControlPlaneService.execute_hub_tool`) records via
            # `_record_tool_execution` BEFORE every exception that
            # propagates through `future.result()` normally -- its own
            # inner `asyncio.TimeoutError` branch and its generic
            # except-and-reraise branch both record-then-raise. Recording
            # again here for those would double the audit trail for one
            # logical failure (Finding F2 originally fixed a genuine gap;
            # this discriminator fixes the over-correction). The three
            # cases where `execute_hub_tool` truly never got a chance to
            # record are:
            #   - the submit itself raised (`future is None` -- the
            #     coroutine never started, e.g. a dead/closed loop);
            #   - the OUTER slack wait timed out
            #     (`concurrent.futures.TimeoutError` -- the wedged-loop
            #     case: the coroutine hadn't finished, and may not even
            #     have reached its own inner timeout clock yet);
            #   - the future was cancelled before completing
            #     (`concurrent.futures.CancelledError`).
            if (
                future is None
                or isinstance(exc, concurrent.futures.TimeoutError)
                or isinstance(exc, concurrent.futures.CancelledError)
            ):
                self._record_decision_safe(
                    tool,
                    decision=decision,
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
