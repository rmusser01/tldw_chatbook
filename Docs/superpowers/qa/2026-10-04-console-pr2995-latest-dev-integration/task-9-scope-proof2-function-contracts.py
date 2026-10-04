from __future__ import annotations
from collections.abc import Callable
from typing import Any


def _approval_decision_fact(
    decision: object, *, unanswered: bool
) -> ApprovalDecision | None: ...


def _build_approval_payload(
    round_id: str,
    session_id: str,
    run_id: str,
    pending: "list[MCPPendingCall]",
    timeout_seconds: float,
    deadline: float | None,
    *,
    read_global_ToolExecutionPolicy: Callable[[], Any],
) -> dict[str, Any]:
    """Marshal one approval round's card payload.

    ADR-090: rows carry ``rationale`` (the model's advisory context) and
    ``description`` (the tool definition's own text, for the external
    summarizer); the payload carries a ``summary`` slot that starts ``None``
    and is filled by the advisory summarizer -- payload-carried so any
    remount re-renders it rather than depending on a live patch surviving.
    """
    ...


def _collect_mcp_pending(
    provider: MCPToolProvider, calls: list["ToolCall"]
) -> list["MCPPendingCall"]:
    """Resolve each call's MCP gate; return the subset that needs asking.

    Extracted so `build_mcp_review_hook` (MCP-only, still used directly by
    its own long-standing tests) and `build_tool_review_hook` (T6: the
    run-level hook that folds built-ins in too) share this ONE walk over
    `provider.pending_gate_for` rather than one copying the other's body.
    `None` per call means either "not an MCP call this provider owns" or
    "an MCP call whose current state doesn't need asking" -- see
    `pending_gate_for`'s own docstring for why callers do not need to
    distinguish those two cases.
    """
    ...


def _review_decision(
    row: MCPPendingCall,
    decisions: Mapping[str, str],
    verdict: str,
    *,
    allowing: tuple[str, ...],
    name_fallback: bool,
    read_global_ToolReviewDecision: Callable[[], Any],
    read_global__approval_decision_fact: Callable[[], Any],
    read_global_append_denial_reason: Callable[[], Any],
    read_global_approval_key_unanswered: Callable[[], Any],
    read_global_selected_approval_key: Callable[[], Any],
) -> ToolReviewDecision:
    """Attach an answered raw choice to the owner's unchanged verdict."""
    ...


def _sibling_approval_refusals(
    rows: Sequence[MCPPendingCall],
    decision_for: Callable[[MCPPendingCall], str | None],
    decisions: Mapping[str, str],
    allowing_for: Callable[[MCPPendingCall], tuple[str, ...]],
    record_refusal: Callable[[MCPPendingCall, bool], None],
    *,
    read_global_TIMEOUT_REFUSAL: Callable[[], Any],
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_UNRESOLVED_REFUSAL: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
) -> dict[str, ToolReviewValue]:
    """Refuse rows that would run only on a same-name sibling's approval.

    TASK-33082. A tool's stamp is name-keyed and keeps the broadest approval
    any row of that name received, so it cannot say "this call, not that
    one". A row whose own answer is missing, ``"timeout"`` or unknown would
    then run on its approved sibling's stamp. Each such row is refused here,
    by its own key, and audited through ``record_refusal``, because the
    runtime never dispatches it to the owner that would otherwise record the
    outcome. A row with no approved sibling is left alone: its name's stamp
    is not an approval, so the owner refuses and audits it at dispatch, as
    before.

    Args:
        rows: The batch's pending approval rows.
        decision_for: Resolves one row's own answer (call id first, then name).
        decisions: The approval round's answers, for the review fact.
        allowing_for: The answers that approve a given row's owner.
        record_refusal: Audits one refused row; the flag is whether its own
            answer was ``"timeout"``.

    Returns:
        Refusal verdicts keyed by call id, or by name for an id-less row.
    """
    ...


def _stamp_answer_provenance(
    stamps: dict[str, str],
    rows: Sequence[MCPPendingCall],
    decisions: Mapping[str, str],
    *,
    read_global_ApprovalDecisions: Callable[[], Any],
    read_global_approval_was_unanswered: Callable[[], Any],
    read_global_selected_approval_key: Callable[[], Any],
) -> ApprovalDecisions:
    """Keep unresolved denies attached to the selected name-scoped stamp."""
    ...


def _stamp_approval_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for an approval round: deny every undecided key."""
    ...


def _stamp_question_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for a question round: the revoked flag says it all."""
    ...


def _stamp_skill_script_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for a skill-script round: allow/remember both off."""
    ...


def approval_was_unanswered(
    row: "MCPPendingCall",
    decisions: Mapping[str, str],
    *,
    read_global_approval_key_unanswered: Callable[[], Any],
    read_global_selected_approval_key: Callable[[], Any],
) -> bool:
    """True when ``row``'s deny came from a Stop/revoke, not from the user.

    Args:
        row: The pending call whose verdict is being recorded.
        decisions: The map ``request_mcp_approvals`` returned -- an
            `ApprovalDecisions` in production, a bare dict in tests and in
            any other `request_approvals` shape (which then reports
            "answered", the pre-fix behaviour).

    Returns:
        Whether the verdict for ``row`` was defaulted by an unresolved round.
    """
    ...


def build_combined_review_hook(
    hooks: list[Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_logger: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Fan one batch through every provider's hook; merge verdict maps.

    Each hook gates only the calls its provider owns (pending_gate_for
    returns None for foreign tools), so merging is collision-free --
    except when two providers own the SAME name (not possible today:
    local names carry fs_/web_/todo_ prefixes, virtual CLI owns only
    virtual_cli, and MCP names carry mcp__*), where
    the later hook's "proceed" would simply win; both stamps are still
    applied by each provider's own hook regardless.

    I3 across providers: every hook runs even when an earlier one RAISES.
    `run_agent_loop` fails the batch OPEN on hook exception
    (agent_runtime.py:367-376), and each hook's clear-first stamp wipe is
    the only thing standing between a stale prior-turn stamp and the
    fail-open runtime handing it to `invoke()`. A naive sequential loop
    would let one hook's raising approval round trip (the documented I3
    mid-shutdown case) skip every LATER hook -- including its entry clear
    -- stranding that provider's stale stamp. So each hook is invoked
    under its own try/except and the FIRST exception is re-raised after
    all hooks have run: every provider gets its clear (and, when its own
    round trip succeeds, its fresh this-turn decisions), and the runtime
    still sees the raise and applies its fail-open policy against stamps
    that are guaranteed non-stale.

    Args:
        hooks: The per-provider review hooks to fan each batch through,
            in application order.

    Returns:
        A `review_tool_calls`-shaped callable that merges every hook's
        verdict map into one.
    """
    ...


def build_local_review_hook(
    provider: "LocalToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__APPROVAL_SCOPE_RANK: Callable[[], Any],
    read_global__APPROVING_DECISIONS: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
    read_global__sibling_approval_refusals: Callable[[], Any],
    read_global__stamp_answer_provenance: Callable[[], Any],
    read_global_approval_was_unanswered: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build this run's review_tool_calls hook for the local provider.

    Identical discipline to build_mcp_review_hook (see its docstring for
    the full rationale -- every binding point applies unchanged here):
    clear-first stamps at entry (I3: a raising approval round trip must
    never leave a stale prior-turn stamp live for the fail-open runtime
    to hand to `invoke()`), and exactly ONE approval round trip per batch.
    Name-keyed stamps keep the widest approved scope because the provider
    gate is tool-scoped; per-call refusal verdicts stop only the denied
    sibling before dispatch. Calls the provider doesn't own resolve
    `None` from `pending_gate_for` and never enter the batch.

    Args:
        provider: This run's already-composed `LocalToolProvider` (built
            by `_compose_local_provider` on the main loop before the
            run's worker thread starts).
        request_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run -- the same
            approval-card bridge the MCP hook uses; it consumes
            `MCPPendingCall` payloads regardless of origin.

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """
    ...


def build_managed_skill_promotion_review_hook(
    gate: "ManagedSkillProposalGate",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build the primary-only approval hook for read-only skill proposals."""
    ...


def build_mcp_review_hook(
    provider: MCPToolProvider,
    request_mcp_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global__collect_mcp_pending: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build this run's T4 `review_tool_calls` hook for one composed MCP provider.

    Handed to `ConsoleAgentBridge.run_reply` (P5-T6), which forwards it
    straight through to `AgentService`/`LoopDeps.review_tool_calls` (T4):
    called ONCE per turn with the full batch of tool calls about to be
    dispatched, before any of them is invoked.

    For every call in the batch, `provider.pending_gate_for(name, args)`
    resolves whether it needs human gating (`None` for both "not an MCP
    call this provider owns" and "an MCP call whose current state doesn't
    need asking" -- `invoke()` re-resolves either case for itself, so
    this hook does not need to distinguish them). When at least one call
    needs asking, this makes exactly ONE `request_mcp_approvals` round
    trip for the whole batch (never one per call) and hands the resulting
    decisions to `provider.apply_batch_decisions` -- a per-turn stamp
    every same-named call `invoke()` makes THIS turn peeks (Finding F1:
    never popped, so two calls to the same tool in one batch both see the
    approval, not just the first).

    Finding F1 also requires this hook to call
    `provider.apply_batch_decisions` on EVERY invocation, even when
    `pending` ends up empty (a turn whose calls are all non-MCP, or all
    already resolved without asking) -- passing `{}` in that case.
    `apply_batch_decisions` REPLACES the stamp set rather than merging, so
    this is what guarantees a stamp from an earlier turn can never survive
    into a later one and be misread as this turn's verdict for a
    repeated tool name.

    I3 (probe-verified): that clear happens at hook ENTRY, before
    `pending_gate_for` is even resolved and before the
    `request_mcp_approvals` round trip -- not only after a successful one.
    `request_mcp_approvals` can raise (e.g. the unguarded
    `_marshal_pending_approval` call mid-shutdown); `run_agent_loop`'s own
    hook-exception handling fails the WHOLE batch open (treats every call
    in it as `"proceed"`) when that happens. If the clear only ran after a
    successful round trip, a raise would leave THIS turn's stamp set
    exactly as the PREVIOUS turn left it -- so the fail-open runtime would
    hand `invoke()` a stale prior-turn stamp (e.g. a real `"approve_once"`)
    for a call the user never decided on this turn. Clearing first means a
    raised round trip always leaves `invoke()` with no stamp to peek,
    falling through to its own fresh gate -- which fails closed for an
    `"ask"` tool with no approval_callback wired.

    Design choice (binding, per the Phase-5 plan): this hook never
    returns a refusal string itself. Every MCP call it stamped is left to
    resolve through `invoke()`'s own gate on dispatch -- `invoke()`
    already handles every decision string uniformly (`approve_once`/
    `approve_session`/`always_allow` execute; `deny`/`timeout` refuse with
    the exact model-facing copy AND record the audit decision), so
    routing every decision through that ONE place keeps the refusal copy
    and the audit trail single-sourced instead of duplicating that logic
    here. The verdict map this hook returns therefore only ever contains
    `"proceed"` entries (for calls it gated this turn) -- purely
    documentary, since `run_agent_loop` already treats any name this hook
    doesn't mention as `"proceed"` by default; returning `{}` when nothing
    needed gating is exactly as correct as omitting entries would be.
    Non-MCP calls are untouched either way: `pending_gate_for` returns
    `None` for any name the provider doesn't own, so they never enter
    `pending` and are never mentioned in the returned map.

    Args:
        provider: This run's already-composed `MCPToolProvider` (P5-T6:
            built and `compose_catalog()`-ed by the caller on the main
            loop before the run's worker thread starts).
        request_mcp_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run -- runs on the
            agent bridge's worker thread and blocks until the batch is
            decided, cancelled, or times out (T5).

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """
    ...


def build_raw_shell_review_hook(
    provider: "RawShellToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Gate model-authored host-shell calls independently by native call id."""
    ...


def build_tool_review_hook(
    builtin_gate: "BuiltinToolGate",
    builtin_provider: "BuiltinToolProvider",
    mcp_provider: MCPToolProvider | None,
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    workspace_id: str | None,
    kill_switch: Callable[[], bool] | None,
    library_provider: Any | None,
    read_global_AGENT_LESSON_APPROVAL_REQUIRED: Callable[[], Any],
    read_global_AGENT_LESSON_DENIED: Callable[[], Any],
    read_global_AGENT_LESSON_FOREGROUND_REQUIRED: Callable[[], Any],
    read_global_Any: Callable[[], Any],
    read_global_ApprovalDecisions: Callable[[], Any],
    read_global_BUILTIN_TOOL_SERVER_KEY: Callable[[], Any],
    read_global_KILL_SWITCH_REFUSAL: Callable[[], Any],
    read_global_MCPPendingCall: Callable[[], Any],
    read_global_TOOL_DESCRIPTION_CAPTURE_CAP: Callable[[], Any],
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__APPROVAL_SCOPE_RANK: Callable[[], Any],
    read_global__APPROVING_DECISIONS: Callable[[], Any],
    read_global__collect_mcp_pending: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
    read_global__sibling_approval_refusals: Callable[[], Any],
    read_global__stamp_answer_provenance: Callable[[], Any],
    read_global_approval_effects_for_tool: Callable[[], Any],
    read_global_approval_key_unanswered: Callable[[], Any],
    read_global_approval_was_unanswered: Callable[[], Any],
    read_global_current_run_actor: Callable[[], Any],
    read_global_logger: Callable[[], Any],
    read_global_path_precheck_failed: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build THIS run's run-level `review_tool_calls` hook (P5-T6/task-545).

    TASK-631: when ``kill_switch`` reports on, EVERY call in the batch is
    refused here, without prompting -- the runtime turns any non-"proceed"
    verdict into the call's result and skips dispatch. MCP composition is
    already skipped and ``BuiltinToolGate.check`` already refuses with the
    switch on, but names neither provider claims (skills,
    ``spawn_subagent``, ``find_tools``, ``load_tools``) used to pass
    through unreviewed and RUN NORMALLY -- the switch's label promises
    "block tool calls in chat", and this hook is the one place every
    parsed call passes, so this is where the promise is kept. Read fresh
    per turn (a callable, not a bool) so flipping the switch mid-run takes
    effect on the next batch.

    Unlike `build_mcp_review_hook`, this is wired UNCONDITIONALLY -- every
    run gets one, even a user with no MCP servers configured at all --
    because built-in tools (calculator/datetime today, more later) must be
    gated regardless of whether MCP happens to be composed this turn.
    `BuiltinToolProvider.invoke` already enforces the gate as defense in
    depth, but without this hook the ONLY review a built-in call would ever
    get is that per-call fallback -- never the batched, one-card-per-turn
    review MCP calls already get, and never a chance to ask before
    dispatch for calls this hook doesn't stamp.

    Routing per call, MCP first: `mcp_provider.pending_gate_for` (when a
    provider was composed this run) is asked before the built-in provider,
    so a name that provider actually owns is never mistakenly re-resolved
    against the built-in side too. Note this hook's own precedent is the
    OPPOSITE of `console_agent_bridge._non_colliding_mcp_names`, which
    resolves a name collision the other way -- it drops the colliding MCP
    name from the run's registry so the built-in wins composition. That
    inconsistency is moot in practice: `MCP/tool_naming.py:106` always
    mints MCP tool names as `mcp__<server>__<tool>`, which can never equal
    a bare built-in name like `calculator`/`get_current_datetime`, so no
    call is ever ambiguous between the two orders. A name neither provider
    claims (a skill, `spawn_subagent`, `find_tools`, ...) passes through
    unreviewed, exactly as it does for `build_mcp_review_hook` today.

    Built-in rows use `server_key=BUILTIN_TOOL_SERVER_KEY`
    (`"agent:builtin"`), `server_label="Built-in"`, and `reason=
    "risk_floored"` when `EffectiveToolState.risk_floored` else `"ask"`
    (built-ins never set `config_changed` -- see `resolve_builtin_state`'s
    own docstring for why). Every built-in row's `path_precheck_failed`
    (TASK-1231/F3 AC2) is set via `Tools.file_operation_tools.
    path_precheck_failed`: for `read_file`/`list_directory`/`write_file`
    this pre-flights the SAME `allowed_file_roots`/`validate_path_multi`
    check `invoke()` runs at dispatch, so the approval card can warn the
    user this exact call will fail even if approved -- it never gates or
    auto-denies; `False` for every other builtin tool and every MCP row.
    Only a resolved `"ask"` state ever produces a row: `"allow"` never
    prompts, and `"deny"` is refused outright by
    `invoke()`'s own gate WITHOUT ever reaching the user -- a tool the
    operator switched Off must not appear on the approval card at all.
    Nor does an `"ask"` tool that already has a live session approval
    (`builtin_gate.is_session_approved(name)`) -- review finding 1
    (T6 review): `resolve()`/`resolve_builtin_state` read the permission
    store ONLY, never session approvals, so without this check a user who
    picked "Approve for session" on turn 1 would be re-prompted on turn 2
    even though `invoke()`'s own `check()` already honors that same
    session approval and would execute it anyway. Mirrors MCP's own
    `pending_gate_for`, which applies the identical
    `_is_session_approved_safe` skip for exactly this reason.

    `options=("approve_once", "approve_session", "deny")` -- deliberately
    excluding ONLY `"always_allow"` (verified at
    `Agents/mcp_tool_provider.py:556-564`: `always_allow` is the sole
    PERSISTENT write via `set_tool_state`; `approve_session` is an
    in-memory session cache and `deny`/`timeout` are turn-scoped refusals
    that persist nothing). `"deny"` MUST stay offered -- an earlier draft
    of this design mistakenly dropped it too, which would have made a
    built-in row impossible to refuse from the card at all (the bulk "Deny
    all" button would silently leave it on whatever the row's default
    was).

    Mirrors `build_mcp_review_hook`'s I3 clear-at-entry discipline, extended
    to the built-in side: `builtin_gate.begin_turn(run_id)` runs FIRST,
    unconditionally -- before the MCP stamp clear, before any
    `pending_gate_for`/`resolve` call, before the `request_approvals` round
    trip -- so a raising round trip can never leave a stale built-in stamp
    (or a stale cached permission payload) live for the next turn to
    consume. `mcp_provider.apply_batch_decisions(run_id, {})` follows the
    same reasoning for the MCP side, only when a provider was actually
    composed this run.

    PR2a Task 5: every one of those mutations is scoped to `run_id`, the
    second argument this hook now receives (`AgentService` binds its own
    run id into the callable it hands `LoopDeps`). The gate and the MCP
    provider are shared by a parent run and every sub-agent it spawns, so
    an unscoped clear here wipes -- and an unscoped stamp overwrites --
    verdicts another run in the tree has already been granted and has not
    yet consumed. It still clears THIS run's own previous turn, which is
    what the I3 discipline above requires.

    Exactly ONE `request_approvals` round trip is made per turn, carrying
    BOTH the MCP and built-in pending rows together -- never one call per
    owner. Decisions are then applied back to each owner separately:
    `mcp_provider.apply_batch_decisions(run_id, ...)` for MCP rows,
    `builtin_gate.stamp(run_id, name, decision)` for built-in rows. The returned
    verdict map carries "proceed" for approved calls and REFUSAL STRINGS
    for per-call denials (TASK-1861), kill-switch blocks (TASK-631), and
    calls that lack an approval of their own while a same-name sibling was
    approved (TASK-33082) -- the runtime enforces those directly, skipping
    dispatch. Approvals are
    still left to `invoke()`'s gate on dispatch, which records the audit
    decision.

    Args:
        builtin_gate: THIS run's `BuiltinToolGate` -- the SAME instance
            the run's `BuiltinToolProvider.invoke` checks, so a stamp
            written here is visible there. Two separate instances would
            mean a decision made here is invisible to `invoke()`, silently
            re-prompting (a stamp `invoke()` never sees) or failing closed
            (an approval that never reaches the gate that checks it).
        builtin_provider: THIS run's `BuiltinToolProvider` (only
            `.tool_for(name)` is used here, to resolve a `ToolCall.name`
            to the `Tool` object `builtin_gate.resolve` needs).
        mcp_provider: THIS run's already-composed `MCPToolProvider`, or
            `None` when no MCP tools should be offered this run (no
            service, kill switch on, or composition yielded nothing) --
            the entire point of this hook existing separately from
            `build_mcp_review_hook` is that built-in gating must not
            depend on this being non-`None`.
        request_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run (the name predates
            built-in gating; the method itself is owner-agnostic -- it
            only reads `MCPPendingCall` fields, never assumes MCP
            ownership).
        workspace_id: THIS run's OWN workspace id (round 1 review CRITICAL
            1) -- e.g. `self.store.session_workspace_id(session_id)` --
            threaded into every builtin file-tool row's `path_precheck_
            failed` computation via `Tools.file_operation_tools.
            path_precheck_failed`'s own `workspace_id=` parameter. Must be
            the SAME workspace id `ConsoleAgentBridge.run_reply` resolves
            for this run's real dispatch (`BuiltinToolProvider(workspace_
            id=...)`) -- otherwise the pre-flight can resolve a DIFFERENT
            workspace than the one the call will actually run against
            (e.g. whatever happens to be active in the UI for a parked
            background session), making the warning wrong in either
            direction. `None` (the default) reproduces the pre-existing
            active-workspace fallback for a caller with no session
            context at all; every caller that has a real session id MUST
            resolve and pass its workspace id.

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """
    ...


def build_virtual_cli_review_hook(
    provider: "VirtualCliProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
    read_global_append_denial_reason: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Gate each selected virtual command while exposing one model tool.

    Approval rows are command-specific Hub entries but verdict stamps are
    keyed by native call id, so multiple ``virtual_cli`` calls in one model
    response remain independently addressable.
    """
    ...


def _head_round_payload_locked(
    store: dict[str, dict[str, Any]], session_id: str | None
) -> dict[str, Any] | None:
    """The session's oldest-armed payload. Caller holds the lock."""
    ...
