from __future__ import annotations


def approval_was_unanswered(
    row: "MCPPendingCall", decisions: Mapping[str, str]
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
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        approval_was_unanswered as _owned_approval_was_unanswered,
    )

    return _owned_approval_was_unanswered(
        row,
        decisions,
        read_global_approval_key_unanswered=lambda: approval_key_unanswered,
        read_global_selected_approval_key=lambda: selected_approval_key,
    )


def _approval_decision_fact(
    decision: object, *, unanswered: bool = False
) -> ApprovalDecision | None:
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _approval_decision_fact as _owned__approval_decision_fact,
    )

    return _owned__approval_decision_fact(decision, unanswered=unanswered)


def _review_decision(
    row: MCPPendingCall,
    decisions: Mapping[str, str],
    verdict: str,
    *,
    allowing: tuple[str, ...] = _APPROVING_DECISIONS,
    name_fallback: bool = True,
) -> ToolReviewDecision:
    """Attach an answered raw choice to the owner's unchanged verdict."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _review_decision as _owned__review_decision,
    )

    return _owned__review_decision(
        row,
        decisions,
        verdict,
        allowing=allowing,
        name_fallback=name_fallback,
        read_global_ToolReviewDecision=lambda: ToolReviewDecision,
        read_global__approval_decision_fact=lambda: _approval_decision_fact,
        read_global_append_denial_reason=lambda: append_denial_reason,
        read_global_approval_key_unanswered=lambda: approval_key_unanswered,
        read_global_selected_approval_key=lambda: selected_approval_key,
    )


def _stamp_answer_provenance(
    stamps: dict[str, str], rows: Sequence[MCPPendingCall], decisions: Mapping[str, str]
) -> ApprovalDecisions:
    """Keep unresolved denies attached to the selected name-scoped stamp."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _stamp_answer_provenance as _owned__stamp_answer_provenance,
    )

    return _owned__stamp_answer_provenance(
        stamps,
        rows,
        decisions,
        read_global_ApprovalDecisions=lambda: ApprovalDecisions,
        read_global_approval_was_unanswered=lambda: approval_was_unanswered,
        read_global_selected_approval_key=lambda: selected_approval_key,
    )


def _sibling_approval_refusals(
    rows: Sequence[MCPPendingCall],
    decision_for: Callable[[MCPPendingCall], str | None],
    decisions: Mapping[str, str],
    allowing_for: Callable[[MCPPendingCall], tuple[str, ...]],
    record_refusal: Callable[[MCPPendingCall, bool], None],
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
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _sibling_approval_refusals as _owned__sibling_approval_refusals,
    )

    return _owned__sibling_approval_refusals(
        rows,
        decision_for,
        decisions,
        allowing_for,
        record_refusal,
        read_global_TIMEOUT_REFUSAL=lambda: TIMEOUT_REFUSAL,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_UNRESOLVED_REFUSAL=lambda: UNRESOLVED_REFUSAL,
        read_global__review_decision=lambda: _review_decision,
    )


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
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _collect_mcp_pending as _owned__collect_mcp_pending,
    )

    return _owned__collect_mcp_pending(provider, calls)


def _build_approval_payload(
    round_id: str,
    session_id: str,
    run_id: str,
    pending: "list[MCPPendingCall]",
    timeout_seconds: float,
    deadline: float | None,
) -> dict[str, Any]:
    """Marshal one approval round's card payload.

    ADR-090: rows carry ``rationale`` (the model's advisory context) and
    ``description`` (the tool definition's own text, for the external
    summarizer); the payload carries a ``summary`` slot that starts ``None``
    and is filled by the advisory summarizer -- payload-carried so any
    remount re-renders it rather than depending on a live patch surviving.
    """
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _build_approval_payload as _owned__build_approval_payload,
    )

    return _owned__build_approval_payload(
        round_id,
        session_id,
        run_id,
        pending,
        timeout_seconds,
        deadline,
        read_global_ToolExecutionPolicy=lambda: ToolExecutionPolicy,
    )


def build_mcp_review_hook(
    provider: MCPToolProvider,
    request_mcp_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
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
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_mcp_review_hook as _owned_build_mcp_review_hook,
    )

    return _owned_build_mcp_review_hook(
        provider,
        request_mcp_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global__collect_mcp_pending=lambda: _collect_mcp_pending,
        read_global__review_decision=lambda: _review_decision,
    )


def build_tool_review_hook(
    builtin_gate: "BuiltinToolGate",
    builtin_provider: "BuiltinToolProvider",
    mcp_provider: MCPToolProvider | None,
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    workspace_id: str | None = None,
    kill_switch: Callable[[], bool] | None = None,
    library_provider: Any | None = None,
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
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_tool_review_hook as _owned_build_tool_review_hook,
    )

    return _owned_build_tool_review_hook(
        builtin_gate,
        builtin_provider,
        mcp_provider,
        request_approvals,
        workspace_id=workspace_id,
        kill_switch=kill_switch,
        library_provider=library_provider,
        read_global_AGENT_LESSON_APPROVAL_REQUIRED=lambda: (
            AGENT_LESSON_APPROVAL_REQUIRED
        ),
        read_global_AGENT_LESSON_DENIED=lambda: AGENT_LESSON_DENIED,
        read_global_AGENT_LESSON_FOREGROUND_REQUIRED=lambda: (
            AGENT_LESSON_FOREGROUND_REQUIRED
        ),
        read_global_Any=lambda: Any,
        read_global_ApprovalDecisions=lambda: ApprovalDecisions,
        read_global_BUILTIN_TOOL_SERVER_KEY=lambda: BUILTIN_TOOL_SERVER_KEY,
        read_global_KILL_SWITCH_REFUSAL=lambda: KILL_SWITCH_REFUSAL,
        read_global_MCPPendingCall=lambda: MCPPendingCall,
        read_global_TOOL_DESCRIPTION_CAPTURE_CAP=lambda: TOOL_DESCRIPTION_CAPTURE_CAP,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_USER_DENIED_REFUSAL=lambda: USER_DENIED_REFUSAL,
        read_global__APPROVAL_SCOPE_RANK=lambda: _APPROVAL_SCOPE_RANK,
        read_global__APPROVING_DECISIONS=lambda: _APPROVING_DECISIONS,
        read_global__collect_mcp_pending=lambda: _collect_mcp_pending,
        read_global__review_decision=lambda: _review_decision,
        read_global__sibling_approval_refusals=lambda: _sibling_approval_refusals,
        read_global__stamp_answer_provenance=lambda: _stamp_answer_provenance,
        read_global_approval_effects_for_tool=lambda: approval_effects_for_tool,
        read_global_approval_key_unanswered=lambda: approval_key_unanswered,
        read_global_approval_was_unanswered=lambda: approval_was_unanswered,
        read_global_current_run_actor=lambda: current_run_actor,
        read_global_logger=lambda: logger,
        read_global_path_precheck_failed=lambda: path_precheck_failed,
    )


def build_local_review_hook(
    provider: "LocalToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
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
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_local_review_hook as _owned_build_local_review_hook,
    )

    return _owned_build_local_review_hook(
        provider,
        request_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_USER_DENIED_REFUSAL=lambda: USER_DENIED_REFUSAL,
        read_global__APPROVAL_SCOPE_RANK=lambda: _APPROVAL_SCOPE_RANK,
        read_global__APPROVING_DECISIONS=lambda: _APPROVING_DECISIONS,
        read_global__review_decision=lambda: _review_decision,
        read_global__sibling_approval_refusals=lambda: _sibling_approval_refusals,
        read_global__stamp_answer_provenance=lambda: _stamp_answer_provenance,
        read_global_approval_was_unanswered=lambda: approval_was_unanswered,
    )


def build_managed_skill_promotion_review_hook(
    gate: "ManagedSkillProposalGate",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build the primary-only approval hook for read-only skill proposals."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_managed_skill_promotion_review_hook as _owned_build_managed_skill_promotion_review_hook,
    )

    return _owned_build_managed_skill_promotion_review_hook(
        gate,
        request_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_USER_DENIED_REFUSAL=lambda: USER_DENIED_REFUSAL,
        read_global__review_decision=lambda: _review_decision,
    )


def build_virtual_cli_review_hook(
    provider: "VirtualCliProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Gate each selected virtual command while exposing one model tool.

    Approval rows are command-specific Hub entries but verdict stamps are
    keyed by native call id, so multiple ``virtual_cli`` calls in one model
    response remain independently addressable.
    """
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_virtual_cli_review_hook as _owned_build_virtual_cli_review_hook,
    )

    return _owned_build_virtual_cli_review_hook(
        provider,
        request_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global__review_decision=lambda: _review_decision,
        read_global_append_denial_reason=lambda: append_denial_reason,
    )


def build_raw_shell_review_hook(
    provider: "RawShellToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Gate model-authored host-shell calls independently by native call id."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_raw_shell_review_hook as _owned_build_raw_shell_review_hook,
    )

    return _owned_build_raw_shell_review_hook(
        provider,
        request_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_USER_DENIED_REFUSAL=lambda: USER_DENIED_REFUSAL,
        read_global__review_decision=lambda: _review_decision,
    )


def build_combined_review_hook(
    hooks: list[Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]],
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
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_combined_review_hook as _owned_build_combined_review_hook,
    )

    return _owned_build_combined_review_hook(
        hooks,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_logger=lambda: logger,
    )


def _stamp_approval_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for an approval round: deny every undecided key."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _stamp_approval_round_closed as _owned__stamp_approval_round_closed,
    )

    return _owned__stamp_approval_round_closed(state)


def _stamp_skill_script_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for a skill-script round: allow/remember both off."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _stamp_skill_script_round_closed as _owned__stamp_skill_script_round_closed,
    )

    return _owned__stamp_skill_script_round_closed(state)


def _stamp_question_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for a question round: the revoked flag says it all."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _stamp_question_round_closed as _owned__stamp_question_round_closed,
    )

    return _owned__stamp_question_round_closed(state)


class ConsoleChatController:
    def __init__(self):
        """Scope fragment replacing the original host import and construction only."""

        def write_controller__pending_decision_order(value):
            """Write through to the controller-owned _pending_decision_order value."""
            self._pending_decision_order = value

        from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

        self._interrupt_host = InterruptRoundHost(
            read_controller__active_assistant_message_ids=lambda: (
                self._active_assistant_message_ids
            ),
            read_controller__advance_lifecycle_revision=lambda: (
                self._advance_lifecycle_revision
            ),
            read_controller__agent_bridge=lambda: getattr(self, "_agent_bridge", None),
            read_controller__announce_detached_approval=lambda: (
                self._announce_detached_approval
            ),
            read_controller__announce_hidden_decision=lambda: (
                self._announce_hidden_decision
            ),
            read_controller__announced_pending_decision_ids=lambda: (
                self._announced_pending_decision_ids
            ),
            read_controller__answerable_decision_by_session=lambda: (
                self._answerable_decision_by_session
            ),
            read_controller__approval_view_is_detached=lambda: (
                self._approval_view_is_detached
            ),
            read_controller__bind_round_cancel_signal=lambda: (
                self._bind_round_cancel_signal
            ),
            read_controller__bind_visit_cancel_signal=lambda: (
                self._bind_visit_cancel_signal
            ),
            read_controller__buddy_sink=lambda: self._buddy_sink,
            read_controller__chat_create_session_grants=lambda: (
                self._chat_create_session_grants
            ),
            read_controller__chat_creation_record_locked=lambda: (
                self._chat_creation_record_locked
            ),
            read_controller__chat_creation_records=lambda: self._chat_creation_records,
            read_controller__chat_creation_revoked_runs=lambda: (
                self._chat_creation_revoked_runs
            ),
            read_controller__chat_start=lambda: self._chat_start,
            read_controller__console_answerable_decision_by_session=lambda: (
                self._console_answerable_decision_by_session
            ),
            read_controller__deliver_permission_summary=lambda: (
                self._deliver_permission_summary
            ),
            read_controller__disposed=lambda: self._disposed,
            read_controller__enrich_chat_create_confirm_payload=lambda: (
                self._enrich_chat_create_confirm_payload
            ),
            read_controller__expire_answerable_decision_if_due=lambda: (
                self._expire_answerable_decision_if_due
            ),
            read_controller__forget_hidden_decision=lambda: (
                self._forget_hidden_decision
            ),
            read_controller__head_round_payload=lambda: self._head_round_payload,
            read_controller__interrupt_bell_enabled=lambda: (
                self._interrupt_bell_enabled
            ),
            read_controller__is_session_cancelled=lambda: self._is_session_cancelled,
            read_controller__marshal_pending_chat_create=lambda: (
                self._marshal_pending_chat_create
            ),
            read_controller__marshal_pending_decision_projection=lambda: (
                self._marshal_pending_decision_projection
            ),
            read_controller__maybe_fire_permission_summary=lambda: (
                self._maybe_fire_permission_summary
            ),
            read_controller__mutate_exact_pending_decision=lambda: (
                self._mutate_exact_pending_decision
            ),
            read_controller__notify_run_hook_approval=lambda: (
                self._notify_run_hook_approval
            ),
            read_controller__observe_chat_creation_record=lambda: (
                self._observe_chat_creation_record
            ),
            read_controller__park_round_payload=lambda: self._park_round_payload,
            read_controller__parked_chat_create_payloads=lambda: (
                self._parked_chat_create_payloads
            ),
            read_controller__pause_answerable_decision=lambda: (
                self._pause_answerable_decision
            ),
            read_controller__pause_pending_decision_state_locked=lambda: (
                self._pause_pending_decision_state_locked
            ),
            read_controller__pending_approvals=lambda: self._pending_approvals,
            read_controller__pending_chat_create_lock=lambda: (
                self._pending_chat_create_lock
            ),
            read_controller__pending_chat_create_rounds=lambda: (
                self._pending_chat_create_rounds
            ),
            read_controller__pending_decision_order=lambda: (
                self._pending_decision_order
            ),
            read_controller__pending_decision_payloads_locked=lambda: (
                self._pending_decision_payloads_locked
            ),
            read_controller__pending_round_kinds=lambda: self._pending_round_kinds,
            read_controller__pending_round_states_snapshot=lambda: (
                self._pending_round_states_snapshot
            ),
            read_controller__permission_summary_worker=lambda: (
                self._permission_summary_worker
            ),
            read_controller__provider_messages_for_session=lambda: (
                self._provider_messages_for_session
            ),
            read_controller__publish_console_attention_change=lambda: (
                self._publish_console_attention_change
            ),
            read_controller__publish_pending_decision=lambda: (
                self._publish_pending_decision
            ),
            read_controller__question_bounces=lambda: self._question_bounces,
            read_controller__raw_shell_providers=lambda: getattr(
                self, "_raw_shell_providers", ()
            ),
            read_controller__record_cancelled_approval_decisions=lambda: (
                self._record_cancelled_approval_decisions
            ),
            read_controller__refresh_answerable_decision=lambda: (
                self._refresh_answerable_decision
            ),
            read_controller__remount_head=lambda: self._remount_head,
            read_controller__remount_parked_chat_create=lambda: (
                self._remount_parked_chat_create
            ),
            read_controller__remount_parked_skill_install=lambda: (
                self._remount_parked_skill_install
            ),
            read_controller__remount_parked_skill_script=lambda: (
                self._remount_parked_skill_script
            ),
            read_controller__remount_session_kinds=lambda: self._remount_session_kinds,
            read_controller__reproject_pending_decision_for_session=lambda: (
                self._reproject_pending_decision_for_session
            ),
            read_controller__resolve_ask_user_timeout_seconds=lambda: (
                self._resolve_ask_user_timeout_seconds
            ),
            read_controller__resolve_mcp_approval_timeout_seconds=lambda: (
                self._resolve_mcp_approval_timeout_seconds
            ),
            read_controller__revoke_chat_create_rounds=lambda: (
                self._revoke_chat_create_rounds
            ),
            read_controller__run_hooks_engine=lambda: self._run_hooks_engine,
            read_controller__session_close_generations=lambda: (
                self._session_close_generations
            ),
            read_controller__settle_pending_decision_timeout_locked=lambda: (
                self._settle_pending_decision_timeout_locked
            ),
            read_controller__summary_tail_messages=lambda: self._summary_tail_messages,
            read_controller__unpark_round_payload=lambda: self._unpark_round_payload,
            read_controller_add_pending_round=lambda: self.add_pending_round,
            read_controller_announce_hidden_decision=lambda: (
                self.announce_hidden_decision
            ),
            read_controller_app=lambda: self.app,
            read_controller_ask_user_timeout_seconds=lambda: (
                self.ask_user_timeout_seconds
            ),
            read_controller_chat_create_confirm_timeout_seconds=lambda: (
                self.chat_create_confirm_timeout_seconds
            ),
            read_controller_decision_monotonic_clock=lambda: (
                self.decision_monotonic_clock
            ),
            read_controller_discard_pending_round=lambda: self.discard_pending_round,
            read_controller_expire_pending_decisions=lambda: (
                self.expire_pending_decisions
            ),
            read_controller_mcp_approval_timeout_seconds=lambda: (
                self.mcp_approval_timeout_seconds
            ),
            read_controller_on_console_attention_changed=lambda: (
                self.on_console_attention_changed
            ),
            read_controller_on_pending_rounds_changed=lambda: (
                self.on_pending_rounds_changed
            ),
            read_controller_park_pending_approval=lambda: self.park_pending_approval,
            read_controller_pending_decision_projection=lambda: (
                self.pending_decision_projection
            ),
            read_controller_project_pending_decision_for_active_session=lambda: (
                self.project_pending_decision_for_active_session
            ),
            read_controller_remount_pending_approval_for_active_session=lambda: (
                self.remount_pending_approval_for_active_session
            ),
            read_controller_set_answerable_decision=lambda: (
                self.set_answerable_decision
            ),
            read_controller_set_pending_approval=lambda: self.set_pending_approval,
            read_controller_set_pending_chat_create=lambda: (
                self.set_pending_chat_create
            ),
            read_controller_set_pending_decision=lambda: self.set_pending_decision,
            read_controller_set_pending_question=lambda: self.set_pending_question,
            read_controller_set_pending_skill_install=lambda: (
                self.set_pending_skill_install
            ),
            read_controller_set_pending_skill_script=lambda: (
                self.set_pending_skill_script
            ),
            read_controller_set_pending_worktree_merge=lambda: (
                self.set_pending_worktree_merge
            ),
            read_controller_set_task_panel=lambda: self.set_task_panel,
            read_controller_skill_install_confirm_timeout_seconds=lambda: (
                self.skill_install_confirm_timeout_seconds
            ),
            read_controller_skill_script_confirm_timeout_seconds=lambda: (
                self.skill_script_confirm_timeout_seconds
            ),
            read_controller_store=lambda: self.store,
            read_controller_update_pending_approval_summary=lambda: (
                self.update_pending_approval_summary
            ),
            read_controller_worktree_merge_confirm_timeout_seconds=lambda: (
                self.worktree_merge_confirm_timeout_seconds
            ),
            write_controller__pending_decision_order=write_controller__pending_decision_order,
            read_global_ASK_USER_TIMEOUT_ENV_VAR=lambda: ASK_USER_TIMEOUT_ENV_VAR,
            read_global_Any=lambda: Any,
            read_global_ApprovalDecisions=lambda: ApprovalDecisions,
            read_global_CONSOLE_PENDING_APPROVAL_KIND=lambda: (
                CONSOLE_PENDING_APPROVAL_KIND
            ),
            read_global_CONSOLE_PENDING_CHAT_CREATE_KIND=lambda: (
                CONSOLE_PENDING_CHAT_CREATE_KIND
            ),
            read_global_ConsolePendingDecisionProjection=lambda: (
                ConsolePendingDecisionProjection
            ),
            read_global_INTERRUPT_BELL_ENV_VAR=lambda: INTERRUPT_BELL_ENV_VAR,
            read_global_Mapping=lambda: Mapping,
            read_global_ToolExecutionPolicy=lambda: ToolExecutionPolicy,
            read_global_UNRESOLVED_DENIED_DECISION=lambda: UNRESOLVED_DENIED_DECISION,
            read_global__ChatCreationToken=lambda: _ChatCreationToken,
            read_global__DEFAULT_ASK_USER_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_ASK_USER_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS
            ),
            read_global__LEGACY_PENDING_APPROVAL_ROUND_ID=lambda: (
                _LEGACY_PENDING_APPROVAL_ROUND_ID
            ),
            read_global__MAX_TRACKED_QUESTION_BOUNCE_RUNS=lambda: (
                _MAX_TRACKED_QUESTION_BOUNCE_RUNS
            ),
            read_global__MCP_APPROVAL_POLL_SECONDS=lambda: _MCP_APPROVAL_POLL_SECONDS,
            read_global__REVOCATION_STAMPS=lambda: _REVOCATION_STAMPS,
            read_global__bool_or_none=lambda: _bool_or_none,
            read_global__build_approval_payload=lambda: _build_approval_payload,
            read_global__normalize_world_info_history=lambda: (
                _normalize_world_info_history
            ),
            read_global_contextlib=lambda: contextlib,
            read_global_current_run_actor=lambda: current_run_actor,
            read_global_current_run_id=lambda: current_run_id,
            read_global_escape_markup=lambda: escape_markup,
            read_global_get_cli_setting=lambda: get_cli_setting,
            read_global_get_runtime_config_snapshot=lambda: get_runtime_config_snapshot,
            read_global_logger=lambda: logger,
            read_global_os=lambda: os,
            read_global_threading=lambda: threading,
            read_global_time=lambda: time,
            read_global_uuid4=lambda: uuid4,
        )
        self._interrupt_host.after_remount["approval"] = (
            self._maybe_fire_permission_summary
        )

    def add_pending_round(
        self, session_id: str, round_id: str, kind: str = CONSOLE_PENDING_APPROVAL_KIND
    ) -> None:
        """Register ``round_id`` as an outstanding approval-like round for ``session_id``.

        TASK-1050 (Defect A): the fleet-visible pending-approval badge used
        to be a single boolean per session (``_pending_approvals`` as a
        plain ``set[str]``, flipped by the now-deprecated ``set_run_
        pending_approval``) shared by THREE independent bridges -- MCP tool
        approvals, skill-install confirms, and skill-script confirms. Any
        one bridge's teardown cleared the badge for its own session_id
        regardless of whether a SIBLING round (same bridge or a different
        one) was still outstanding for that same session, so the badge
        could go dark while a live confirm was still waiting on the user.

        ``_pending_approvals`` is now keyed by session id to the SET of
        round ids currently outstanding for it -- a session reads as
        "pending" (``run_marker_for``/``fleet_summary_counts``) iff that
        set is non-empty. Idempotent: adding an already-registered
        ``round_id`` again is a no-op (set semantics), so a caller never
        needs to check first.

        Every genuine bridge round already mints a fresh ``uuid4()`` round/
        request id before arming (``request_mcp_approvals``'s ``round_id``,
        ``request_skill_install_confirm``'s/``request_skill_script_
        confirm``'s ``request_id``) -- this is the id each bridge now
        passes here instead of the old boolean.

        Args:
            session_id: The session the round belongs to.
            round_id: The round's own unique id (a real bridge round id, or
                the reserved ``_LEGACY_PENDING_APPROVAL_ROUND_ID`` sentinel
                -- see ``set_run_pending_approval``).
            kind: Which interrupt kind is waiting -- a
                registered interrupt key (``approval``, ``question``,
                ``skill_install``, ``skill_script``, ``worktree_merge``, or
                the standalone ``chat_create`` confirmation). Qodo #4: the badge and
                lifecycle do not care, but the run chip and activity line do
                -- they used to translate this registry's generic "something
                is pending" into "Waiting for your approval" even for a
                question. Defaults to ``approval``, which is what every
                caller without a kind of its own (the deprecated boolean
                shim, direct test drives) has always meant.
        """
        return self._interrupt_host.add_pending_round(session_id, round_id, kind)

    def discard_pending_round(self, session_id: str, round_id: str) -> None:
        """Clear ``round_id`` from ``session_id``'s outstanding approval-like rounds.

        TASK-1050 (Defect A) counterpart to ``add_pending_round``: discards
        only THIS round's id from the session's round-id set. The fleet
        badge (``run_marker_for``) clears only once that set is empty --
        i.e. once every bridge round for the session has resolved, not just
        this one. Idempotent: discarding an id that was never added (or was
        already discarded) is a safe no-op, and discarding the SAME id
        twice never double-decrements anything (set semantics -- there is
        nothing to corrupt).

        Args:
            session_id: The session the round belongs to.
            round_id: The round's own unique id, as passed to the matching
                ``add_pending_round`` call.
        """
        return self._interrupt_host.discard_pending_round(session_id, round_id)

    def _publish_console_attention_change(self) -> None:
        """Best-effort notification that the runtime should re-derive attention."""
        return self._interrupt_host._publish_console_attention_change()

    def has_pending_approval_round(self, session_id: str) -> bool:
        """Return whether ``session_id`` currently has ANY outstanding approval-like round.

        TASK-1050: exposed so a caller that lacks a round id of its own
        (see ``set_run_pending_approval``'s docstring) can check whether a
        REAL round is already registered before redundantly stamping the
        deprecated boolean shim -- ``ChatScreen._park_console_approval`` is
        the one production caller that needs this (its owning bridge always
        registers the real round id via ``add_pending_round`` moments
        before invoking the park callback, so by the time this runs, the
        real round is normally already present).

        Args:
            session_id: The session to check.

        Returns:
            ``True`` iff at least one round id is currently registered for
            ``session_id``.
        """
        return self._interrupt_host.has_pending_approval_round(session_id)

    def pending_round_kinds(self, session_id: str) -> frozenset[str]:
        """Return the KINDS of ``session_id``'s outstanding interrupt rounds.

        Qodo #4: ``has_pending_approval_round`` answers "is anything waiting
        on the user", which is the right question for the badge and the
        lifecycle but the wrong one for copy -- the shared registry holds
        questions, skill-install/skill-script confirms and worktree-merge
        confirms as well as MCP approvals, and translating the generic
        predicate into "Waiting for your approval" mislabels the other four
        (and can disagree with the inspector, which counts mounted approval
        cards only).

        Args:
            session_id: The session to read.

        Returns:
            Every distinct registered kind outstanding for ``session_id``,
            including standalone chat creation, empty when nothing is.
        """
        return self._interrupt_host.pending_round_kinds(session_id)

    def pending_round_count(
        self, session_id: str, *, kind: str = CONSOLE_PENDING_APPROVAL_KIND
    ) -> int:
        """Count one session's outstanding rounds of the requested kind.

        Args:
            session_id: The owning session to inspect.
            kind: Interrupt kind to count, including queued or hidden rounds.

        Returns:
            Number of registered rounds of this kind for the session.
        """
        return self._interrupt_host.pending_round_count(session_id, kind=kind)

    def set_run_pending_approval(self, session_id: str, pending: bool) -> None:
        """DEPRECATED boolean shim -- prefer ``add_pending_round``/``discard_pending_round``.

        Parallel-agents spec §6 (Task 7 stores/exposes the flag; Task 9
        wired the approval paths -- MCP batch approvals, skill-install/
        script confirms -- that originally called this). TASK-1050 (Defect
        A) migrated all three bridges to the round-keyed ``add_pending_
        round``/``discard_pending_round`` instead, since a plain boolean
        cannot represent "N independent rounds outstanding for one
        session" without one clobbering another's clear.

        This shim survives for the ONE remaining caller genuinely without a
        round id of its own: ``ChatScreen._park_console_approval`` (wired
        as ``park_pending_approval``), whose own public contract is a
        single-arg ``Callable[[str], None]`` with no room for a round id --
        changing that would ripple into every test that wires ``park_
        pending_approval = some_list.append`` -- and it is ALSO used
        directly, standalone, by tests exercising the marker/badge
        lifecycle without a live round (mirrors how those tests already
        drive other controller seams directly).

        Internally represented as the reserved
        ``_LEGACY_PENDING_APPROVAL_ROUND_ID`` sentinel round id, so it
        composes safely alongside real round ids in the same per-session
        set -- ``pending=True`` adds the sentinel, ``pending=False``
        discards ONLY the sentinel (a real round registered separately via
        ``add_pending_round`` is untouched either way). Because of this, a
        caller that calls this with ``pending=True`` while a REAL round is
        ALREADY registered for the session adds a harmless, redundant
        no-op-visible entry -- but that same caller must not rely on this
        call's own ``pending=False`` (or a real round's ``discard_pending_
        round``) to fully clear the badge on its own; whichever one runs
        last is the one that actually clears it. ``ChatScreen._park_
        console_approval`` avoids this ambiguity by checking ``has_
        pending_approval_round`` first and only falling back to this shim
        when no real round is registered yet.

        Args:
            session_id: The session whose pending-approval flag to update.
            pending: ``True`` to mark the session as awaiting a decision,
                ``False`` to clear it.
        """
        return self._interrupt_host.set_run_pending_approval(session_id, pending)

    def request_mcp_approvals(
        self, pending: list[MCPPendingCall], *, session_id: str | None = None
    ) -> dict[str, str]:
        """Bridge one batch of pending tool-approval rows to the Console UI and back.

        TASK-630: OWNER-AGNOSTIC, despite the legacy ``mcp`` in the name.
        Since TASK-545/P1's run-level ``build_tool_review_hook``, the rows
        handed here may come from MCP tools OR from built-in agent-runtime
        tools (``server_key="agent:builtin"``); every row is marshalled to
        the same ``ChatApprovalCard`` and resolved through the same
        Event-polling loop below. There is no separate approval path for
        built-ins -- a reader assuming "MCP-only" here would go looking for
        one that does not exist. (The name is kept: it is the wire between
        this method, ``resolve_pending_approval``, and the round-id
        plumbing, and renaming it is churn without a defect.)

        WORKER THREAD. Bound (via a ``functools.partial`` binding this
        run's ``session_id``, Task 9) as ``MCPToolProvider``'s
        ``approval_callback`` and ``build_tool_review_hook``'s
        ``request_approvals``, so this runs on the agent bridge's
        background OS thread (the ``asyncio.to_thread`` call inside
        ``_run_agent_reply``) -- it must never touch a widget directly,
        only through ``self.app.call_from_thread``.

        Builds a fresh ``threading.Event`` + shared decisions dict (stored
        under this round's own entry in ``_pending_approval_rounds``, keyed
        by a freshly minted ``round_id`` -- see that map's own docstring
        for why a single shared slot, or a slot keyed by session id alone,
        could not survive concurrent sessions or same-session round
        replacement). Either MOUNTS the card immediately (``session_id`` is
        the currently ACTIVE/viewed session, or unknown -- legacy
        no-session callers keep the pre-Task-9 always-mount behavior) or
        PARKS it (``session_id`` is a DIFFERENT, background session --
        Task 9: the retained ``payload`` goes into
        ``_parked_approval_payloads`` for ``switch_session`` to mount
        later, while the controller raises one sanitized app-wide notice
        for this exact stable decision id instead of touching a screen-owned
        parking hook). PR0
        adds a third case: an ACTIVE-session round that is not its
        session's FIFO head neither mounts nor parks -- an older sibling
        still owns the card, and this round's payload is retained under
        its own ``round_id`` until that sibling's teardown promotes it.
        Either way it then polls ``event.wait(1.0)`` re-checking this run's OWN
        cancel signal (``_is_session_cancelled``) and -- only when a
        POSITIVE timeout is configured (ADR-067: the default is 0 = none)
        -- a deadline, every second until one of three things happens: the
        user submits a decision (``resolve_pending_approval``, called from
        the UI thread once the card's own stamped ``round_id`` is delivered
        back, sets the Event -- Fix round 1: NOT "whichever round belongs
        to the active session", see ``resolve_pending_approval``'s own
        docstring for why that was a real cross-session misattribution
        hazard), the run is cancelled/torn down (``_is_session_cancelled``
        -- F5 fix, Qodo wave: this round's OWN cancel event, or real
        process teardown via ``_shutdown_requested``, never any OTHER
        session's bare Stop -- see that method's own docstring), or the
        configured approval timeout elapses. With no deadline armed the
        round simply waits for one of the first two, however long the
        human takes -- the wait is marked in ``Agents.human_input_wait``
        so a per-call wrapper hosting it pauses its ceiling. Whichever
        addressable verdict key (native ``call_id`` when present, otherwise
        ``llm_name``) never received an explicit decision by then
        fails closed to ``"deny"``
        (cancellation) or ``"timeout"`` (deadline) -- see
        ``MCPToolProvider._apply_verdict`` for how each decision string is
        consumed. The mounted card (if any) is always cleared afterwards
        (``finally``), regardless of outcome -- but ONLY if this round's
        session is STILL the active one at that moment, so a background
        round resolving (timeout/cancel) while some OTHER session's card is
        showing never clobbers it.

        Args:
            pending: One turn's pending tool calls awaiting approval. Native
                calls use their call id as the verdict key; id-less fence
                calls sharing a name share one name-keyed verdict.
            session_id: The run's OWNING session (Task 3 threads it through
                ``_run_agent_reply``). ``None`` preserves every pre-Task-9
                call site's behavior (always mounts against whatever
                session is active at ROUND-key time; no parking).

        Returns:
            An `ApprovalDecisions` (a plain verdict `dict` carrying
            ``unresolved_keys``) holding a decision string
            (``approve_once``/``approve_session``/``always_allow``/
            ``deny``/``timeout``) for every addressable call-id-or-name
            verdict key in ``pending``. Keys listed in ``unresolved_keys``
            hold the fail-closed ``"deny"`` default of a round nobody
            answered, not a user refusal -- see that class.
        """
        return self._interrupt_host.request_mcp_approvals(
            pending, session_id=session_id
        )

    def _record_cancelled_approval_decisions(
        self, keys: list[str], call_by_key: dict[str, "MCPPendingCall"]
    ) -> None:
        """Best-effort audit log for calls denied by a stop/unmount mid-approval.

        Finding I3: see the cancellation branch's own comment in
        ``request_mcp_approvals`` for why this direct call is necessary --
        `MCPToolProvider._record_decision_safe` (the normal recording
        path) is never reached for these calls, since `run_agent_loop`
        cancels the whole turn before dispatching any of them. Reached via
        `self.app.unified_mcp_service` (the same object
        `_compose_mcp_provider` built this run's `MCPToolProvider` from --
        see that method), never raises: a missing app/service, or the
        service lacking `record_tool_decision`, is a silent no-op, and any
        exception the real call raises is logged and swallowed, mirroring
        `MCPToolProvider._record_decision_safe`'s own never-raise
        contract.
        """
        return self._interrupt_host._record_cancelled_approval_decisions(
            keys, call_by_key
        )

    def _marshal_pending_approval(
        self, payload: dict[str, Any] | None, *, fire_summary: bool = True
    ) -> None:
        """Push ``payload`` (or clear it) onto the UI thread, if wired.

        Args:
            payload: The approval payload dict, or ``None`` to clear.
            fire_summary: Whether to run the ADR-090 advisory-summary
                trigger check after delivery. The arm-time head-mount site
                passes ``False`` and fires the check itself inside the
                ``use_human_input_wait`` mark, so the hook's config read
                cannot sit between payload delivery and the wait mark.
        """
        return self._interrupt_host._marshal_pending_approval(
            payload, fire_summary=fire_summary
        )

    def _maybe_fire_permission_summary(self, payload: dict[str, Any]) -> None:
        """Fire the external summarizer once per round, if configured.

        ADR-090 trigger: ``fallback`` only when some pending row lacks a
        rationale, ``always`` for every round with rows. One call per
        ``round_id`` -- no-call outcomes also consume the once-flag, so
        exactly one trigger check runs per round no matter how many times
        it mounts. Called from EVERY path that marshals a stored approval
        payload to the UI: ``_marshal_pending_approval`` (arm-time head
        mount), the session-activation mounts (``new_session``/
        ``switch_session``/``close_session`` neighbor activation,
        ``remount_pending_approval_for_active_session`` headless attach)
        and ``_remount_head`` (sibling promotion on resolve/revoke) -- so
        a round that armed while parked fires when its card actually
        mounts. Never raises.
        """
        return self._interrupt_host._maybe_fire_permission_summary(payload)

    def _permission_summary_worker(
        self, round_id: str, payload: dict[str, Any], resolution: object, tail: list
    ) -> None:
        """Worker THREAD: run the advisory call, deliver on the UI thread.

        The approval wait loop is never blocked and the round's deadline is
        unaffected; a slow call that outlives the round is dropped on
        delivery. Content-free failures only (ADR-090).

        ``tail`` arrives already built: it is store-derived, and the store
        belongs to the thread that spawned this one (TASK-32801.4). Nothing
        here may touch ``self.store``.
        """
        return self._interrupt_host._permission_summary_worker(
            round_id, payload, resolution, tail
        )

    def _summary_tail_messages(self, payload: dict[str, Any]) -> list:
        """User/assistant text projection of the round's stored conversation.

        Uses the same message flattening as world-info scanning
        (``_normalize_world_info_history``) and the same stored-message
        source its call sites feed it (``_provider_messages_for_session``,
        which reads ``self.store.messages_for_session`` and emits the
        provider-dict shape the flattener consumes); keeps the defensive
        no-raise posture.
        """
        return self._interrupt_host._summary_tail_messages(payload)

    def _deliver_permission_summary(
        self, round_id: str, payload: dict[str, Any], text: str
    ) -> None:
        """UI THREAD: store the summary, then patch the mounted card.

        Drops resolved/revoked rounds and unknown ids; writes the payload's
        ``summary`` slot (the source of truth for remounts) before the live
        patch. Never re-runs ``set_batch``.
        """
        return self._interrupt_host._deliver_permission_summary(round_id, payload, text)

    def _publish_pending_decision(
        self,
        *,
        round_state: dict[str, Any],
        payload: dict[str, Any],
        decision_type: Literal["approval", "skill_install", "skill_script"],
        decision_id: str,
        timeout_seconds: float,
        retained_store: dict[str, dict[str, Any]] | None,
    ) -> bool:
        """Atomically admit, order, retain, and derive one decision head.

        Admission order is the order in which fully built decision payloads
        acquire ``_approval_state_lock`` here. The order stamp and retained
        payload become visible in the same transaction, so another decision
        can observe neither half of an admission. The five-kind host calls
        this after releasing its non-reentrant registration lock.
        """
        return self._interrupt_host._publish_pending_decision(
            round_state=round_state,
            payload=payload,
            decision_type=decision_type,
            decision_id=decision_id,
            timeout_seconds=timeout_seconds,
            retained_store=retained_store,
        )

    def _pending_round_states_snapshot(self) -> dict[str, dict[str, Any]]:
        """Snapshot existing round records without nesting registry locks.

        Lock order is type registry lock, release, then
        ``_approval_state_lock`` in callers. No code acquires a type lock while
        holding the shared projection lock.
        """
        return self._interrupt_host._pending_round_states_snapshot()

    def _pending_decision_payloads_locked(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        return self._interrupt_host._pending_decision_payloads_locked(session_id)

    def _pause_pending_decision_state_locked(
        self, state: dict[str, Any], *, now: float
    ) -> None:
        return self._interrupt_host._pause_pending_decision_state_locked(state, now=now)

    def _mutate_exact_pending_decision(
        self, decision_id: str, mutate: Callable[[dict[str, Any]], Any]
    ) -> Any:
        """Mutate one live round under its registry then projection lock.

        All five kinds share the host's non-reentrant lock. Mutations must
        never acquire a second aliased lock or project UI while holding it.
        """
        return self._interrupt_host._mutate_exact_pending_decision(decision_id, mutate)

    @staticmethod
    def _settle_pending_decision_timeout_locked(
        state: dict[str, Any],
    ) -> threading.Event | None:
        """Stamp one exact live round timeout once; caller holds its locks."""
        from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

        return InterruptRoundHost._settle_pending_decision_timeout_locked(
            state, read_global_threading=lambda: threading
        )

    def _pause_answerable_decision(
        self,
        session_id: str,
        decision_id: str,
        *,
        now: float,
        claim_revision: int | None = None,
    ) -> bool:
        """Pause one exact head, terminally timing it out at zero."""
        return self._interrupt_host._pause_answerable_decision(
            session_id, decision_id, now=now, claim_revision=claim_revision
        )

    def _expire_answerable_decision_if_due(
        self, session_id: str, decision_id: str, *, now: float
    ) -> bool:
        """Settle one still-mounted head only when its active allowance is due."""
        return self._interrupt_host._expire_answerable_decision_if_due(
            session_id, decision_id, now=now
        )

    def pending_decision_projection(
        self, session_id: str
    ) -> ConsolePendingDecisionProjection | None:
        """Return the session's one stable mixed-type FIFO head."""
        return self._interrupt_host.pending_decision_projection(session_id)

    def project_pending_decision_for_active_session(self) -> bool:
        """Project only the active session's ordered mixed-type head."""
        return self._interrupt_host.project_pending_decision_for_active_session()

    def _reproject_pending_decision_for_session(self, session_id: str) -> None:
        """Re-derive one session through the unified or legacy card seams."""
        return self._interrupt_host._reproject_pending_decision_for_session(session_id)

    def active_session_changed(self) -> None:
        """Pause stale heads and derive the newly active session's head."""
        return self._interrupt_host.active_session_changed()

    def _cancel_pending_decisions_for_session(self, session_id: str) -> None:
        """Fail closed only the rounds owned by a destructively closed session."""
        return self._interrupt_host._cancel_pending_decisions_for_session(session_id)

    def _marshal_pending_decision_projection(self) -> None:
        """Worker-thread marshal of the active-session derived head."""
        return self._interrupt_host._marshal_pending_decision_projection()

    def set_answerable_decision(self, session_id: str, decision_id: str | None) -> bool:
        """Update Console's claim without erasing another visible owner's claim."""
        return self._interrupt_host.set_answerable_decision(session_id, decision_id)

    def _refresh_answerable_decision(self, session_id: str) -> str | None:
        """Reconcile rendered Console/Buddy claims against one typed FIFO clock."""
        return self._interrupt_host._refresh_answerable_decision(session_id)

    def expire_pending_decisions(self) -> tuple[str, ...]:
        """Fail closed every answerable head whose active allowance elapsed."""
        return self._interrupt_host.expire_pending_decisions()

    @staticmethod
    def _head_round_payload_locked(
        store: dict[str, dict[str, Any]], session_id: str | None
    ) -> dict[str, Any] | None:
        """The session's oldest-armed payload. Caller holds the lock."""
        from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

        return InterruptRoundHost._head_round_payload_locked(store, session_id)

    def _park_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head."""
        return self._interrupt_host._park_round_payload(store, round_id, payload)

    def _head_round_payload(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should currently show (remaining-time snapshot)."""
        return self._interrupt_host._head_round_payload(store, session_id)

    def _session_round_payloads(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Every payload ``store`` retains for ``session_id``, arm order first."""
        return self._interrupt_host._session_round_payloads(store, session_id)

    def _unpark_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str
    ) -> None:
        """Drop ``round_id``'s retained payload, if any."""
        return self._interrupt_host._unpark_round_payload(store, round_id)

    def _remount_head(
        self,
        store: dict[str, dict[str, Any]],
        setter: Callable[[dict[str, Any] | None], None] | None,
        session_id: str | None,
    ) -> None:
        """Push ``store``'s head for ``session_id`` through ``setter`` on the UI thread.

        The pre-host body, kept verbatim: every kind's activation re-derive
        and the approval attach path call it with their own store and
        setter, and it fires the ADR-090 permission summary for any dict
        payload -- ``_maybe_fire_permission_summary`` itself ignores round
        ids it does not know, so a skill payload passes through untouched.

        Args:
            store: A per-kind parked-payload dict.
            setter: The kind's UI-thread setter, or None to do nothing.
            session_id: The session whose head to push; None means the
                store's active session.
        """
        return self._interrupt_host._remount_head(store, setter, session_id)

    def _remount_session_kinds(self, session_id: str) -> None:
        """Re-derive every non-approval kind's head card for ``session_id``.

        The one call the three session-activation sites (new, switch,
        close) share; the kinds come from the host module's
        ``SESSION_REMOUNT_KINDS`` and approvals stay on the sites' own block.

        Args:
            session_id: The session being activated.
        """
        return self._interrupt_host._remount_session_kinds(session_id)

    def on_console_view_visibility_changed(self, visible: bool) -> None:
        """Project screen visibility without changing execution or cancellation."""
        return self._interrupt_host.on_console_view_visibility_changed(visible)

    def remount_pending_approval_for_active_session(self) -> bool:
        """Mount the ACTIVE session's still-armed approval round, if any.

        task-15860 Task 5. UI THREAD (called from
        ``ConsoleRuntime.attach_view``, which runs on it). Re-derives the
        card from ``_parked_approval_payloads`` exactly as
        ``switch_session`` does -- same single source of truth, no second
        copy of "what is this session's card showing".

        Deliberately mounts NOTHING when no round is armed: pushing
        ``None`` here would clear a card on every new claim, and an attach
        is not a reason to hide anything.

        PR0 (task-15661, fixed): ``_parked_approval_payloads`` is keyed by
        ROUND now, so two rounds armed for one session each keep their own
        payload. This mounts the session's FIFO HEAD -- the oldest-armed
        round -- and each later sibling mounts in turn as the head ahead of
        it resolves. Covered by ``Tests/UI/test_console_headless_approval.
        py::test_two_headless_rounds_each_mount_in_turn``.

        Returns:
            True when a card was mounted.
        """
        return self._interrupt_host.remount_pending_approval_for_active_session()

    def _approval_view_is_detached(self) -> bool:
        """True when Console is hidden or its approval view hooks are absent.

        TASK-31520 retains hooks during navigation. Attachment alone therefore
        cannot tell whether the user can see a card; modals also suspend it.
        """
        return self._interrupt_host._approval_view_is_detached()

    def on_pending_rounds_changed(self, total: int, kind: str, raised: bool) -> None:
        """task-31385: attention when a round blocks on the user off-screen.

        WORKER THREAD; the host calls this after every round arms (mounted
        or parked) and after every teardown. Two effects, both on the UI
        thread: the Console entry in the app navigation carries a
        pending-interrupt badge while ``total`` is non-zero, and a round
        that ARMS while Console is hidden or detached -- another screen or
        modal is visible, or Console has not been opened this launch --
        rings the terminal bell once. The bell is governed by
        ``[console] interrupt_bell`` (default on) and never fires in a
        headless app, so tests and embedded runs emit no control bytes.
        ``_approval_view_is_detached`` combines suspend/resume visibility
        with the absent-hook fallback for a truly detached view.

        Args:
            total: Rounds of every kind registered after this change.
            kind: The round kind that changed (unused; kept for callers
                that want to specialise).
            raised: True for an arm, False for a teardown.
        """
        return self._interrupt_host.on_pending_rounds_changed(total, kind, raised)

    def _interrupt_bell_enabled(self) -> bool:
        """Resolve ``[console] interrupt_bell``: environment, then config, then on.

        Returns:
            False only when ``TLDW_CONSOLE_INTERRUPT_BELL`` (a non-empty
            value) or the config key coerces to False.
        """
        return self._interrupt_host._interrupt_bell_enabled()

    def announce_hidden_decision(self, session_id: str, kind: str) -> None:
        """Keep typed notices under their live stable-ID privacy authority."""
        return self._interrupt_host.announce_hidden_decision(session_id, kind)

    def _announce_detached_approval(
        self, session_id: str, *, kind: str = CONSOLE_PENDING_APPROVAL_KIND
    ) -> None:
        """Raise the app-wide toast for a round with no visible Console view.

        WORKER THREAD. ``App.notify`` is documented thread-safe (it posts
        a message), so this needs no ``call_from_thread`` marshal -- and
        the toast renders on whatever screen the user is currently
        looking at, which is the whole point: the screen-owned seam
        (``ChatScreen._park_console_approval``) is unreachable here.

        Best-effort in both directions. An app double with no ``notify``
        (several controller-level tests) is silently skipped, and a
        raising/incompatible ``notify`` is logged rather than allowed to
        break the round -- a missing toast must never turn into a missing
        approval.

        Args:
            session_id: The round's owning session, used only to name the
                conversation in the notice.
        """
        return self._interrupt_host._announce_detached_approval(session_id, kind=kind)

    def _announce_hidden_decision(
        self,
        decision_type: Literal["approval", "skill_install", "skill_script"],
        session_id: str,
        decision_id: str,
    ) -> None:
        """Raise one content-free app notice for a hidden stable decision.

        WORKER THREAD. ``App.notify`` is documented thread-safe (it posts
        a message), so this needs no ``call_from_thread`` marshal -- and
        the toast renders on whatever screen the user is currently
        looking at, which is the whole point: the screen-owned seam
        (``ChatScreen._park_console_approval``) is unreachable here.

        Best-effort in both directions. An app double with no ``notify``
        (several controller-level tests) is silently skipped, and a
        raising/incompatible ``notify`` is logged rather than allowed to
        break the round -- a missing toast must never turn into a missing
        approval.

        Args:
            session_id: The round's owning session. It is routing context
                only and is never interpolated into the notice.
        """
        return self._interrupt_host._announce_hidden_decision(
            decision_type, session_id, decision_id
        )

    def _forget_hidden_decision(self, decision_id: str) -> None:
        """Release one terminal decision's app-wide announcement marker."""
        return self._interrupt_host._forget_hidden_decision(decision_id)

    def _resolve_mcp_approval_timeout_seconds(self) -> float:
        return self._interrupt_host._resolve_mcp_approval_timeout_seconds()

    def _console_tool_kill_switch_reader(self) -> Callable[[], bool] | None:
        """Return a fresh-per-call kill-switch reader, or ``None`` without a service.

        TASK-631. A callable rather than a bool so `build_tool_review_hook`
        observes a mid-run flip on the next batch; reading raises -> the
        hook fails CLOSED (refuses the turn), which is the only safe answer
        for a security control that cannot be read.

        Returns:
            A zero-arg callable returning the switch state, or ``None``
            when the app has no ``unified_mcp_service`` (nothing to honor).
        """
        return self._interrupt_host._console_tool_kill_switch_reader()

    def resolve_pending_approval(
        self, decisions: dict[str, str], *, round_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's batch decision, releasing the waiting worker thread.

        Called by ``ChatScreen``'s ``ChatApprovalCard.ApprovalDecided``
        handler, which forwards ``event.round_id`` -- the SAME id
        ``request_mcp_approvals`` stamped into the payload the card was
        built from (``ChatApprovalCard.set_batch`` stashes it;
        ``_submit_batch_decisions`` echoes it back on submit, mirroring
        ``resolve_pending_skill_script``'s identical ``request_id``
        round-trip).

        Fix round 1 (review CRITICAL finding): resolves ONLY the round
        whose id matches ``round_id`` -- never "whichever round belongs to
        the currently active session". ``ApprovalDecided`` travels as an
        async Textual message: a ``switch_session`` landing in the gap
        between the user's click and this handler running would otherwise
        let session A's decision resolve session B's completely different,
        unreviewed batch (or, for the same session, let a STALE decision
        from an already-ended round 1 resolve a newer round 2 that
        happened to arm before the stale message was delivered). A
        mismatched or stale ``round_id`` -- including one belonging to a
        round that already resolved and was popped -- is a safe no-op: the
        real round (if any) stays pending and its card re-derives
        unchanged on the next visit; nothing is ever auto-approved or
        denied-by-accident here.

        TASK-913 (AC#2): ``round_id=None`` no longer falls back to
        "whichever round belongs to the currently active session" -- it
        fails closed immediately, mirroring
        ``resolve_pending_skill_script``'s/``resolve_pending_skill_install``'s
        identical ``if request_id is None: return`` contract. Production
        (``ChatApprovalCard``/``ChatScreen``) has only ever had a single
        emitter (``ChatApprovalCard._submit_batch_decisions``) and it
        always threads the real ``round_id`` through; the active-session
        fallback existed only for legacy direct-call tests, which have
        been migrated to pass the real round id captured from the
        mounted/parked payload instead.

        A no-op both when ``round_id`` is ``None`` and when it doesn't
        match any currently-armed round (e.g. a stale message arriving
        after a timeout/cancellation already resolved and cleared it) --
        the real round (if any) stays pending and undecided; nothing is
        ever auto-approved or denied-by-accident here.

        NOTE: Snapshots the round's ``decisions``/``event`` into locals to
        avoid TOCTOU race: the worker thread's ``finally`` block pops the
        round entry out of ``_pending_approval_rounds`` concurrently. Guard
        and act only on the snapshots.

        Args:
            decisions: The user's per-``llm_name`` decision strings
                (``approve_once``/``approve_session``/``always_allow``/
                ``deny``) to merge into the round's shared decisions dict.
            round_id: The specific round to resolve (the id stamped onto
                the card the user actually decided). ``None`` (the
                default) never matches an armed round, so an un-migrated
                or malformed caller fails closed by omission.
        """
        return self._interrupt_host.resolve_pending_approval(
            decisions, round_id=round_id
        )

    def complete_definitive_tool(
        self, run_id: str, call_key: str, tool_name: str
    ) -> None:
        """WORKER THREAD: clear one finishing row at its real terminal.

        The primary key is the provider call id.  Fence/local rows that did
        not carry one fall back to the tool name; only one matching row is
        consumed per callback so repeated same-name calls remain visible
        until each sequential mutation actually finishes.
        """
        return self._interrupt_host.complete_definitive_tool(
            run_id, call_key, tool_name
        )

    def complete_definitive_run(self, run_id: str) -> None:
        """WORKER THREAD: remove finishing rows a run never dispatched."""
        return self._interrupt_host.complete_definitive_run(run_id)

    def _discard_approval_rows_for_closing_session(self, session_id: str) -> None:
        """Drop every approval payload owned by a closing session.

        This uses the same lock as the approval-to-finishing transition, so
        whichever operation wins first, no later transition can retain a row
        for a session that is being deleted.
        """
        return self._interrupt_host._discard_approval_rows_for_closing_session(
            session_id
        )

    def revoke_approval_rounds_for_run(self, run_id: str) -> int:
        """Fail every approval round owned by ``run_id`` closed, right now.

        PR2a Task 7 (safety). The approval wait blocks inside
        ``_call_with_timeout``'s per-call daemon thread, which keeps
        running after the fleet cooperatively cancels -- or outright
        ABANDONS -- the child that owns it. Until this existed, that
        child's card stayed on screen and stayed live: pressing Approve
        resolved the round, the waiting thread returned the approval, and
        the tool EXECUTED FOR REAL (a file written, a message sent) for a
        run whose handle and run row already read ``cancelled``. The
        documented ``approval_timeout < max_tool_call_seconds`` invariant
        (see ``_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS``) bounds the same
        class of hazard for the timeout path; this closes the
        cancellation path.

        Called by ``AgentService`` (through its injected
        ``revoke_approvals`` seam) at both moments a child stops being
        allowed to act: the cooperative cancel and the end-of-turn
        abandon. Safe to call for a run that never armed a card -- the
        common case -- and never touches another run's rounds, which
        matters because every child of a fleet turn shares ONE console
        session: session-keyed teardown could not tell a cancelled child's
        card from its live sibling's.

        Covers tool-call approvals, run_skill_script confirms, and ask_user
        questions. Skill-install and worktree-merge confirms are primary-only
        and are not swept. The host also fences future arms for the revoked
        run and these kinds for its lifetime, even when no round exists yet.

        Each revoked round is (a) marked ``revoked`` so the waiting thread
        fails closed even if a click lands in its shared decision box
        afterwards, (b) pre-filled with the closed verdict, (c) removed
        from its registry, so a late ``resolve_pending_approval``/
        ``resolve_pending_skill_script`` finds nothing to resolve, (d)
        released via its Event, so the waiting thread returns immediately
        rather than at its auto-deny deadline, (e) discarded from
        ``_pending_approvals`` so the session's NEEDS_APPROVAL badge
        clears once its last round is gone, and (f) taken off screen
        through the SAME FIFO-head re-derive (``_remount_head``) that the
        round's own teardown uses, so a sibling round's card is never
        clobbered.

        Thread-safe. InterruptRoundHost records the per-kind revocation
        fences and sweeps the registries under one shared non-reentrant lock.
        Exact-round payload cleanup and badge/UI callbacks run after that
        critical section; callbacks may acquire the same lock themselves.

        Args:
            run_id: The cancelled/abandoned run whose cards must die. A
                falsy id is a no-op -- ``""`` is the "no run bound" key
                that rounds armed outside any agent run carry, and
                sweeping those would deny cards no run owns.

        Returns:
            How many existing rounds were revoked across the swept kinds
            (``0`` when the run had none; the late-arm fence still persists).
        """
        return self._interrupt_host.revoke_approval_rounds_for_run(run_id)

    def revoke_raw_shell_authority(self) -> int:
        """Fail closed only raw-shell stamps and approval rounds on disarm."""
        return self._interrupt_host.revoke_raw_shell_authority()

    def _revoke_tool_approval_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Fail this run's tool-approval rounds closed. Registry work only.

        Args:
            run_id: The cancelled/abandoned run.

        Returns:
            ``(round_id, session_id)`` for each revoked round, for the
            caller's badge/card teardown (which must run outside the lock
            held here).
        """
        return self._interrupt_host._revoke_tool_approval_rounds(run_id)

    def _revoke_skill_script_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Fail this run's ``run_skill_script`` confirms closed.

        The host fences and sweeps under its shared lock, then removes only
        each swept round's retained payload. Same-session siblings retain
        their own round-keyed payloads through revocation and teardown.

        Args:
            run_id: The cancelled/abandoned run.

        Returns:
            ``(request_id, session_id)`` for each revoked confirm.
        """
        return self._interrupt_host._revoke_skill_script_rounds(run_id)

    def request_skill_install_confirm(
        self, url: str, *, session_id: str | None = None
    ) -> bool:
        """WORKER THREAD: ask the user to confirm a skill install before any fetch.

        TASK-910: mirrors ``request_mcp_approvals``' park/mount/retain
        contract. Registers a fresh round (event + decision box + owning
        session id) under a freshly minted request id in
        ``_pending_skill_install_rounds`` (mirrors ``_pending_skill_script_
        rounds``' identical per-round design -- the pre-TASK-910 single
        ``_pending_skill_install_event``/``_pending_skill_install_decision``
        pair could not survive two DIFFERENT sessions each raising their own
        install confirm concurrently, exactly the hazard task-581 already
        fixed for skill-script). Either MOUNTS the card immediately
        (``session_id`` is the active/viewed session, or unknown -- legacy
        no-session callers keep the pre-TASK-910 always-mount behavior) or
        PARKS it (a different, background session -- the retained payload
        goes into ``_parked_skill_install_payloads`` for ``switch_session``/
        ``new_session``/``close_session`` to remount later, while the
        controller raises one sanitized app-wide notice for this exact
        stable decision id).

        Then polls re-checking this round's OWN cancel signal
        (``_is_session_cancelled``, scoped to ``session_id`` when known) and
        a deadline. Cancel/stop (of the OWNING session, or real process
        teardown via ``_shutdown_requested``), timeout, or no wired UI all
        resolve to DENY (fail-closed). A plain switch away no longer denies
        -- the round parks and stays alive until its own resolution,
        cancellation, or shutdown. Returns True only on an explicit Allow.

        Args:
            url: The skill source URL the model wants to install, surfaced
                verbatim on the confirm card for the user to inspect.
            session_id: The run's OWNING session (Task 3/9/TASK-910).
                ``None`` preserves the pre-Task-9 VIEWED-session/global-flag
                fallback (see ``_is_session_cancelled``) and never parks.

        Returns:
            True only on an explicit Allow; every other path (deny, cancel,
            stop, timeout, or no wired UI) returns False.
        """
        return self._interrupt_host.request_skill_install_confirm(
            url, session_id=session_id
        )

    def _remount_parked_skill_install(self, session_id: str) -> None:
        """Re-derive the mounted skill-install confirm card for ``session_id``.

        TASK-910: called from `switch_session`/`new_session`/`close_session`
        exactly like the MCP approval card's own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        Args:
            session_id: The session now being activated/viewed.
        """
        return self._interrupt_host._remount_parked_skill_install(session_id)

    def _marshal_pending_skill_install(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-install confirm payload to the UI thread.

        No-op when no UI bridge is wired (``self.app`` or
        ``set_pending_skill_install`` is None).

        Args:
            payload: The pending confirm's ``{"url", "timeout_seconds"}``
                dict to show, or None to clear/hide the card.
        """
        return self._interrupt_host._marshal_pending_skill_install(payload)

    def resolve_pending_skill_install(
        self, allow: bool, *, request_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's Allow/Deny, releasing the worker thread.

        TASK-910: strict match against ``request_id``, mirroring
        ``resolve_pending_skill_script``'s identical contract -- a resolve
        carrying no id, or an id belonging to any round other than the one
        it names, is silently dropped rather than resolved. This closes the
        same stale-late-click hazard ``resolve_pending_skill_script``'s own
        docstring documents: once two sessions can each have their own
        concurrent install-confirm round (TASK-910 parking), "whichever
        round happens to be active" is no longer a safe fallback the way it
        was pre-TASK-910 (a single global slot could only ever have one
        candidate).

        Args:
            allow: True to allow the pending install, False to deny it.
            request_id: The armed round's id, as echoed back by the UI
                (``SkillInstallConfirmCard.InstallDecided.request_id``).
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        return self._interrupt_host.resolve_pending_skill_install(
            allow, request_id=request_id
        )

    def pending_skill_install_ids(self) -> list[str]:
        """Return the request ids of every currently-armed install-confirm round.

        Mirrors ``pending_skill_script_ids`` -- exposed for tests and for
        any surface that needs to know whether a decision is outstanding.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending.
        """
        return self._interrupt_host.pending_skill_install_ids()

    def request_skill_script_confirm(
        self, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm running a skill's script.

        Mirrors request_skill_install_confirm, but carries a two-part decision:
        allow this run, and whether to remember the choice for this skill.

        Each call arms a fresh round under a newly-generated request id
        (embedded in the payload handed to the UI as ``"request_id"``) so
        that ``resolve_pending_skill_script`` can reject a decision left
        over from a prior, already-torn-down round -- see that method's
        docstring for why this matters.

        TASK-910: also carries the SAME park/mount/retain contract as
        ``request_mcp_approvals``/``request_skill_install_confirm`` -- see
        ``request_skill_install_confirm``'s docstring for the full
        mount-vs-park/retain rationale, identical here. The per-round
        registry (keyed by ``request_id``, task-581) now also stores this
        round's owning session id, so teardown can distinguish "another
        round for a DIFFERENT session is still armed" (must not suppress
        clearing THIS session's card) from "another round for the SAME
        session is still armed" (must not clear it out from under that
        sibling round, preserving task-581's original guarantee).

        Args:
            payload: Confirm details to render ({"skill_name", "script_path",
                "mechanism", "args", ...}); "timeout_seconds" and
                "request_id" keys are added before marshaling to the UI.
            session_id: The run's OWNING session (Task 3/9/TASK-910), scoping
                the cancel check (``_is_session_cancelled`` -- PA-T9 finding
                #1) and the park/mount decision. ``None`` preserves the
                pre-Task-9 VIEWED-session/global-flag fallback and never
                parks.

        Returns:
            ``{"allow": bool, "remember": bool}``. Every non-Allow path (deny,
            cancel, stop, timeout, no wired UI) returns ``allow=False``.
        """
        return self._interrupt_host.request_skill_script_confirm(
            payload, session_id=session_id
        )

    def _remount_parked_skill_script(self, session_id: str) -> None:
        """Re-derive the mounted skill-script confirm card for ``session_id``.

        TASK-910: called from `switch_session`/`new_session`/`close_session`
        exactly like the MCP approval card's own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        PR0: re-keyed by round, so this now re-derives the session's FIFO
        head instead of a single per-session slot. It already runs on the
        UI thread, so it calls `_head_round_payload` directly rather than
        `_remount_head`.

        Args:
            session_id: The session now being activated/viewed.
        """
        return self._interrupt_host._remount_parked_skill_script(session_id)

    def _marshal_pending_skill_script(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-script confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        return self._interrupt_host._marshal_pending_skill_script(payload)

    def _marshal_task_panel(
        self, session_id: str, tasks: list[dict[str, object]]
    ) -> None:
        """WORKER THREAD: hand a session's task snapshot to the pinned panel.

        PRD Feature B (AC-B4): fires on every ``todo_*`` change alongside
        the transcript marker. The screen-side setter ignores snapshots
        for sessions that are not the viewed one.

        Args:
            session_id: The session whose todo store changed.
            tasks: Its full task list after the change.
        """
        return self._interrupt_host._marshal_task_panel(session_id, tasks)

    def _remount_task_panel(self, session_id: str | None) -> None:
        """UI THREAD: re-derive the pinned task panel for ``session_id``.

        Called from `switch_session`/`new_session`/`close_session` next to
        the card re-derives, and from `ConsoleRuntime.attach_view` when a
        new screen claims the surviving runtime, so the panel always shows
        the VIEWED session's tasks (AC-B5) and hides when that session has
        none -- or when there is no session at all.

        Args:
            session_id: The session now being activated/viewed, or None
                when none is (the last session was just closed).
        """
        return self._interrupt_host._remount_task_panel(session_id)

    def resolve_pending_skill_script(
        self, allow: bool, remember: bool, request_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's decision, releasing the worker thread.

        ``request_id`` must be the exact ``"request_id"`` value the pending
        confirm's payload carried (``request_skill_script_confirm`` embeds
        a fresh one per round, and the confirm card built in a later task
        MUST echo it back here unchanged). This is a strict match: a
        resolve carrying no id, or an id from any round other than the one
        currently armed, is silently dropped rather than resolved.

        This guards against a real arbitrary-code-execution hazard: if
        round 1 ends (deadline, cancel, stop, conversation switch) and the
        agent immediately issues a second ``run_skill_script`` call
        arming round 2, a ``Button.Pressed`` queued for round 1 just
        before its teardown could otherwise be handled after round 2 is
        armed -- resolving round 2 (a script the user never saw) with
        round 1's stale click. Widget messages and ``call_from_thread``
        calls are separate queues, so ordering across a round boundary is
        not guaranteed.

        Args:
            allow: True to run the script this once.
            remember: True to also grant this skill standing permission.
            request_id: The armed round's id, as echoed back by the UI.
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        return self._interrupt_host.resolve_pending_skill_script(
            allow, remember, request_id
        )

    def pending_skill_script_ids(self) -> list[str]:
        """Return the request ids of every currently-armed confirm round.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending. Exposed for tests and for any surface that needs to
            know whether a decision is outstanding.
        """
        return self._interrupt_host.pending_skill_script_ids()

    def _enrich_chat_create_confirm_payload(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Fill the confirm card's fork facts, default title, and run id.

        Final-review fix wave (Finding 1): the card renders
        ``fork_source_title`` / ``fork_message_count`` (its "Copies N
        messages from '<title>'" line) and a non-empty header title, but no
        production payload producer ever set the fork keys -- every card
        read "Copies ? messages from ''" -- and an agent-omitted title left
        the header blank until the executor computed its default
        post-confirm. The controller is the only side holding
        store/persistence/db access at arm time, so it enriches here,
        BEFORE the round is armed.

        EVERYTHING is best-effort: any failure degrades -- the fork line's
        keys are omitted, a fork title falls back to the owning session's
        display title so the header still renders -- and can never block,
        delay, or deny the round.
        """
        return self._interrupt_host._enrich_chat_create_confirm_payload(payload)

    def request_chat_create_confirm(
        self, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm an agent-initiated chat create.

        Mirrors request_skill_script_confirm for the fork_chat/new_chat
        tools, with one addition: a session-scoped "remember" grant store.
        A prior allow+remember decision for ``(session_id, tool)`` (recorded
        in ``_chat_create_session_grants``) short-circuits this call with
        ``{"allow": True, "remember": True}`` before any round is armed --
        no card, no wait. Grants die with their session (``close_session``
        pops the whole set).

        Each call arms a fresh round under a newly-generated request id
        (embedded in the payload handed to the UI as ``"request_id"``) so
        that ``resolve_pending_chat_create`` can reject a decision left
        over from a prior, already-torn-down round -- see that method's
        docstring for why this matters.

        Carries the SAME park/mount/retain contract as
        ``request_skill_script_confirm`` -- see that method's docstring for
        the full mount-vs-park/retain rationale, identical here.

        Args:
            payload: Confirm details to render ({"tool" ("fork_chat"|
                "new_chat"), "title", "opening_prompt", "instructions"});
                "fork_source_title"/"fork_message_count" (fork_chat only),
                a default "title" when the agent omitted one, and a
                normalized "run_id" are enriched by
                ``_enrich_chat_create_confirm_payload`` before arming, and
                "timeout_seconds", "request_id", "session_id" and
                "deadline_monotonic" keys are added before marshaling to
                the UI.
            session_id: The run's OWNING session, scoping the cancel check
                (``_is_session_cancelled``), the park/mount decision, and
                the remember-grant lookup. ``None`` preserves the
                viewed-session fallback and never parks.

        Returns:
            ``{"allow": bool, "remember": bool}``. Every non-Allow path
            (deny, cancel, stop, timeout, no wired UI) returns
            ``allow=False``.
        """
        return self._interrupt_host.request_chat_create_confirm(
            payload, session_id=session_id
        )

    def _remount_parked_chat_create(self, session_id: str) -> None:
        """Re-derive the mounted chat-create confirm card for ``session_id``.

        Called from `switch_session`/`new_session`/`close_session` exactly
        like the sibling confirm cards' own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        Already runs on the UI thread, so it calls `_head_round_payload`
        directly rather than `_remount_head`.

        Args:
            session_id: The session now being activated/viewed.
        """
        return self._interrupt_host._remount_parked_chat_create(session_id)

    def _marshal_pending_chat_create(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: project a current chat-create decision on the UI.

        Recheck scoped ownership after dispatch; legacy unparked rounds keep
        their unconditional initial projection. A clear derives the current head.

        Args:
            payload: Proposed confirmation, or None to rederive the active head.
        """
        return self._interrupt_host._marshal_pending_chat_create(payload)

    def resolve_pending_chat_create(
        self, allow: bool, remember: bool, request_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's decision, releasing the worker thread.

        ``request_id`` must be the exact ``"request_id"`` value the pending
        confirm's payload carried (``request_chat_create_confirm`` embeds a
        fresh one per round, and the confirm card built in a later task
        MUST echo it back here unchanged). This is a strict match: a
        resolve carrying no id, or an id from any round other than the one
        currently armed, is silently dropped rather than resolved.

        Same hazard class as ``resolve_pending_skill_script``: if round 1
        ends (deadline, cancel, stop, conversation switch) and the agent
        immediately issues a second fork_chat/new_chat call arming round 2,
        a ``Button.Pressed`` queued for round 1 just before its teardown
        could otherwise be handled after round 2 is armed -- resolving
        round 2 (a chat the user never saw) with round 1's stale click.
        Widget messages and ``call_from_thread`` calls are separate
        queues, so ordering across a round boundary is not guaranteed.

        Args:
            allow: True to create the chat this once.
            remember: True to also grant this tool standing permission in
                the owning session.
            request_id: The armed round's id, as echoed back by the UI.
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        return self._interrupt_host.resolve_pending_chat_create(
            allow, remember, request_id
        )

    def pending_chat_create_ids(self) -> list[str]:
        """Return the request ids of every currently-armed chat-create round.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending. Exposed for tests and for any surface that needs to
            know whether a decision is outstanding.
        """
        return self._interrupt_host.pending_chat_create_ids()

    def _resolve_ask_user_timeout_seconds(self) -> float:
        """PRD A7: the question deadline -- seam, else env, else config, else 0.

        The injected seam exists for tests; production precedence is
        ``TLDW_CONSOLE_ASK_USER_TIMEOUT_SECONDS`` -> ``[console]
        ask_user_timeout_seconds`` -> ``0``. An empty or unparseable env
        value is ignored.

        Returns:
            Seconds before an unanswered question auto-continues; ``0.0``
            (the default) means no deadline. Never negative.
        """
        return self._interrupt_host._resolve_ask_user_timeout_seconds()

    def request_user_questions(
        self, questions: list[dict[str, Any]], *, session_id: str | None = None
    ) -> dict[str, Any]:
        """WORKER THREAD: show ``questions`` on a card and wait for the answers.

        PRD Feature A (A5-A7, A9-A11, A14). Clones
        ``request_worktree_merge_confirm``'s round machinery -- fresh
        request id, park-or-mount under the TASK-910 contract, poll under
        ``use_human_input_wait`` so the owning run's tool clock pauses,
        cancel/deadline checks -- with a question-shaped decision. Two
        differences: a second call while this session already has a live
        round returns ``busy`` at once (A9: depth is expressed by batching
        questions, never by queueing rounds), and every outcome is recorded
        in the transcript on resolve (A14).

        Args:
            questions: Validated questions (``ask_user_questions.
                validate_questions`` output).
            session_id: The run's OWNING session; ``None`` never parks.

        Returns:
            ``{"answered": True, "answers": [...]}`` or ``{"answered":
            False, "reason": "timeout" | "cancelled" | "busy"}``.

        Raises:
            AskUserBusyRefusal: ``MAX_CONSECUTIVE_BUSY`` consecutive busy
                results in one run (A9's retry-loop ceiling).
        """
        return self._interrupt_host.request_user_questions(
            questions, session_id=session_id
        )

    def resolve_pending_question(
        self, answers: list[dict[str, Any]], request_id: str | None = None
    ) -> None:
        """UI THREAD: hand the card's answers to the waiting worker thread.

        Strict ``request_id`` match, exactly like
        ``resolve_pending_skill_script``: a resolve with no id, or an id
        from any round but the armed one, is silently dropped. The answers
        are validated (``ask_user_questions.validate_answers``) before the
        worker sees them; a malformed list is dropped the same way.

        Args:
            answers: One PRD A6 answer dict per question, in order.
            request_id: The armed round's id as echoed back by the card.
        """
        return self._interrupt_host.resolve_pending_question(answers, request_id)

    def pending_question_ids(self) -> list[str]:
        """Return the request ids of every armed question round, arm order.

        Returns:
            The armed round ids; empty when none is pending.
        """
        return self._interrupt_host.pending_question_ids()

    def _marshal_pending_question(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a question payload to the UI thread.

        Args:
            payload: The card payload to show, or None to hide the card.
        """
        return self._interrupt_host._marshal_pending_question(payload)

    def _remount_parked_question(self, session_id: str) -> None:
        """UI THREAD: re-derive the question card for the session now viewed.

        Called from ``switch_session``/``new_session``/``close_session``
        beside the other card re-derives (PRD A10).

        Args:
            session_id: The session being activated/viewed.
        """
        return self._interrupt_host._remount_parked_question(session_id)

    @property
    def worktree_confirmation_enabled(self) -> bool:
        """Only the real disposable worktree surface enables new tool disclosure."""
        return self._interrupt_host.worktree_confirmation_enabled

    def request_worktree_merge_confirm(
        self,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
        operation_cancel_event: threading.Event | None = None,
    ) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm merging/discarding an
        agent worktree before ``AgentService`` mutates anything.

        Clones ``request_skill_script_confirm``'s round machinery -- arm a
        fresh request id, park-or-mount under the same TASK-910 contract
        (mount when ``session_id`` is the active/viewed session or
        unknown, park a different background session's round for
        ``switch_session``/``new_session``/``close_session`` to remount
        later), poll under ``use_human_input_wait`` so the owning run's
        tool-call deadline pauses while the card is up, and fail closed on
        cancel/stop/timeout/no-UI -- with a single-key decision instead of
        that method's two-part one: worktree merge/discard has no
        "remember" concept, just Allow/Deny.

        ``merge_agent_worktree``/``discard_agent_worktree`` are wired
        PRIMARY-agent-only (``AgentService.run_turn``'s ``fleet_active``
        gate -- a worktree only ever exists for a fleet-launched CHILD,
        merged/discarded by the PRIMARY that launched it), so unlike the
        skill-script confirm this never needs
        ``revoke_approval_rounds_for_run``'s sweep: there is no separate
        "a child was abandoned but its session lives on" case to guard --
        the round's own ``_is_session_cancelled`` check already covers
        "the primary's turn stopped."

        Args:
            payload: Confirm details to render (``{"handle_id", "mode" |
                "action", "branch", "worktree", "diffstat"}``, built by
                the ``AgentService`` closures -- see
                ``merge_agent_worktree_tool``/``discard_agent_worktree_
                tool``); ``"timeout_seconds"``, ``"request_id"``,
                ``"session_id"``, and ``"deadline_monotonic"`` are added
                before marshaling to the UI.
            session_id: The run's OWNING session -- always the PRIMARY's,
                per the gate above. ``None`` preserves the legacy VIEWED-
                session/global-flag fallback and never parks.

        Returns:
            ``{"allow": bool}`` -- the exact shape ``AgentService``'s
            ``merge_agent_worktree_tool``/``discard_agent_worktree_tool``
            closures read via ``decision.get("allow", False)``. Every
            non-Allow path (deny, cancel, stop, timeout, no wired UI)
            returns ``allow=False``.
        """
        return self._interrupt_host.request_worktree_merge_confirm(
            payload,
            session_id=session_id,
            operation_cancel_event=operation_cancel_event,
        )

    def _remount_parked_worktree_merge(self, session_id: str) -> None:
        """Re-derive the mounted worktree-merge confirm card for
        ``session_id``. Mirrors ``_remount_parked_skill_script``.

        Args:
            session_id: The session now being activated/viewed.
        """
        return self._interrupt_host._remount_parked_worktree_merge(session_id)

    def _marshal_pending_worktree_merge(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a worktree-merge confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        return self._interrupt_host._marshal_pending_worktree_merge(payload)

    def resolve_pending_worktree_merge(
        self, allow: bool, *, request_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's Allow/Deny, releasing the worker thread.

        Strict ``request_id`` match, mirroring
        ``resolve_pending_skill_script`` -- see that method's docstring
        for why a resolve carrying no id, or an id belonging to any round
        other than the one it names, is silently dropped.

        Args:
            allow: True to allow the pending merge/discard, False to deny.
            request_id: The armed round's id, as echoed back by the UI.
                ``None`` (the default) never matches an armed round.
        """
        return self._interrupt_host.resolve_pending_worktree_merge(
            allow, request_id=request_id
        )

    def pending_worktree_merge_ids(self) -> list[str]:
        """Return the request ids of every currently-armed worktree-merge
        confirm round. Mirrors ``pending_skill_script_ids``.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending.
        """
        return self._interrupt_host.pending_worktree_merge_ids()

    def _notify_run_hook_approval(
        self, kind: str, payload: dict[str, Any], state: dict[str, Any]
    ) -> None:
        """Publish one successfully admitted permission round, including headless runs."""
        return self._interrupt_host._notify_run_hook_approval(kind, payload, state)
