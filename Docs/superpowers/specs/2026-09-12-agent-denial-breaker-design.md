# Consecutive denial breaker design

Date: 2026-09-12. Scope: TASK-18929.

## Outcome

An agent that repeatedly requests refused tools stops before another model request once a completed tool batch ends with the configured number of consecutive denials. The default is 3; explicit 0 disables. Only an authoritative explicit user Deny or configured permission Off counts. Successful/approved calls and all non-denial results reset the streak. Each invocation of the runtime owns a fresh counter, including children and resumed children.

## Global Constraints

- `[agents] denial_circuit_breaker_limit` defaults to integer 3; explicit integer 0 disables.
- Count only authoritative explicit user Deny or configured permission Off; never infer a denial from display/result text.
- Timeout, no_callback, root-change, unavailable authority, unresolved approval, cancellation and restored_pending do not count.
- Successful `ToolResult.ok=True`, authoritative approval and any non-denial settled result reset the streak.
- Evaluate the trailing streak only after a complete tool batch; approved/successful tails reset it, and four denials in one batch at limit 3 report count 4.
- Preserve current cancellation and restoration/persistence-error precedence, complete native tool-call/reply pairing and already-approved dispatch semantics.
- Stop without another model call, preserve streamed partial assistant text, and emit an honest System explanation plus a run-local log record.
- Fresh runtime invocation means streak 0; no durable denial counter, sibling state, shared service counter or transcript reconstruction.
- No schema migration, dependencies, broad repeat-guard refactor, full test sweep or live configuration import during investigation.

## Internal contracts

In `Agents/agent_models.py`, add `ApprovalDecision = Literal["approved", "denied"]`. `ToolResult` gains `approval_decision: ApprovalDecision | None = field(default=None, kw_only=True)`. Extend `blocked(error, *, approval_decision=None)` without changing existing positional arguments or `outcome='blocked'` behavior. Keyword-only avoids altering positional subclass signatures.

Add frozen `ToolReviewDecision(verdict: str, approval_decision: ApprovalDecision | None = None)` and alias `ToolReviewValue = str | ToolReviewDecision`. Add `normalize_tool_review(value: ToolReviewValue) -> ToolReviewDecision`; strings normalize with no approval fact. A malformed approval fact normalizes to None. Do not introduce a str subclass or rewrite freeform refusal strings. Existing legacy callbacks still dispatch/refuse identically, but unannotated refusals do not advance this guard.

`LoopDeps.review_tool_calls`, AgentService's corresponding callback and Console hook annotations accept `dict[str, ToolReviewValue]`. Keep `_effective_review_verdict` returning a string for current control/trace behavior, using the same call-id-first/name fallback; add `_effective_review_decision` returning normalized metadata with that exact lookup. Service observation normalizes before comparison and sends the original verdict text to observation, never dataclass repr. Combined hooks preserve structured values through merges.

Authoritative Console builders annotate raw approved/deny decisions before flattening to verdict strings. `approval_was_unanswered(row, decisions)` must exclude a defaulted deny caused by Stop/unresolved cards. Cover build_tool_review_hook, legacy build_mcp_review_hook, build_local_review_hook, build_managed_skill_promotion_review_hook, build_virtual_cli_review_hook and build_raw_shell_review_hook wherever they flatten an authoritative decision. Approval scopes already recognized by each owner remain unchanged; an approval fact cannot create permission.

Direct MCP invocation annotates explicit user deny and permission-Off, not ask-with-no-callback despite shared refusal copy. Local invocation annotates only `LocalToolInvocationReason.APPROVAL_REFUSED` when authoritative, and `PERMISSION_OFF`. Existing `final_gate/approval_consumed/reason_code` facts identify approved execution and can preserve approved provenance when execution subsequently fails. Other tool providers stay conservative when no authoritative fact exists. No parsing of canonical refusal strings.

## Counter and batch algorithm

Initialize `consecutive_denials = 0` in `run_agent_loop`. Resolve review metadata for each call in dispatch order, not the up-front Trace decision loop. On each fully settled result:

1. If `ok=True`, reset.
2. If the dispatched result carries authoritative denied, increment even if review earlier approved: a final permission change/denial supersedes the earlier grant.
3. If review refused with authoritative denied, increment once.
4. Otherwise reset, including any approved call and any ordinary failure or unrelated block.

A legacy refusal is blocked in existing presentation, but has no new denial authority and resets this guard. Count once per call, never both review and invocation. Count both continuation-refusal and common dispatch result branches. A restored_pending synthetic refusal always resets; already-completed restored calls are not re-counted. A newly denied invocation during restore can count normally.

At the end of `for call in calls`, check cancellation first and preserve existing continuation-error exits. For a completed batch set `coherent_len = len(messages)` after all append/expand_restore_history operations succeeded, then if enabled and streak >= limit emit the terminal step/log and return RUN_STUCK. Do not drain a mailbox or call a model to obtain that coherent boundary. Do not clear a pending current batch merely to stop; every reviewed call gets its normal result. Existing unrelated guards may still stop earlier under their existing semantics.

Terminal copy: `Agent stopped: {count} consecutive tool calls were denied. Review the denial reasons or rephrase, then retry.` Emit one STEP_ERROR and one `_emit_record(deps, "error", content=summary, status=RUN_STUCK)`. No extra denial bodies, paths or tool payloads enter this record. `_outcome` retains the complete settled batch and existing earlier history. Do not put this system explanation into provider history.

## Config and ownership

Add RunBudget field default 3 and a strict small integer resolver in the pure models module for this field. Accept integer values and decimal digit strings; reject bool, negatives, fractional numbers, nonfinite values and invalid strings to default 3. Normalize the RunBudget field in frozen `__post_init__` with `object.__setattr__` so direct library callers have the same semantics; preserve existing step validation.

`console_run_budget` reads `[agents]` explicitly using a function-local config import. Child `clamp_child_budget` and `contain_child_budget` copy the field; `intersect_console_run_budget` uses existing zero-as-unlimited `ceiling` logic. Preserve field through other explicit budget constructions discovered in the targeted search. Do not fix unrelated inherited fields. Service children and resumed children already construct their own runtime; no FleetCoordinator mutation is needed.

The setting acceptance criterion specifies default 3 and explicit 0 disable. Document batch-boundary enforcement and per-run ownership in Related settings and the commented config example.

## Console evidence

`_agent_failure_visible_copy` already surfaces the last STEP_ERROR without double-prefixing `Agent stopped:`. `_finalize_agent_failure` preserves normal streamed placeholder content and appends a System row. Its missing-placeholder fallback currently may only append a failed assistant. Use a structured additive `RunOutcome.denial_count: int = 0` field if a narrowly scoped fallback discriminator is needed; populate only on denial terminal outcomes. This avoids identifying the feature by parsing summary text, and requires no persisted schema. The fallback should emit the System row for denial_count > 0 after selecting/marking the failed assistant, without changing other failure behavior. Preserve existing assistant content and state transition; avoid appending duplicate explanation to the assistant when the System row carries it.

Primary Console runs get the System explanation. Child runs retain STEP_ERROR, run-local JSONL and RUN_STUCK, and existing child completion delivery communicates it; do not fabricate child messages in the supervisor's transcript as if the supervisor tripped.

## ADR-154 compatible extension

### Consecutive-denial control provenance (TASK-18929)

`blocked` remains the presentation classification for all policy refusals; it does not assert that a user denied a request. The runtime denial guard uses a separate optional bounded approval decision (`approved`, `denied`) carried on ToolResult and structured ToolReviewDecision values. Owners set denied only for explicit user denial or configured permission Off, and preserve approved facts through ordinary execution failure. Unanswered, timeout, cancellation, missing callback, unavailable authority and workspace-root changes carry no denial decision. Legacy string verdicts preserve dispatch/refusal behavior but supply no new denial provenance. No result-text classifier supplies control authority.

The guard has a default limit of 3 and explicit 0 opt-out. It consumes results in call order and evaluates only the trailing streak at a completed batch boundary, allowing previously approved calls to execute and reset the streak. It stops before another model call, with complete tool reply pairing, a RUN_STUCK error step and local run-log explanation. Its counter is ephemeral per runtime invocation, including children and restored/resumed runs; it is not shared, persisted or inferred from history. Existing cancellation and continuation-persistence failures retain precedence.

These are additive internal contracts. Existing ToolResult positional construction, broad blocked presentation, legacy review strings, external provider messages and SQLite layout remain compatible. RunOutcome may carry an additive denial_count for terminal presentation without parsing text; this is not a durable counter. The field does not grant tool permission or change approval stamps. No SQLite migration is required.

ADR required: yes (new compatible extension; accepted ADR-078 stays immutable). ADR path: backlog/decisions/154-agent-denial-streak-boundary.md . Reason: bounded additive provider/review/runtime control provenance. Link ADR-078 as the unchanged broad outcome contract; ADR-131 and ADR-134/135 remain unchanged accounting/admission boundaries. Canonical decision: backlog/decisions/154-agent-denial-streak-boundary.md.
