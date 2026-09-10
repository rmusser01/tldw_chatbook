# Native Goal Runs Implementation Plan

> **For agentic workers:** Use the executing-plans skill to implement this plan task by task after design approval. Use subagent-driven-development only when delegation is authorized. Steps use checkbox syntax for tracking.

**Goal:** Let a Chatbook user start a bounded objective, receive independently recorded increments and evidence, and safely pause, stop, review and resume the work.

**Architecture:** Add an app-owned goal service above the native Console agent turn. Extend the current automatic-work attempt ledger and reuse the controller's provider preparation, capacity, permissions, CLI/skill/MCP tools, transcript, and review paths. Workflows authoring and whole external-agent iteration adapters can integrate with the same service later.
wtf 
**Tech stack:** Python >=3.11, Textual >=8.0.0,<9, Pydantic, existing SQLite/run-log infrastructure; no new runtime dependency.

**Spec:** [Review and proposed design](../specs/2026-09-08-gnhf-inspired-goal-runs-design.md).

**Status:** Implementation complete in the isolated native-goal-runs worktree, following the user's 2026-09-08 authorization and requested [preimplementation review](../reviews/2026-09-08-goal-runs-preimplementation-review.md). The feature belongs inside Chatbook and includes existing CLI tools. All five slices have task-scoped specification and quality approval. The three functional gaps and two minor issues from whole-branch review were fixed in `525c61a944`; the final scoped re-review approved `052fd4b95f` with no open goal findings. TASK-32116 through TASK-32120 are Done. Qualification limits and the remaining integration prerequisite are recorded below.

ADR required: yes
ADR path: `backlog/decisions/141-native-console-goal-runs.md`
Reason: Adds a durable goal lifecycle, new automatic submission/attempt semantics, storage, and long-lived Console UX. The [ADR is accepted](../../../backlog/decisions/141-native-console-goal-runs.md); the accepted extension preserves existing ADR-134/135 fleet semantics.

## Global constraints

- Python >=3.11; Textual >=8.0.0,<9; no new external runtime dependency.
- One active goal per Console runtime; no goal-created subagents in the first release.
- Both first and later goal iterations are automatic work in the same causal chain.
- Initial goal limits: 3 iterations, 32 model calls including helpers, 500000 budget tokens, 8192 maximum output tokens per call and 900 elapsed seconds. Each iteration additionally uses at most 8 model turns, 64 steps and 240 seconds, narrowed by remaining chain allowance and applicable executor limits.
- Goal enablement and `max_goal_*` policy are independent of fleet `autowake_enabled` / `max_autowake_*`. Both origins share accounting, physical capacity and manual reserves. Apply origin-aware policy at every admission/check; live increases cannot refill a launch snapshot.
- Goal deadlines include waits/pauses after first acceptance under ADR-134. A human response does not extend or refill an allowance.
- Preserve normal manual Console and fleet-wake behavior, including automatic/manual capacity reserves.
- Reuse tool permissions, skill trust/grants, configured executable integrations and the selected project binding. Existing CLI execution is in scope; do not infer its absence from the separate native raw-shell ADR. Tool selection narrows authority only, with no automatic approval or provider fallback.
- Objective and criteria each <=8 KiB UTF-8; report <=64 KiB; draft <=32 KiB; <=32 evidence references; <=8 learnings; checkpoint/learnings handoff <=16 KiB, with objective/criteria included separately and the complete request subject to stricter context limits.
- Retained verification records <=128 KiB each and <=4 MiB per goal; goal payloads also have a 128 MiB aggregate default ceiling. Reserve result space before execution. Explicit removal of settled private payloads retains accounting tombstones; active/uncertain work is protected. This is not a global cap on existing logs/artifacts.
- First scenario uses existing CLI validation and authorized file edits. A launch-bound `VerificationSpec` identifies the trusted check outside the editable fixture, arguments and input scope. Typed process outcomes and artifact versions provide evidence; `ToolResult.ok` and model text do not prove a passing check.
- `GoalToolScope` restricts catalog and runtime tools at discovery, schema and invocation boundaries, including restored calls. CLI scripts retain the executor's actual authority: scratch cwd and local file-tool confinement do not sandbox arbitrary script code.
- Result review and recovery of uncertain effects have separate states/actions. Quality approval cannot release workers, erase uncertain spend or authorize command replay.
- Run state and authorizing mutations belong to service/ledger owners, never widgets or model output.
- Keep automatic ledger/diagnostic projections body-free; keep AGENTS bodies ephemeral; store reports/evidence privately.
- Use canonical F9 Settings, current Textual workers and keybinding conventions. Do not bind terminal-convention/global keys to new screen actions.
- Targeted tests only unless the user explicitly requests a full sweep. Use source checks only for source facts, not as execution evidence.

## Before implementation

- [x] Confirm the native goal-loop direction with the user and review the proposal before implementation. Existing CLI capability is an implementation input. Review corrections and remaining validation are recorded in the linked review.
- [x] Read and reconcile the corresponding Backlog records: 32034–32037 (native ownership and budgets), 32077 and 32088–32095 (Workflows design/first milestone). Use the preserved native baseline below; Workflows remains an independent active branch. Recheck concurrent changes before integration.
- [x] Accept ADR-141 under the user's implementation authorization, recheck its identifier against 461 local refs and 12 worktrees, and add it to the ADR index.
- [x] File the five slices below through Backlog CLI in dependency order (TASK-32116 through TASK-32120). Slice labels below are planning labels, not invented Backlog IDs. Allocate/check IDs at filing time, give each measurable acceptance criteria, and link this plan, spec and ADR. Change each task to In Progress before adding its implementation plan or writing code.
- [x] Use the isolated `.worktrees/native-goal-runs` checkout, with the reviewed agent changes preserved at baseline `77bc58dc17`. The reviewed checkout is extensively dirty; do not stage or overwrite its unrelated changes. Record the precise implementation baseline and update file/version references if it has advanced.

Reconciliation snapshot (2026-09-09): runtime prerequisites TASK-32034–32037 are recorded Done in the preserved baseline; TASK-32037 explicitly reports its shared-checkout changes uncommitted. The original checkout remains at `22aa927f0287e84c76ab39836a05a028f2a73e82` on `docs/lesson-adr-number-collisions`. The separate `codex/workflows-local-file-to-note` branch is at `eda0e5c9f9`: its storage/readiness/runtime/recovery slices are recorded Done, adapters/editor remain In Progress, and integration/qualification remain To Do. Goal execution continues on the native Console contract; no Workflows implementation was imported. Integration must reconcile the preserved prerequisite baseline and concurrent shared Console/settings changes with their owners. The current 20 worktrees show no competing ADR-141 title.

## File responsibilities

| File | Responsibility |
| --- | --- |
| New `tldw_chatbook/Agents/goal_models.py` | Immutable requests/policies/tool scopes/verification specifications, strict report schema, typed evidence and decisions. No I/O. |
| New `tldw_chatbook/DB/goal_runs.py` | Goal/iteration/report persistence using AgentRunsDB connections and revision checks. No scheduling. |
| New `tldw_chatbook/Agents/goal_run_service.py` | Sole goal transition owner: create, checkpoint, pause, resume, stop, review and recovery. |
| New `tldw_chatbook/Agents/goal_iteration.py` | Bounded prompt handoff, strict report parsing, evidence resolution and progress/completion decisions. |
| New `tldw_chatbook/Chat/console_goal_runs.py` | Typed native dispatch authorization and bounded scheduling; adapts goals to existing Console submission. |
| Existing `Agents/automatic_work_budget.py`, `Agents/automatic_work_runtime.py`, `DB/automatic_work.py` | Goal-kind attempts sharing existing authority, accounting and recovery. |
| Existing `Chat/console_chat_models.py`, `console_chat_controller.py`, `console_agent_bridge.py`, `console_runtime.py` | Origin/dispatch/lifetime integration. Keep new logic in the focused modules above. |
| Existing `Chat/chat_persistence_service.py`, `console_fleet_wake.py` | Idempotent conversation provisioning and the shared runtime startup audit. Preserve chat/workspace and fleet ownership. |
| Existing `Agents/agent_models.py`, `agent_service.py`, `agent_runtime.py` | Goal-scoped runtime tools, typed termination reasons and narrowed iteration budgets; preserve manual defaults. |
| Existing `Skills_Interop/local_skills_service.py`, `skill_script_runner.py`, `MCP/client.py`, `ACP_Interop/runtime_process.py` | Reuse existing CLI/script invocation, executable transports and runtime launch. Connect their observed results to goal evidence; do not build another subprocess runner. |
| New `Widgets/Console/console_goal_setup_modal.py`, `console_goal_status.py`, `UI/Console_Modules/goals.py` | Setup, state projection, and implemented user actions. |

Paths below are repository-relative planning references. Tests are added alongside the corresponding runtime owner. Existing production files are inspected again at execution time because current agent changes are uncommitted.

## Task 1: Durable goal records and immutable launch contract

**Deliverable:** A goal can be created, inspected and reopened without running a model. Duplicate Start delivery cannot create duplicate goals or conversations. Its launch intent and budget chain are allocated atomically; cross-store conversation provisioning is recoverable.

**Files:**

- Create: `tldw_chatbook/Agents/goal_models.py`, `tldw_chatbook/DB/goal_runs.py`, `tldw_chatbook/Agents/goal_run_service.py`.
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py`, `tldw_chatbook/DB/automatic_work.py`, `tldw_chatbook/Chat/chat_persistence_service.py`.
- Create migration reference: `tldw_chatbook/DB/migrations/agent_runs_v15_to_v16_goal_runs.sql`, after confirming v15 is still the reconciled baseline. If another migration landed, allocate the next version and revise this path before coding.
- Test: `Tests/Agents/test_goal_models.py`, `Tests/DB/test_goal_runs.py`, `Tests/DB/test_goal_runs_migration.py`, `Tests/Chat/test_goal_conversation_provisioning.py`.

**Contract:** `GoalRunService.create(request: GoalRequest, *, launch_id: str) -> GoalSnapshot`; `GoalRunService.get(goal_id: str) -> GoalSnapshot`. `GoalRequest` contains objective, criteria, immutable provider/resource references, `GoalToolScope`, `VerificationSpec` records, execution binding references and policy. `GoalSnapshot` exposes `id`, `revision`, `status`, `chain_id`, `iteration_count`, `pause_reason` and policy/accounting projections. Display text cannot become authorization. Creation may return Starting while idempotent provisioning completes; it grants no dispatch authority.

- [x] Add failing tests for strict byte/numeric limits; reject booleans, negative/unbounded policy, unknown fields and oversized multibyte text. Cover zero admission and a missing/retargeted source binding.
- [x] Add real-SQLite tests: same launch ID plus identical payload returns the same goal; changed payload conflicts; reopening preserves criteria, chain and reports. Inject a failure between goal and chain creation and require both writes to roll back.
- [x] Persist the launch ID/payload hash, preallocated conversation UUID and chain before creating chat history. Extend the chat persistence owner with an idempotent provisioning path for that UUID; the underlying `ChaChaNotesDB.add_conversation` already accepts an explicit ID. Verify an existing row belongs to this launch, and reconcile workspace membership without duplicating or adopting another conversation. Do not pretend AgentRunsDB, ChaChaNotesDB and workspace storage share one transaction. Test crashes before/after each store write, repeated Start/retry and conflict/missing-binding states. Dispatch remains blocked until provisioning is complete.
- [x] Implement the three private tables named in the spec, parameterized writes and optimistic revision checks. Share the transaction with existing chain creation through a small internal connection-taking helper; do not nest independent commits. New APIs accept typed inputs only.
- [x] Update both `_CURRENT_SCHEMA_VERSION` and the recorded version row, following AgentRunsDB's guarded migration convention. Include `attempt_kind` on the existing automatic attempt storage, defaulting existing rows to `fleet_wake`; preserve its legacy physical table name to avoid a gratuitous storage rename. Goal iterations reference its attempt ID and do not invent source claims.
- [x] Run the new focused tests plus `Tests/DB/test_agent_runs_db.py`, `Tests/DB/test_automatic_work_migration.py` and affected `Tests/Chat/test_chat_persistence_service.py` cases. Self-review the schema/transaction boundary, update the task notes and commit only this slice's files in the implementation checkout.

Representative contract assertion:

```python
first = service.create(request, launch_id="start-1")
again = service.create(request, launch_id="start-1")
assert again.id == first.id
assert again.chain_id == first.chain_id
assert service.get(first.id).iteration_count == 0
```

Here `service` is a `GoalRunService` bound to a real temporary AgentRunsDB; `request` is a valid request built in this test module. Fixtures must reopen the same database file for persistence assertions.

## Task 2: One native iteration with durable dispatch authority

**Deliverable:** Execute exactly one goal iteration, including a real CLI check, through the real Console path while sharing permissions, accounting, capacity and cancellation. Repetition is not enabled yet. Prove this vertical path before UI work.

**Files:**

- Create: `tldw_chatbook/Chat/console_goal_runs.py`.
- Modify: `Agents/goal_run_service.py`, `Agents/automatic_work_budget.py`, `Agents/automatic_work_runtime.py`, `Agents/agent_models.py`, `Agents/agent_service.py`, `Agents/agent_runtime.py`, `DB/automatic_work.py`, `Chat/console_chat_models.py`, `Chat/console_chat_controller.py`, `Chat/console_agent_bridge.py`, `Chat/console_runtime.py`, `Chat/console_fleet_wake.py`, `config.py` under `tldw_chatbook/`.
- Test: `Tests/DB/test_goal_attempts.py`, `Tests/Chat/test_console_goal_dispatch.py`, `Tests/Chat/test_console_goal_authority.py`, `Tests/Chat/test_goal_cli_verification.py`.

**Contract:** `AutomaticWorkLedger.prepare_goal_iteration(goal_id: str, *, owner_id: str) -> GoalAttempt`; `accept_goal_iteration(attempt_id: str, *, owner_id: str) -> bool`, where only the first successful acceptance grants dispatch. `ConsoleGoalCoordinator.dispatch_once(goal_id: str) -> GoalIterationResult` is async and owns a typed `GoalIterationAuthorization`. Extend `submit_draft` with a typed `goal_authorization` argument only for `GOAL_ITERATION`. `GoalIterationResult` carries the exact goal, ordinal, attempt, native run ID, `RunOutcome`, typed termination reason and observed tool records; it carries no inferred completion verdict. New optional native outcome metadata must preserve existing callers.

- [x] Add failing SQLite race tests: two concurrent prepares/accepts start one increment; wrong goal/session/owner is refused; a fleet-wake claim still requires real survivor records. Empty `run_ids_json` is legal only for a goal-kind attempt proven by its goal/iteration link.
- [x] Validate attempt kind in read/accept/abort/complete and fleet-specific result/delivery queries. Preserve shared occupancy across kinds. Test that wake APIs cannot authorize or settle a goal attempt, goal APIs cannot touch survivor claims, and repeated terminal callbacks grant no new dispatch. A goal context requires its current accepted goal attempt; do not inherit the fleet allowance for surviving children to use a completed attempt.
- [x] Add a recording provider at the real controller/gateway boundary. Force acceptance persistence to fail and assert zero helper/model/tool calls. Exercise cancellation during preparation and verify primary occupancy is released only through its owning cleanup path.
- [x] Implement goal acceptance in the existing FULL transaction policy. Use the same chain for every call; acceptance of the first goal increment consumes a generation. Refusal before dispatch refunds only that generation reservation. Keep estimates on ambiguous completion.
- [x] Share the runtime startup audit and its owner ID before either coordinator admits work. The existing fleet coordinator calls the global `ledger.recover`; do not let constructing the goal service call it again and revoke a live fleet owner. Starting/attaching a screen is not recovery. Test lazy goal-service creation during live fleet work and both coordinators waiting on one startup failure/success.
- [x] Add distinct goal enablement/finite settings and an origin-aware policy resolver in automatic context checks, output caps and call/attempt admission. Test fleet disabled + goals enabled, the reverse combination, lowered settings during approval and increases after pause. Narrow `RunBudget` per iteration and preserve typed limit/permission/unknown-effect outcomes; never classify `stuck` or an exception by matching prose. Refuse new work after a limit while retaining a tool until its executor settles.
- [x] Add trusted submission handling without growing a second copy of the controller. In particular, `submit_draft` currently clears automatic context for every origin other than `AGENT_WAKE`; include validated goal origin there and at every automatic dispatch branch. Audit composer preservation, history provenance, native-only admission, preparation calls, primary reservations, review gates and leave-Console behavior. Do not obtain manual budgets through the new path.
- [x] Add an explicit goal request-history branch at the existing preparation seam. Persist the transcript normally but send only the approved goal handoff plus current authorized context at iteration start. Do not restore an earlier iteration's provider continuation or append its entire conversation history. Keep within-iteration continuation intact. Capture final provider requests in tests after all context injections and prove unrelated/old transcript bodies are absent.
- [x] Disable child-agent spawning for goal runs at runtime configuration. Keep direct `run_skill_script` and MCP tool calls available: they do not require spawning a child agent. Test the actual bridge closure through `SkillsScopeService` to `LocalSkillsService.run_skill_script` and `run_script_subprocess`, including remembered grants and rejection of a modified/revoked skill. Reject a model route without native agent capability before dispatch; this native-turn requirement does not exclude external executable tools.
- [x] Enforce `GoalToolScope` on catalog and runtime schemas, find/load results, restored calls and the last dispatch boundary after approval. `config.allowed_tools` alone is insufficient: runtime schemas and direct callbacks bypass that filter. Test an unadvertised script/install tool call, later-loaded disallowed tool, changed skill fingerprint and retargeted MCP server. Allowed CLI calls continue through their existing gates; a broader remembered grant cannot override the narrower goal scope.
- [x] Capture `ScriptRunResult` in a typed goal evidence observer before the bridge converts it to `ToolResult.content`: exit code, stdout/stderr, timeout/output-cap flags, duration, actual verifier/args and output references. Do not alter the meaning of `ToolResult.ok` for ordinary chat or parse printed `exit_code` text as proof. Test nonzero exit and timeout with `ToolResult.ok=True`, spoofed stdout, missing exit status and stale callback identity. Preserve scratch-cwd semantics and the existing POSIX executor boundary; it is not a filesystem/network sandbox. Pass remaining time into supported script limits and retain ownership until actual cleanup.
- [x] Start `test_goal_cli_verification.py` here with a deterministic provider and real controller/agent dispatch into an independently trusted temporary skill and real subprocess. Assert that a failed command produces typed failed evidence and successful invocation alone cannot complete a goal. Existing `test_e2e_run_skill_script.py` captures the bridge closure by intercepting `AgentService`; it is useful regression coverage, but does not substitute for this full dispatch test.
- [x] Run new tests plus targeted regressions in `Tests/Chat/test_automatic_wake_dispatch.py`, `test_automatic_provider_budget.py`, `test_automatic_approval_dispatch.py`, `test_console_runtime_lifetime.py`, `Tests/Agents/test_execution_capacity.py`, `Tests/Skills/test_e2e_run_skill_script.py`, and `Tests/Skills/test_skill_script_runner.py`. Review/commit the slice and record evidence.

The dispatch integration must prove this ordering at the actual boundary:

```text
validated goal identity
  -> primary/capacity reservation
  -> FULL accepted-attempt commit
  -> automatic context binding
  -> any billable preparation and exact-request reservation
  -> native agent/tool dispatch
  -> owned-worker settlement and exact iteration result
```

A changed provider/tool/source binding pauses the goal; it does not silently refresh an immutable launch into a new authority.

## Task 3: Bounded memory, observed progress and completion review

**Deliverable:** Each finished turn becomes a durable, inspectable checkpoint. False success reports and repeated no-ops cannot produce verified completion.

**Files:**

- Create: `tldw_chatbook/Agents/goal_iteration.py`.
- Modify: `Agents/goal_models.py`, `Agents/goal_run_service.py`, `DB/goal_runs.py` and, only where retrieval integration requires it, `Agents/run_log_search.py` under `tldw_chatbook/`. `run_log_eviction.py` owns send-context trimming, not persistent evidence retention.
- Test: `Tests/Agents/test_goal_iteration_report.py`, `Tests/Agents/test_goal_progress.py`, `Tests/Agents/test_goal_memory.py`, `Tests/DB/test_goal_evidence_retention.py`.

**Contract:** `parse_iteration_report(text: str) -> IterationReport`; `build_goal_handoff(goal: GoalSnapshot, checkpoints: Sequence[GoalCheckpoint]) -> str`; `evaluate_iteration(report: IterationReport, evidence: Sequence[GoalEvidence], previous: GoalCheckpoint | None, criteria: Sequence[GoalCriterion]) -> GoalDecision`. `GoalDecision.action` is one of `continue`, `pause`, `awaiting_result_review`, `recovery_required`, `completed`; it includes bounded reason/check metadata. Only a runtime evidence resolver constructs `GoalEvidence`. `GoalRunService.checkpoint(result: GoalIterationResult) -> GoalSnapshot` atomically stores report/evidence/checkpoint/decision and the matching attempt transition in one FULL transaction. Only conclusively settled work completes its attempt; uncertain work retains a recovery-required attempt and conservative charges, even when available evidence was saved. A repeated identical payload is idempotent; conflicting content for the same attempt fails. Settled provider accounting is not charged again.

The immutable launch field `GoalRequest.human_review_required` defaults to `True`. Explicit `False` requires at least one launch-bound verifier and declares those checks sufficient for objective-only completion. Do not infer this choice from prose or model output; preserve older launch identity bytes through the safe default.

`IterationReport` fields are `summary`, `learnings`, `next_action`, `candidate_draft`, `evidence_ids`, and `completion_recommended`. Exact per-field bounds follow the spec. `GoalCheckpoint` carries the accepted report, normalized evidence/draft digests and criterion results; it is private payload, not an automatic-work table row.

- [x] Write parser tests for strict booleans/types, extra fields, malformed/oversized JSON and malicious identifiers. A model-supplied `verified`, `budget`, `provider`, `workspace_root` or `permission` field must be rejected.
- [x] Write completion tests in which the model recommends success but no evidence resolves, a reference belongs to another run, a report repeats previous evidence, and structural checks pass while a human criterion remains unreviewed. None may become verified completion. An already-satisfied empty goal may complete from actual checks without performing a speculative edit.
- [x] Implement report parsing after the native turn. Do not add an unbudgeted model judge or a parser-repair loop. A malformed report is one failed attempt with its spend retained. Resolve evidence IDs through exact run/source ownership, or explicitly accepted earlier checkpoints of this goal. Persist checks separately from model assertions. Match each CLI result against its launch-bound `VerificationSpec`, not whichever command the model chose to run.
- [x] Test a passing verifier followed by an agent edit, a manual edit before review, a changed verifier/argument, wrong working target and a source changed during checking. Compare the declared input manifest before/after verification and at completion/approval; changed or indeterminate versions invalidate the check. Record the checked version separately from live workspace state. Treat incomplete required output or an MCP tool without a verifiable structured adapter as unavailable for objective completion.
- [x] Inject failures between report/evidence insertion, goal revision/counter update and attempt completion and require one atomic rollback, retained accepted work and zero successor dispatch. Test duplicate identical checkpoint delivery and conflicting payload delivery. There is no rollback of tool effects when persistence fails; recovery remains explicit.
- [x] Implement deterministic progress from novel observed source records, changed draft digest or newly satisfied criteria. Normalize/deduplicate stable source identities so repeatedly retrieving the same unchanged source is not progress. Summary wording alone cannot reset the no-progress counter. Two consecutive no-progress increments pause; three consecutive failed increments pause, subject to earlier caps.
- [x] Build bounded next-iteration context from objective/criteria, latest checkpoint and recent learnings. Test UTF-8 limits, unchanged objective, excluded automatic project-instruction bodies and retained protocol coherence. Verify the actual outgoing second/third requests, not just the handoff helper's string length; neither old transcript history nor an earlier provider continuation may reintroduce prior payloads. Pause if mandatory content cannot fit after all preparation.
- [x] Copy necessary evidence into private goal records under the 128 KiB per-record, 4 MiB per-goal and 128 MiB aggregate payload bounds. Reserve enough result capacity before admitting an iteration, including space for its report. Test repeated goals exhausting the aggregate allowance, script artifact pruning, missing external sources, oversized output and removal of settled payloads. Removal never deletes workspace files or accounting history. Do not add a run-log pinning framework or mistake send-context eviction for disk retention.
- [x] Run the four new focused modules and affected existing run-log tests. Self-review privacy/retention and completion checks, document tradeoffs and commit.

Representative adverse report:

```json
{
  "summary": "Finished the comparison",
  "learnings": [],
  "next_action": "",
  "candidate_draft": "",
  "evidence_ids": ["a-record-owned-by-another-conversation"],
  "completion_recommended": true
}
```

Expected result: no verified evidence or completed criterion; retain the report as an unsuccessful checkpoint and expose the evidence error. Never follow an evidence ID as a filesystem path or URL supplied by the model.

## Task 4: Bounded repetition, wait states and restart recovery

**Deliverable:** A goal can run multiple increments autonomously, stop predictably, and recover without duplicate effects or replenished budget.

**Files:**

- Modify: `Agents/goal_run_service.py`, `Chat/console_goal_runs.py`, `Chat/console_runtime.py`, `DB/goal_runs.py`, `DB/automatic_work.py` under `tldw_chatbook/`.
- Test: `Tests/Agents/test_goal_run_service.py`, `Tests/Chat/test_console_goal_scheduling.py`, `Tests/Chat/test_console_goal_recovery.py`, `Tests/DB/test_goal_restart_process.py`.

**Contract:** `GoalRunService.pause(goal_id: str) -> GoalSnapshot`, `stop(goal_id: str) -> GoalSnapshot`, `resume(goal_id: str, *, expected_revision: int) -> GoalSnapshot`, `review_result(goal_id: str, *, expected_revision: int, checkpoint_id: str, artifact_digest: str, accepted: bool) -> GoalSnapshot`, and `project_recovery(audit: RuntimeRecoveryResult) -> Sequence[GoalSnapshot]`. The audit comes from the single trusted runtime recovery owner. Result review operates only on `awaiting_result_review` and revalidates evidence freshness. Recovery of uncertain work has a separate typed `resolve_recovery(..., resolution: RecoveryResolution)` contract: it records an adapter-proven outcome or closes the interrupted goal while preserving unknown effects/charges; it never grants replay or impersonates quality approval. Resume is allowed only from a clean settled checkpoint with remaining allowance. The coordinator schedules only service-admitted work; it does not own another budget counter.

- [x] Test two successive native increments with one immutable chain; a third reaches the configured cap and a fourth is never dispatched. Include a failed increment and verify charges persist. For different limit combinations, assert the earliest applicable bound wins.
- [x] Test denied permission, revoked bindings, no-progress, permanent error, typed pre-effect rate rejection and unknown-effect timeout as separate outcomes. Count provider attempts in the retry interval; a quiet transcript is not proof against a retry storm.
- [x] Implement event-driven continuation after durable checkpoint settlement. Use the existing automatic primary/manual-reserve admission. A busy manual session keeps its priority, and capacity refusal registers bounded retry/release handling without consuming a new generation or busy-looping. Persist retry time and keep it inside the original elapsed deadline.
- [x] Implement pause-after-increment and cooperative Stop now. Never free a root lease because a row became terminal. Keep late worker output attached to the original increment; it cannot advance a stopped goal. Confirm no goal children can keep mutating after a checkpoint.
- [x] Count approval waits, admitted work and Stopping as active ownership; settled paused/review goals may remain in history. Manual Send in a goal conversation pauses future increments and obeys current occupancy. It cannot silently mutate launch criteria or scope. A separate ordinary conversation's composer remains independent.
- [x] Project explicit startup recovery from the shared audit: old-owner prepared and accepted/ambiguous work become `recovery_required`; clean checkpointed goals previously eligible to continue project Paused until explicit Resume; an existing settled result review remains available without dispatch. Reject a quality-review call for a recovery-required goal, a stale checkpoint/artifact and repeated contradictory decisions. Unknown charges remain reserved; a human acknowledgment does not prove that a command never ran. Recheck provider/resource identity and remaining allowance on resume. An expired allowance cannot be revived by approval, navigation or a settings increase.
- [x] Add a child-process harness using a temporary profile/database: terminate after durable acceptance and before completion, restart, and assert zero redispatched provider/tool operations and retained reservations. A second case stops after a complete checkpoint and requires explicit Resume. This is process-restart evidence, not simulated power-loss certification.
- [x] Run the new modules and relevant `Tests/DB/test_automatic_runtime_owner.py`, `test_automatic_work_deadlines.py`, `Tests/Chat/test_automatic_wake_recovery.py`, and `test_automatic_wake_scheduling.py`. Review/commit and record the exact scope of tested cancellation guarantees.

## Task 5: Console controls and first useful end-to-end result

**Deliverable:** A user can launch, observe and control a CLI-backed goal through real Textual controls, inspect actual verification results and edits, and preserve work across navigation/restart.

**Files:**

- Create: `tldw_chatbook/Widgets/Console/console_goal_setup_modal.py`, `console_goal_status.py`, and `tldw_chatbook/UI/Console_Modules/goals.py`.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/config.py`, `tldw_chatbook/Chat/console_runtime.py`; use the existing Console action/command-palette registration site found during execution.
- Test: `Tests/UI/test_console_goal_setup.py`, `Tests/UI/test_console_goal_controls.py`, `Tests/UI/test_console_goal_navigation.py`, `Tests/Chat/test_goal_cli_verification.py`.
- Docs: `Docs/User_Guide/console/agent-runs-and-tools.md`, `Docs/User_Guide/settings.md`, relevant diagnostic/private-data inventory if changed by the implementation.

**Contract:** UI consumes `GoalSnapshot` and service methods from slices 1/4. Setup passes immutable source/provider/tool/check selections and a stable launch ID to the service; it does not independently create a conversation. Selection and review identify the exact goal/iteration/artifact, never whichever run was most recent.

- [x] Add mounted Textual tests that press actual Start/Pause/Stop/Resume/Review controls, inspect both displayed state and resulting service/provider activity, and preserve an ordinary conversation's composer draft. Include stale checkpoint review and double Start.
- [x] Implement the minimal setup form/status card. Explain effective executor authority, finite goal/iteration limits, unknown usage and the elapsed deadline. Expose the independent `agents.goal_runs_enabled` gate and finite goal settings through canonical F9 Settings. Keep fleet controls' existing meaning. Present quality review and interrupted-execution recovery as distinct actions. Provide explicit removal of settled goal payloads when reclaiming the bounded history store, preserving accounting tombstones and protecting active/uncertain work.
- [x] Keep the runtime alive across Console unmount. If approval cannot be handled headlessly, save an actionable wait/refusal using existing behavior. At shutdown, refuse new increments before draining provider/tool owners and closing stores.
- [x] Extend the slice-2 end-to-end fixture with a temporary editable project containing invalid input, an independently trusted validation skill outside that project, a deterministic recording provider, real local file tools, real subprocess execution, real SQLite and the actual controller/bridge. The verifier receives the project path explicitly. Prove a first failed check, a later authorized file correction, and a passing rerun of the unchanged verifier across at least two increments. Assert actual exit statuses, stdout/stderr provenance, artifact freshness, file diff, preserved counters and the fixture's expected file effects. Scope local file tools and inspect designated external sentinels; do not claim the trusted CLI process was OS-confined to that project. A model changing only its report cannot complete the goal. State POSIX-only qualification of this skill path separately from other configured executors.
- [x] Verify the rendered layout at 80x24 and the project's ordinary terminal size. Check keyboard focus, wrapped wait reasons, exact evidence navigation and only implemented footer actions. Regenerate CSS only if styling changed, using the existing build process.
- [x] Run the new integration/UI modules and the touched settings/runtime tests. Obtain an explicit local-model/live-provider test configuration before any real model demonstration; save the actual command/result/diff trace and inspect the final corrected artifact. If live configuration is unavailable, leave that acceptance criterion open. The deterministic test must still execute a real CLI process; a mocked process result is not CLI integration evidence.
- [x] Run scoped Ruff/format and whitespace checks on modified files. Run applicable migration/private-data/diagnostic architecture checks because these owner kinds changed; do not run the full suite without permission. Finish self-review, update Backlog AC/notes and relevant docs, then commit the scoped slice.

Task5 qualification: independent specification and quality review approved the slice after fixes for restart hydration, binding-dependent tool choices, submitted summary and older-history access. The carried form-state finding was fixed and approved in the final fix wave. The configured local model did not produce the requested corrected live artifact: both bounded attempts yielded malformed reports and no tools, so the inspected final artifact remained invalid. The actual failed traces are retained; successful correction was demonstrated by the deterministic provider through the real CLI path. This deviation and all baseline check failures are recorded in the [qualification report](../reviews/2026-09-09-goal-runs-qualification.md), with no live-success claim.

## Final whole-branch fix wave

ADR required: yes
ADR path: backlog/decisions/141-native-console-goal-runs.md (Accepted; selected-resource and confirmation contract clarified before fixes)
Reason: restore immutable setup and usable selected-resource integration within the existing native authority boundary; no new executor, grant or allowance owner.

Affected Backlog tasks TASK-32117, TASK-32118 and TASK-32120 were reopened with explicit behavioral acceptance criteria before code. One implementer addressed the five final-review findings and one scoped re-review approved the resulting diff. All three tasks are now Done with their amended criteria checked and implementation notes retained.

- [x] Keep every displayed launch field identical to the submitted request across awaited validation; restore usable controls on failure, including tool choices.
- [x] Include bounded exact selected verifier invocations, execution root and checked inputs in actual initial/later model requests without objective duplication or fixture-only knowledge. Canonical executor owners provide the executable call details; execution-time verification remains authoritative. Keep the existing 128 KiB launch ceiling, 16 KiB checkpoint memory and complete prepared-request budget; refuse/pause explicitly if mandatory context cannot fit.
- [x] Make additional selected read-only source bindings usable through existing bounded context/file tools and permission gates. Test actual selected reads, unselected sibling refusal, primary-only writes, identity changes and real native request/tool behavior. Source data never becomes automatic project instructions or permission grants.
- [x] Remove the blank separator before ADR-141 in the existing index table. The earlier separator before ADR-129 was also removed so all affected entries render in the table; entry contents are unchanged.
- [x] Reproduce affected behavior before changes, run focused amended gates/static checks, update actual traces and user docs, then complete the single independent fix-wave re-review. Preserve existing live-model and baseline-diagnostic limitations honestly.

Example targeted verification commands, once the new files exist and the dev environment is activated:

```bash
python -m pytest Tests/Agents/test_goal_models.py Tests/Agents/test_goal_iteration_report.py Tests/Agents/test_goal_progress.py Tests/Agents/test_goal_memory.py Tests/Agents/test_goal_run_service.py
python -m pytest Tests/DB/test_goal_runs.py Tests/DB/test_goal_runs_migration.py Tests/DB/test_goal_attempts.py Tests/DB/test_goal_evidence_retention.py Tests/DB/test_goal_restart_process.py
python -m pytest Tests/Chat/test_console_goal_dispatch.py Tests/Chat/test_console_goal_authority.py Tests/Chat/test_console_goal_scheduling.py Tests/Chat/test_console_goal_recovery.py Tests/Chat/test_goal_cli_verification.py
python -m pytest Tests/Chat/test_goal_conversation_provisioning.py
python -m pytest Tests/UI/test_console_goal_setup.py Tests/UI/test_console_goal_controls.py Tests/UI/test_console_goal_navigation.py
git diff --check
```

Use the focused existing regressions listed in each slice as well. Record overlapping runs separately; do not add their counts into a misleading total. A missing module, skipped real dispatch or substituted controller is not a passing acceptance test.

## Follow-on plans, after the native milestone

| Milestone | Prerequisite and scope | Completion evidence |
| --- | --- | --- |
| Workflows invocation | Reconcile with existing tasks 32088–32095. Add an advertised local goal-operation contract invoking the same service; explicitly preserve unsupported/server definitions. Keep workflow and goal run identity separate and apply both owners' limits without double billing. | A saved workflow starts the exact goal operation once, follows its result in Console, and preserves both identities/allowances across a wait. No silent server fallback. |
| Repository checkpoints | CLI verification already ships in the first milestone. Add worktree/commit policy using the selected existing CLI capability: repository scope, dirty-tree handling, hooks/signing and commit repair. Reuse change review. | Real temporary Git repos demonstrate preservation of dirty files/index, failed hook repair, no commit before observed verification, truthful cleanup and no push/merge. |
| Mutating domain tools | Admit tool/effect classes through current permissions and domain services; record idempotency/replay contracts. | Notes/files/network effects have correct uncertain-outcome review; a retry cannot duplicate an unconfirmed mutation. |
| Closed-app scheduling | Use existing Scheduling/server-offload ownership and an explicit backend contract. | Execution/cancellation limits are enforced by the actual background/server owner; a UI disconnect is not reported as a stopped remote run. |

These are separate future plans rather than hidden acceptance requirements for the first release. Existing CLI tools and verification are already included in slices 2/3/5. A whole external-agent adapter is a separate protocol/accounting integration choice; it is not evidence that Chatbook lacks CLI execution.

## Review checklist

- [x] Every model/helper/tool call is behind the same accepted attempt and automatic budget, including the first increment.
- [x] Goal policy is independent of fleet enablement and each native iteration has finite sublimits and typed termination reasons.
- [x] Existing fleet wakes and ordinary manual chat retain their semantics; goal attempts have their own validated kind rather than fabricated survivor claims.
- [x] Catalog, runtime and progressively loaded tools obey the same immutable goal scope at actual dispatch; CLI authority is described accurately.
- [x] Completion and human review use typed verifier outcomes and the exact current artifact/checkpoint identity; quality approval cannot resolve interrupted effects.
- [x] Launch provisioning is idempotent across stores; checkpoint and attempt settlement share one FULL transaction; startup recovery runs once for both coordinators.
- [x] Pause/retry/restart cannot refill counters, change resource authority or overlap a still-owned effect.
- [x] Source/body privacy, bounded payload retention and fresh iteration context are verified at the persistence and actual provider-request boundaries.
- [x] Console controls invoke real actions, preserve ordinary drafts, and survive screen navigation.
- [x] The first local scenario works through the actual runtime and CLI with a deterministic provider; the unsuccessful live-model trials and their limits are stated accurately.
- [ ] Backlog/ADR identifiers, migration version, concurrent work and documentation are reconciled before integration.

Execution evidence lives in the [qualification report](../reviews/2026-09-09-goal-runs-qualification.md), and the [final review record](../reviews/2026-09-09-goal-runs-final-review.md) closes all five findings. The sole unchecked item is integration with independently owned prerequisites and concurrent work. The branch and worktree were preserved at the initial handoff; the user subsequently requested one PR against `dev` containing all session changes and the `/goal` composer follow-up tracked by TASK-32194. The final local identifier scan found no conflicts, but it does not replace reconciliation at integration time. Four proven pre-existing test failures across two gates and unsuccessful live-model correction remain explicit qualification limits.
