# Native Shared-Worktree Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute approved coding phases with native agents in one existing worktree, retaining truthful results and explicit restart recovery.

**Architecture:** Keep execution in ConsoleRuntime, the controller/bridge and AgentService/FleetCoordinator. Extend existing automatic admission, physical ownership, atomic file tools, evidence and Change Review owners with workflow-specific authority, activity reservations and durable results. Views project state; they never launch workers or own grants.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, SQLite, Pydantic, existing native provider/tool integrations, pytest. No new runtime dependency.

**Spec:** [Approved shared-worktree design](../specs/2026-11-01-native-phased-agent-workflows-design.md)

**Decision:** [ADR-285](../../../backlog/decisions/285-native-phased-agent-workflows.md)

**Planning task:** [TASK-38523](../../../backlog/tasks/task-38523%20-%20Plan-native-shared-worktree-workflow-implementation.md)

**Baseline:** finalized design commit `8f989edc9f`. Source mapping was inspected in the current dirty checkout; unrelated staged additions are not implied to be committed prerequisites. Recheck the integrated source, symbols and schema before execution.

**Status:** Implementation in progress. Tasks 1–7 passed their independent review gates; Tasks 8–10 remain pending. Runtime/UI delivery is incomplete.

```text
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Direct implementation of the accepted design. Amend that ADR/spec
before changing scope, authority, timing, recovery or storage contracts.
```

## Global Constraints

These constraints apply to every task; quoted requirements are copied from the spec.

- "Tasks work in the same selected worktree, including its existing uncommitted content."
- "Stop preserves edits." No child/integration checkout, automatic Git mutation, PR publishing, daemon or general DAG engine.
- "Tool access narrows existing permissions; plan prose and task results grant no authority."
- "Goal and fleet absolute deadlines remain unchanged." Goals retain their no-child policy.
- "Restore, import, UI selection or a transcript row cannot reconstitute an old execution capability."
- "Opening history or constructing the workflow service never runs a second global sweep."
- "Detailed native run logs remain supplementary and may be disabled or truncated without losing required handoffs."
- "No workflow feature silently enables capture or turns snapshots into a general rollback guarantee."
- The execution coordinator is app-owned; no additional repeating workflow poller, permission store, execution loop or budget ledger.
- Read each Backlog task before changing code, set it In Progress and add its own Implementation Plan then. The linked To Do records intentionally contain outcomes, dependencies and document references rather than premature per-task plans.
- Honor the user's shared-worktree instruction during execution too. Preserve unrelated work and stage/commit only the current task's paths/hunks. Never run broad add/reset/clean/stash commands.
- Run targeted checks only. Do not run a full suite without the user's explicit request. Read the applicable testing/live-verification lessons and UI constitution before those slices.

### Bounds and policy names

| Contract | Exact value |
| --- | --- |
| Plan | 128 KiB; six execution phases; twelve task definitions; 64 footprint entries per task |
| Required result storage | 128 KiB per record; 4 MiB per workflow; 128 MiB aggregate workflow payloads |
| Settings gate | `[agents] workflow_runs_enabled = false` |
| Accepted planning/execution attempts | `max_workflow_generations = 8` |
| Task launches | `max_workflow_child_launches = 12` |
| Provider calls, including helpers/fallback | `max_workflow_model_calls = 32` |
| Tokens | `max_workflow_budget_tokens = 500000` |
| Per-call output tokens | `max_workflow_output_tokens = 8192` |
| Cumulative active seconds | `max_workflow_wall_seconds = 900`; label as active time, not ordinary elapsed deadline |
| History/task page | At most 50 rows |

Use existing finite-integer parsing; booleans/nonfinite values never become limits. Zero disables admission. Limits snapshot at creation, intersect live reductions, and never refill on approval/revision/resume. Normal task-native budgets also apply. Insufficient remaining allowance pauses useful work honestly rather than promising plan completion.

## Before starting Task 1

- [ ] Read the spec, ADR, `Docs/Architecture/{console,agent-runtime,tool-catalog,console-file-authority,database-layer,backup-recovery}.md` and `backlog/docs/lessons-{testing-evidence,live-verification,backlog-hygiene}.md`.
- [ ] Record exact checkout/root, HEAD, source changes and AgentRuns schema. Schema 24 was inspected; allocate the next actual integrated version at execution, without reserving or renumbering a shipped migration here.
- [ ] Reproduce the historical bootstrap failure with existing isolated configuration fixtures:

```bash
.venv/bin/python -m pytest -q --tb=short Tests/Agents/test_fs_read_ledger.py Tests/Agents/test_local_tool_provider.py::test_two_writer_race_refuses_second_writer Tests/Agents/test_fleet_runtime.py::test_two_children_run_concurrently_and_wait_collects_both
```

Expected prerequisite: those existing paths execute their claimed bodies. Earlier observations were seven ledger passes, five provider/fleet failures and a teardown error; the isolated provider failed at `raw_source_selection_changed` before useful execution. Attribute any current failure before editing workflow code. Do not bypass bootstrap/storage admission, alter live config, or call an unrelated red green by testing a replacement harness. If a prerequisite has an existing task owner, report that dependency instead of repairing it under an unrelated slice.

- [ ] Pin tests to private temporary profile/data/config paths and existing bootstrap markers. Use actual native owners for acceptance tests; scripted provider turns are suitable for deterministic control flow, not live-provider qualification.

## File and ownership map

New modules are proposed exact paths; create only the responsibility described. Existing large owners receive narrow wiring, not wholesale rewrites.
Product paths in task file lists are relative to `tldw_chatbook/`; `Tests/`,
`Docs/` and `backlog/` paths are repository-relative. Inline API signatures are
contracts to implement, not claims that those new APIs already exist.

| Owner | Responsibility |
| --- | --- |
| `Agents/workflow_models.py` | Strict immutable requests, revisions, tasks, decisions, results and snapshots; reuse compatible Goal binding/provider value types and AutomaticWorkLimits. |
| `DB/workflow_runs.py` | Workflow structural state, private payload reservations, revision and task-launch/result CAS in AgentRunsDB. |
| `DB/recovery_workflow_schema.py` | Exact installed constructor/migration catalog delta and inert restored-row validation. |
| `Workspaces/worktree_activity.py` | Runtime-owned complete read/write/opaque activity claims, tied to ExecutionOwner drain. No execution or accounting. |
| `Agents/workflow_task_scope.py` | Private per-run task authorization and restriction adapters; actual provider ownership/path validation remains authoritative. |
| `Agents/workflow_run_service.py` | Finite offloaded lifecycle/read/checkpoint/evidence/recovery operations over existing persistence owners. |
| `Chat/console_workflow_runs.py` | Phase authority, native scheduling, dependency settlement and Console lifecycle integration. |
| `UI/Console_Modules/workflows.py` | Late-bound view/controller wiring only. |
| `Widgets/Console/console_workflow_setup.py`, `console_workflow_status.py` | Setup and bounded status/history/detail projections, using existing patterns. |
| Existing `AgentService`, controller/bridge/runtime, automatic ledger/context, file workers and Change Review | Keep provider execution, permissions, accounting, physical retirement and change ownership. |

Do not add a child executor alongside AgentService, a second database connection pool, generic plugin registry or abstract workflow engine. Put SQL behind the existing AgentRunsDB transaction owner. The first storage slice includes constructor/recovery/backup compatibility so there is no interval where shipped schema cannot be restored.

## Delivery order

| Step | Task | Dependencies | Reviewable deliverable |
| --- | --- | --- | --- |
| 1 | [TASK-38524](../../../backlog/tasks/task-38524%20-%20Persist-bounded-workflow-plans-and-private-task-results.md) | Planning | Passive durable plans/results with exact recovery catalogs |
| 2 | [TASK-38525](../../../backlog/tasks/task-38525%20-%20Admit-workflow-attempts-with-cumulative-active-time-budgets.md) | 1 | Typed admission and distinct workflow timing/settings |
| 3 | [TASK-38526](../../../backlog/tasks/task-38526%20-%20Coordinate-complete-worktree-footprints-across-native-runs.md) | 1 | Pure/testable activity reservation owner |
| 4 | [TASK-38527](../../../backlog/tasks/task-38527%20-%20Preserve-observed-content-preconditions-through-edits-and-patches.md) | 1 | Existing atomic file operations retain observed digests |
| 5 | [TASK-38528](../../../backlog/tasks/task-38528%20-%20Enforce-workflow-task-scope-at-native-tool-invocation.md) | 1–4 | Enforced run-bound task scope |
| 6 | [TASK-38529](../../../backlog/tasks/task-38529%20-%20Share-the-existing-fleet-child-launch-path-without-behavior-drift.md) | 1, 2, 5 | Shared native child-launch seam with parity evidence |
| 7 | [TASK-38530](../../../backlog/tasks/task-38530%20-%20Checkpoint-workflow-evidence-and-recover-explicit-resume.md) | 1–3 | Exact checkpoint/freshness and conservative restart service |
| 8 | [TASK-38531](../../../backlog/tasks/task-38531%20-%20Schedule-approved-phases-through-the-existing-Console-runtime.md) | 1–7 | Complete headless native phase path, disabled by default |
| 9 | [TASK-38532](../../../backlog/tasks/task-38532%20-%20Expose-workflow-setup-and-phase-controls-in-Console.md) | 8 | Mounted native controls and navigation |
| 10 | [TASK-38533](../../../backlog/tasks/task-38533%20-%20Qualify-shared-worktree-workflows-across-execution-and-restart.md) | 8–9 | Retained A1–A12 and real-provider qualification |

Each task ends with targeted checks, diff/self-review, concise implementation notes, AC updates and a scoped commit. Do not mark a task Done on a planned test or unexecuted assertion. The default order is sequential; task independence is not authorization to launch subagents during planning.

<a id="task-1"></a>
## Task 1: Bounded models, durable state and private results

**Task:** TASK-38524. **Coverage:** spec 2–3, 7–8; A9/A11/A12 foundations.

**Files:** Create `Agents/workflow_models.py`, `DB/workflow_runs.py`, `DB/recovery_workflow_schema.py`, `Tests/Agents/workflow_support.py`, `Tests/Agents/test_workflow_models.py`, `Tests/DB/test_workflow_runs.py`, `Tests/DB/test_workflow_runs_migration.py`, `Tests/Backup_Recovery/test_workflow_recovery_schema.py`. Modify `DB/AgentRuns_DB.py`, `DB/recovery_core_schema.py`, `DB/recovery_core.py`, relevant existing `DB/recovery_operations.py` paths, and the next actual `DB/migrations/agent_runs_*.sql`. All product paths here are under `tldw_chatbook/`.

**Interfaces produced:** strict `WorkflowRequest`, `WorkflowPlan`, `WorkflowTaskSpec`, `WorkflowDecision`, `WorkflowTaskResult`, `WorkflowSnapshot`, `WorkflowTaskAttempt`, `WorkflowCheck` and `WorkflowPayloadReservation` values. `WorkflowRequest` contains `conversation_id`, `objective`, `binding`, `targets` and `policy`; reuse compatible `GoalBindingRef`, `GoalProviderRef`, and `AutomaticWorkLimits` as structural values, not Goal execution authority. `WorkflowSnapshot` includes `id`, `chain_id`, immutable `request`, `control_revision`, separate `progress_cursor`, `status`, saved revisions/results and accounting. `WorkflowRunsStore(db)` exposes `create(request, *, launch_id) -> WorkflowSnapshot`, `get(workflow_id) -> WorkflowSnapshot`, bounded `list_for_conversation(conversation_id, *, limit, offset)`, result capacity reservation/read/removal and revision CAS. These create no execution capability.

- [x] **1. Add real failing boundary and SQLite tests.** Reject duplicate JSON keys, boolean/nonfinite limits, more than six phases/twelve tasks/64 scopes, invalid earlier dependencies and multibyte overflow. Prove duplicate launch identity and all-or-nothing chain creation. Define the following small shared test factory; it grants no live authority:

```python
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
from tldw_chatbook.Agents.goal_models import GoalBindingRef, GoalProviderRef
from tldw_chatbook.Agents.workflow_models import WorkflowRequest

def make_workflow_request(root):
    return WorkflowRequest(
        conversation_id="workflow-test-chat", objective="Correct the selected source",
        binding=GoalBindingRef(workspace_id="w", binding_id="b",
            locator=str(root.resolve()), access="rw"),
        targets=(GoalProviderRef(provider="openai", model="test-model",
            config_ref="api_settings.openai", authority_ref="unauthenticated",
            endpoint_ref="provider-default"),),
        policy=AutomaticWorkLimits(generations=8, child_launches=12),
    )
```

```python
def test_duplicate_start_preserves_identity_and_no_dispatch(tmp_path):
    from contextlib import closing
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.workflow_runs import WorkflowRunsStore
    from Tests.Agents.workflow_support import make_workflow_request
    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        store = WorkflowRunsStore(db)
        first = store.create(make_workflow_request(tmp_path), launch_id="start-1")
        again = store.create(make_workflow_request(tmp_path), launch_id="start-1")
        assert (first.id, first.chain_id) == (again.id, again.chain_id)
        assert first.status == "awaiting_plan"
```

- [x] **2. Run the two new focused modules red.** Expected failure is absent workflow behavior, not bootstrap, an invalid fixture or unrelated source import. Run `.venv/bin/python -m pytest -q Tests/Agents/test_workflow_models.py Tests/DB/test_workflow_runs.py`.
- [x] **3. Implement storage behind the existing transaction owner.** Use immutable plan rows, separate control/progress revisions, unique workflow/phase/task identities and parameterized SQL. Create the chain and workflow structural rows in one existing connection-taking FULL transaction; never nest independent commits. Reserve required payload storage before dispatch, refuse overflow without truncating success, and retain tombstones after settled payload removal. Check exact run/workflow/attempt ownership before result publication. Missing required payload is a failure, not empty success.
- [x] **4. Migrate and qualify recovery in this slice.** Choose the next actual version; preserve every shipped migration and existing catalog. Capture exact installed SQL via the current recovery catalog pattern; add workflow tables/indices/triggers, row validation, staging and inert-import handling. Cover actual old-schema data, upgrade, reopen, duplicate migration, malformed/cross-owned rows, active evidence retention, capacity refusal, and complete backup/rebackup. Required payload storage stays within the existing admitted private store; do not introduce another root.
- [x] **5. Run new tests plus** `Tests/DB/test_automatic_work_migration.py`, `Tests/DB/test_goal_runs_migration.py`, `Tests/Backup_Recovery/test_agent_runs_recovery_schema.py`, `Tests/Backup_Recovery/test_goal_recovery_schema.py`. Expected: all named checks execute the real SQLite/catalog paths and pass.
- [x] **6. Review and commit only this slice.** Confirm ordinary DB opens create no new runtime or global recovery sweep. Record exact chosen migration/catalog changes, tested archive paths and payload limits; commit message `feat: persist bounded native workflow state`.

<a id="task-2"></a>
## Task 2: Workflow admission, accounting, active intervals and Settings

**Task:** TASK-38525. **Coverage:** spec 5–6/8; A5/A7/A11.

**Files:** Modify `Agents/workflow_models.py`, `Agents/automatic_work_budget.py`, `Agents/automatic_work_runtime.py`, `DB/automatic_work.py`, `Chat/console_chat_models.py`, `Chat/console_automatic_submission.py`, `DB/recovery_workflow_schema.py`, `DB/recovery_core_schema.py`, `DB/recovery_core.py`, `config.py`, `UI/Screens/settings_screen.py`; create the phase-authorization portion of `Chat/console_workflow_runs.py`, `UI/Screens/settings_workflow_policy.py`, `Tests/DB/test_workflow_attempts.py`, `Tests/DB/test_workflow_active_time.py`, `Tests/Agents/test_workflow_policy.py`, `Tests/UI/test_console_workflow_settings.py`. Extend the exact storage/catalog delta introduced in Task 1 using the next unshipped migration, never editing a shipped one.

**Consumes:** Task 1 state and native automatic reservations. **Produces:** `ConsoleSubmissionOrigin.WORKFLOW_PHASE`; `AutomaticWorkContext.attempt_kind == "workflow_phase"`; immutable-origin `policy_kind == "workflow_phase"`. Add ledger APIs `prepare_workflow_phase(workflow_id, *, revision_id, phase_id, owner_id, session_id, command_id) -> WorkflowPhaseAttempt`, `accept_workflow_phase(attempt_id, *, owner_id) -> bool`, and `claim_workflow_task(attempt_id, *, task_id, owner_id) -> WorkflowTaskAttempt`. The attempt values contain exact workflow/revision/phase/chain/session/owner/state and stable identities. `WorkflowPhaseAuthorization` is private process-local custody issued only after native owner validation and FULL acceptance; an accepted database ID alone is not its constructor.

- [x] **1. Add failing duplicate-acceptance, wrong-owner and active-time tests** using real SQLite and injected wall/monotonic clocks, following `Tests/DB/test_automatic_work_deadlines.py`. Check identical decisions return the original attempt, conflicting payloads refuse, revocation wins before effect dispatch, and prepared/uncertain work never grants replay.
- [x] **2. Run red:** `.venv/bin/python -m pytest -q Tests/DB/test_workflow_attempts.py Tests/DB/test_workflow_active_time.py Tests/Agents/test_workflow_policy.py`. Inspect failure cause before implementation.
- [x] **3. Extend every automatic-kind reader/acceptance/recovery branch together.** Audit the controller's origin-specific handling, `_read_attempt`, `policy_kind`, enabled-setting selection, SQL attempt-kind constraints, active-per-conversation index, admission guards and restore activation. Reuse physical attempt storage without inventing child completion claims. Workflow calls/children charge their one workflow allowance; manual requests stay manual. Example required policy assertion:

```python
def test_workflow_policy_does_not_inherit_goal_or_wake_settings(monkeypatch):
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
    monkeypatch.setattr(run_log, "_setting", lambda key, default: default)
    limits = AutomaticWorkLimits.from_settings("workflow_phase")
    assert limits.generations == 8
    assert limits.child_launches == 12
    assert limits.wall_seconds == 900
    assert AutomaticWorkLimits.from_settings("goal_iteration").child_launches == 0
```

- [x] **4. Implement workflow-only union-of-active-interval accounting.** Open before attributable dispatch, retain charging through alive/unknown cleanup, and close only after positive drain and durable settlement. Parallel task durations overlap rather than add. Fully drained persisted review/capacity waits exclude time; an unproven permission wait does not. Persist charged time and clock owner; foreign open intervals cannot become zero after restart. No workflow deadline reset through approvals or limits raised later. Keep existing Goal/wake absolute deadlines byte/behavior compatible.
- [x] **5. Fold canonical Settings and config into this slice.** Follow `settings_goal_policy.py` draft/Save/Revert/validation patterns using the exact workflow keys/defaults above, labels that distinguish active seconds, and zero-disable behavior. Do not add the retired settings surface. Run new tests plus `Tests/DB/test_automatic_work_budget.py`, `Tests/DB/test_automatic_work_deadlines.py`, `Tests/Agents/test_goal_models.py`, `Tests/UI/test_console_goal_settings.py` and the new workflow Settings tests.
- [x] **6. Review and commit.** Include overnight review, two overlapping clocks, live reduction, exhaustion, clock reversal and interrupted interval receipts. Commit `feat: admit workflow phases with retained active-time budgets`.

<a id="task-3"></a>
## Task 3: Complete runtime worktree activity claims

**Task:** TASK-38526. **Coverage:** spec 4; A3/A4/A6/A8.

**Files:** Create `Workspaces/worktree_activity.py`, `Tests/Workspaces/test_worktree_activity.py`; modify `Chat/console_runtime.py` for lazy construction and `Agents/execution_capacity.py` only where existing drain integration requires it. Reuse existing path/root identity validation, not a new normalization algorithm.

**Consumes:** captured admitted roots and `ExecutionOwner`. **Produces:** immutable `WorktreeFootprint(reads, writes, opaque)` with exact/subtree scope entries and a `WorktreeActivity` owner. `try_acquire(*, owner: ExecutionOwner, footprint: WorktreeFootprint) -> WorktreeActivityClaim | None` acquires a complete claim without waiting. `claim.settle(*, durable: bool)` records conclusive journal settlement; release requires that plus `on_drained(True)`. No public force-release. `WorktreeFootprint.opaque` is runtime-wide; ordinary unclassified root work is whole-root exclusive. Scope paths have `path: Path` and `subtree: bool`; inputs must come from the existing admitted resolver.

- [x] **1. Add real-path/barrier tests** for read/read sharing, write/read conflicts, directory create/list conflicts, alias/nested-root overlap, complete multi-root admission, refusal unwind and opaque effects. Use actual ExecutionOwner operations and stable real root identity controls.
- [x] **2. Run red:** `.venv/bin/python -m pytest -q Tests/Workspaces/test_worktree_activity.py`.
- [x] **3. Implement only reservation logic.** All-or-none under one short owner lock; no I/O, callbacks or model calls in that lock. Take a runtime-wide opaque claim before any per-root claims, never upgrade while holding a partial claim. Acquire footprint before nonblocking native capacity; capacity refusal unwinds the unlaunched footprint. Use absolute canonical target relationships with captured identities to reconcile alias/case/subtree semantics. This module never launches work or charges a budget.
- [x] **4. Bind retirement exactly:**

```python
# Registration pattern inside the activity owner; the callback receives proof.
owner.on_drained(lambda proven: activity.record_drain(claim.id, proven=proven))
```

`record_drain(claim_id, *, proven: bool)` is a private idempotent method of `WorktreeActivity`; `settle(durable=True)` and a positive drain must both exist before removing a claim. A negative outcome, terminal row or Stop preserves the claim. `claim.attach_child(owner: ExecutionOwner)` retains a normal turn tree's existing reservation for a genuinely admitted native child before thread start; it does not reacquire a conflicting footprint. Release waits for every attached owner's positive drain and required durable settlement. Only trusted launch custody can attach a child, never a model-supplied ID. Callbacks run outside the capacity lock. Add the regression with a timed-out tool physically held after its caller returned and a second attempted claim refused until release.
- [x] **5. Run new tests plus** `Tests/Agents/test_execution_capacity.py`, `Tests/Chat/test_fleet_execution_ownership.py`, `Tests/Workspaces/test_change_review_finalization.py`. Check actual native fixture file existence before selecting each regression; changed names are reconciled in task notes rather than replaced by weaker tests.
- [x] **6. Review and commit** `feat: coordinate runtime worktree activity through physical drain`. Actual ordinary-turn/phase admission wiring is Task 8; this slice creates no global singleton or second scheduler.

<a id="task-4"></a>
## Task 4: Atomic observed-content checks for edit and patch

**Task:** TASK-38527. **Coverage:** spec 4; A2/A3/A10.

**Files:** Modify `Tools/local_tool_impls.py`, `Tools/patch_tool_impls.py`, `Tools/workspace_tool_worker.py`, `Tools/workspace_tool_protocol.py` only if the existing closed request schema needs a precondition field, and `Agents/local_tool_provider.py`; extend `Tests/Tools/test_patch_tool_impls.py`, `Tests/Agents/test_local_tool_provider.py`; create `Tests/Tools/test_edit_patch_preconditions.py`. Reuse existing local/remote worker dispatch and read-ledger parsing.

**Consumes:** existing `_atomic_write_target`, digest/absence validation, workspace workers. **Produces:** edit/patch mutation carries the digest of its actual read bytes into the existing atomic writer; workflow dispatch can require a non-blind precondition. No second file-writing implementation and no promise of whole-patch transactionality.

- [x] **1. Add a held-read race** that replaces bytes between edit/patch read and mutation, then prove the peer's bytes survive. Also test absent→created, disappearing targets, ledger eviction, same-agent repeated writes and multi-target partial effect reporting. Do not rely only on a sequential provider stale check.
- [x] **2. Run red:** `.venv/bin/python -m pytest -q Tests/Tools/test_edit_patch_preconditions.py`.
- [x] **3. Carry observed bytes through the existing primitive:**

```python
import hashlib

def observed_digest(original_bytes):
    return hashlib.sha256(original_bytes).hexdigest()

# At the existing edit/modify-patch write site, hash the actual read bytes,
# preserve newline bytes, then pass expected_sha256 into _atomic_write_target.
```

The example demonstrates the calculation, not a required new one-line helper. Keep it inline at the existing read/write sites. Preserve path validation, descriptor/no-follow behavior, permission errors, dry-run semantics and lock-contention classification. Workflow `fs_write` requires an actual read/absence precondition even when the bounded ledger evicts an entry; the ordinary non-workflow blind-write contract stays unchanged.
- [x] **4. Exercise local and remote protocol paths.** Any added host-originated precondition is validated by the closed worker schema; never allow model arguments to forge workflow authority. Distinguish pre-dispatch refusal, settled mutation and uncertain/partial effects using existing typed worker receipts. A partial multi-file patch is not automatically retried wholesale.
- [x] **5. Run** the new module, `Tests/Tools/test_patch_tool_impls.py`, relevant stale-write cases in `Tests/Agents/test_local_tool_provider.py`, `Tests/Agents/test_fs_read_ledger.py`, and affected `Tests/Tools/test_remote_executor_loopback.py` cases. Include platform-supported lock races without claiming advisory locks constrain arbitrary external editors.
- [x] **6. Review and commit** `fix: preserve observed bytes through atomic edit and patch`. Check every caller of changed functions and update worker fixtures in the same slice.

<a id="task-5"></a>
## Task 5: Run-bound scope at every native invocation

**Task:** TASK-38528. **Coverage:** spec 3–5; A2/A3/A5.

**Files:** Create `Agents/workflow_task_scope.py`, `Tests/Agents/test_workflow_task_scope.py`; modify `Agents/agent_runtime.py`, `Agents/agent_service.py`, `Agents/tool_catalog.py`, `Agents/local_tool_provider.py` and necessary existing builtin/raw-shell/virtual/skill dispatch seams. No new tool provider with duplicate filesystem behavior.

**Consumes:** accepted phase custody from Task 2, task definitions from Task 1, footprints from Task 3 and required atomic preconditions from Task 4. **Produces:** private `WorkflowTaskAuthorization` containing its exact phase/task attempt, native run, immutable mode/tools/scopes and live authority guard. `check_task_call(authorization, *, registry, call: ToolCall) -> None` raises a stable refusal on mismatch; its host-owned issuer requires a live accepted phase authorization and claimed task. IDs in tool arguments never select authority. Final invocation uses the same bound restriction after waits.

- [x] **1. Add negative tests** covering wrong/outside/nested alias, readonly/revoked binding, shared-provider sibling identity, alternate builtin `write_file`, a previously approved-session shell, progressive discovery, skill spawning, child/new-chat callbacks and unknown-effect MCP. Test the actual dispatch entry, not just the pure scope predicate.
- [x] **2. Run red:** `.venv/bin/python -m pytest -q Tests/Agents/test_workflow_task_scope.py`.
- [x] **3. Wire narrowing around existing owners:**

```python
# Existing guard/dispatch closures retain actual registry owner resolution.
check_task_call(authorization, registry=registry, call=call)
# Only then proceed through the unchanged native permission owner.
```

Filter advertised tools by captured mode, but enforce the same boundary at runtime callbacks and actual invocation. Invoke-bound checks use the actual current run/task, captured native binding and registry owner; `PathAwareToolProvider.path_targets` is not a new permission grant. A sibling sharing LocalToolProvider must not see or overwrite another task's restriction. Do not replace configured hooks; compose restriction-only behavior with their current ordering and fail-closed rules.
- [x] **4. Check opaque mode separately.** It requires the Task 3 runtime-wide reservation and existing exact tool permission, shell arming or script trust. Serialization never converts shell/remote authority into confinement. Additional scope/dependency needs stop for replanning; neither Allow nor task prose widens the footprint. Required input/handoff paths must resolve to admitted sources.
- [x] **5. Run new tests plus** `Tests/Agents/test_tool_catalog.py`, `Tests/Agents/test_tool_catalog_owner_cache.py`, `Tests/Agents/test_local_tool_provider.py` affected cases, `Tests/Agents/test_fleet_runtime.py` runtime-tool gates and `Tests/Chat/test_response_rules_goal_intersection.py` to preserve existing hook/Goal separation. Add typed private-source and negative forged-ID tests at every new callback.
- [x] **6. Review and commit** `feat: enforce native workflow task scope at dispatch`. Inventory each supported effect route and its negative test in task notes; unsupported opaque operations refuse rather than silently falling back.

<a id="task-6"></a>
## Task 6: Share native child launch without changing ordinary fleet behavior

**Task:** TASK-38529. **Coverage:** spec 2/5; A1/A5/A6.

**Files:** Modify `Agents/agent_service.py`, `Agents/fleet_coordinator.py` only where needed for typed durable membership; create `Tests/Agents/test_workflow_child_launch.py`; extend `Tests/Agents/test_fleet_runtime.py`, `Tests/Agents/test_fleet_continuation.py`, and affected routing/hook/client tests. Keep execution helpers in the existing service unless a cohesive child-launch implementation warrants a single adjacent file; no generic worker framework.

**Consumes:** native spawn/continuation, Task 2 phase/task claims and Task 5 authorization. **Produces:** a shared private prepared-child launch path and, for Task 8 integration, `AgentService.launch_workflow_task(task: WorkflowTaskSpec, *, authorization: WorkflowTaskAuthorization, should_cancel: Callable[[], bool]) -> WorkflowChildReceipt`. `WorkflowChildReceipt` contains `attempt_id`, `run_id`, `handle_id` and its `ExecutionOwner`; it grants no new authority. The runtime-only caller must supply Task 5 custody; that module, including its type, must be present before positive workflow dispatch is enabled. Ordinary spawning still enters the same shared private launcher with ordinary native custody. A persisted identifier cannot call this seam directly.

- [x] **1. Characterize actual `_launch_fleet_child`, `spawn`, continuation and finalization callers** before moving code. Add assertions for one capacity reservation, one persisted run, owner/run binding, exactly one hook lifecycle, frozen endpoint/fallback refs, current per-run approval stamps, provider closure and result settlement. Cover a child surviving its parent, cancelled/uncertain workers and failed thread start.
- [x] **2. Run the existing cases before refactoring:** `.venv/bin/python -m pytest -q Tests/Agents/test_fleet_runtime.py Tests/Agents/test_fleet_continuation.py Tests/Chat/test_fleet_client_lifecycle.py`. Retain actual bootstrap failures separately; do not use the refactor to hide them.
- [x] **3. Extract the existing composition/launch tail with explicit captured inputs.** Preserve per-turn roster, instruction activation, task budget, native origin, model client and cancellation closures. Never retain a closure into a later phase's mutable turn state. Ordinary inline/skill paths and `max_live_subagents <= 1` behavior remain unchanged. Host-planned shared-worktree tasks use the same authority and capacity owners and cannot select isolation as a side effect.

```python
def test_untrusted_host_launch_never_creates_a_child(service):
    import pytest
    # `service` is the existing real AgentService fixture used by this module.
    with pytest.raises(PermissionError):
        service.launch_workflow_task(
            None, authorization=None, should_cancel=lambda: False
        )
```

Define the `service` fixture in the new test module with the same real temporary AgentRunsDB/registry construction used in `Tests/Agents/test_fleet_runtime.py`; record its original row and capacity counts and assert both remain unchanged. The deliberately invalid arguments must fail authorization before task construction or child creation.
- [x] **4. Preserve failure unwind.** Proven pre-dispatch refusal refunds only unconsumed reservations. An already claimed launch with unknown thread/provider/effect state remains inspectable and unavailable for replay. Root completion cannot release held model/tool operations. Do not change automatic survivor delivery yet; Task 8 adds typed membership handling with end-to-end tests.
- [x] **5. Run parity plus** `Tests/Agents/test_agent_routing_integration.py`, `Tests/Agents/test_fleet_stop_semantics.py`, `Tests/Chat/test_individual_fleet_settlement.py` and the new negative-launch module. Add positive approved-task tests when Task 5 custody is available; Task 8 must exercise the same seam with the actual coordinator.
- [x] **6. Review and commit** `refactor: share native fleet child launch custody`. Report moved behavior and measured regression results rather than claiming a new executor was built.

<a id="task-7"></a>
## Task 7: Required evidence, checkpoint settlement and explicit recovery

**Task:** TASK-38530. **Coverage:** spec 7–8/11; A8/A9/A10/A11.

**Files:** Create `Agents/workflow_run_service.py`, `Tests/Agents/test_workflow_run_service.py`, `Tests/Chat/test_workflow_recovery.py`, `Tests/DB/test_workflow_evidence_retention.py`, `Tests/DB/workflow_restart_child.py`; extend Task 1 store/recovery files and existing `Agents/goal_iteration.py` only for a genuinely shared bounded evidence primitive. Modify `Chat/console_runtime.py` startup-audit wiring, not a second global sweep.

**Consumes:** Task 1 durable state, Task 2 claims/active time, Task 3 physical claims and existing persistence/manifest/recovery services. **Produces:** `WorkflowRunService(db, persistence)` with `call(operation, *args, **kwargs)` finite DB offload; `get(workflow_id) -> WorkflowSnapshot`; `checkpoint_task(result: WorkflowTaskResult, *, drained: bool) -> WorkflowSnapshot`; `prepare_decision(workflow_id, *, phase_id) -> WorkflowDecision`; `resume(workflow_id, *, expected_control_revision, command_id) -> WorkflowSnapshot`; `project_recovery(audit)`. `WorkflowDecision` captures `workflow_id`, `revision_id`, `phase_id`, `control_revision`, `predecessor_digest` and `input_manifest`. Quality-review and execution-recovery decisions have separate tagged types; one cannot clear the other.

- [x] **1. Add failing real-file/SQLite tests:** exact output/verifier/argument digests, stale inputs after review opens, missing/oversize mandatory data, wrong native run ownership, same-result idempotency and changed-result conflict. Turn native logs off or force truncation; reopen a fresh DB/process and require complete authoritative handoffs.
- [x] **2. Run red:** `.venv/bin/python -m pytest -q Tests/Agents/test_workflow_run_service.py Tests/DB/test_workflow_evidence_retention.py Tests/Chat/test_workflow_recovery.py`.
- [x] **3. Adapt existing Goal evidence and lifetime patterns.** Use current bounded `capture_manifest`/freshness checks and typed process observations; do not call Goal execution, change its no-child policy or create another authority grant. Reserve/copy required results within Task 1 limits and retain active/uncertain references against pruning. Reuse `run_owned_db_call`/`operation_owned_connection` custody for the complete finite callback, including preflight work; never abandon thread-owned SQLite handles.

```python
def test_model_done_is_not_a_passed_check():
    from tldw_chatbook.Agents.workflow_models import WorkflowCheck
    check = WorkflowCheck(
        verifier_id="unit-tests", outcome="missing", exit_code=None,
        input_manifest=None, verifier_sha256=None,
    )
    assert check.outcome != "passed"
```

`WorkflowCheck`, introduced with Task 1 models, records `verifier_id`, `outcome` (`passed/failed/skipped/stale/missing`), optional typed exit code, input manifest and verifier digest. Real service tests must demonstrate how model-reported done plus missing process evidence prevents phase completion; the value assertion above is only the minimum schema example.
- [x] **4. Settle result, checkpoint/evidence, task status and its accounting in one FULL transaction after positive physical retirement.** Reuse existing Change Review window references with consent/exclusion/attribution disclosure; do not manufacture per-task diffs or force-enable capture. Stale evidence remains history but cannot authorize a dependent phase. Failure after a known effect retains uncertainty, charges and claim protection.
- [x] **5. Recover only through the existing startup owner.** Feed its one audit into workflow projection. Test fresh processes at prepared, accepted, thread-start, effect and result-commit boundaries. Clean waiting becomes Resume required, unknown effects Recovery required. Revalidate captured binding, provider/tools, results, input manifests and remaining allowance; fresh Resume custody never replays uncertain work or resets counters. A recovered/imported row is inert.
- [x] **6. Run new cases plus** `Tests/DB/test_goal_evidence_retention.py`, `Tests/DB/test_goal_restart_process.py`, `Tests/Chat/test_goal_storage_lifetime.py`, `Tests/Chat/test_console_goal_recovery.py` and relevant exact-schema backup cases. Review private payload removal/tombstones and commit `feat: checkpoint workflow evidence and conservative resume`.

<a id="task-8"></a>
## Task 8: Native phase scheduling and runtime-wide admission

Complete internal partition A (ordinary source composition): capture original
agent storage selection in the existing issuing-submit scope before SessionStart;
retain a private carrier through actual native worker enqueue and reuse the exact
selected run-log root at the real bridge writer. Qualify the stock writer and
bound descendant, preserving strict source equality through actual child MODEL.
First reproduce with real ConsoleRuntime/bridge/AgentService/SQLite/MCP; then
cover original permission revocation, unrelated edits, replaced DB/writer/root,
late sources and closed/negative-drain context. Pin tested bytes, run one named
affected selection and scoped static/preservation checks, then exact-path commit.
ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md
and backlog/decisions/163-expanded-console-hook-runtime.md. Reason: direct precise
source qualification. Earlier references/initializer admission, the full ordinary
owner and borrowers, child writes, SSH and late events remain required next.

D2C finite explicit-reference producer/consumer: freeze the exact existing explicit registry/workspace/run-scope/sandbox roots, cwd and exclusions before expansion. Missing sandbox adds no root and is never created; existing authorized folders remain. Use complete READ coverage for stock file/folder callbacks and initially OPAQUE for reachable Git or unknown callbacks. Own the existing run_preparation_read worker and original controller assignment/discard through operation.finish RETURN, original source/storage and runtime close. Extract the gated Git qualification/managed-tree transport over explicit executable/repository prefix; validate captured cwd and actual repo top-level within the same maximum before output, rebase existing exclusion pathspecs at the repo root, retain unknown cleanup honestly. Add real RED/GREEN worker/process/source/host/cancellation/route tests, targeted affected tests, final-byte/static/protection and exact-path commit. No model/child/workflow row, new executor or unused future borrow API; later ordinary integration must consume its original covering claim before enqueue. ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md; backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md. Reason: bounded completion of the existing source/physical/host contract. Costs: metadata checks, Git qualification/containment overhead, coarse Git contention and potentially indefinite original-read/storage/disposal retention on unavailable proof. Parent remains In Progress/all AC unchecked.

D2 Git custody (approved source contract): keep all three native Git launches under the same initially OPAQUE ticket on the existing worker/publisher thread, adapting the existing low-level ProcessTreeController with a local asyncio Runner. Join real streams, containment-empty/close, original worker and host return, then operation.finish RETURN before root/claim retirement. Bounded cleanup failure retains original scopes/operation for the same controller late wait; genuinely unavailable proof retains cleanup-unproven original worker/storage indefinitely, including possible indefinite disposal. Gate only initial native attachments using native_workflow_qualified; preserve ordinary unsupported-platform behavior, bool stdin callers, qualified source checks, Git helper policy, privacy/history and timeouts. Genuine wrapper initializer/snapshot RED/GREEN and targeted timeout/late-proof/unknown/host/finish-return/parity evidence, final hashes and protected116+C106+c5 preservation. ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md; backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md. Reason: complete existing physical/source/host contract. Parent remains In Progress/all AC unchecked; references, ordinary-tree borrowing and workflow coordinator remain closed.

D2B2b2 (approved source contract): extend original finite event custody to the complete metadata-selected reachable MCP/nested Pre/Post/Failure worker union. Retain exact original context/registry/provider/schema/permission/causal maximum before existing enqueue and capacity close; nested consumers borrow only their original live branch. Same-worker sequential validators share one operation; distinct workers retain separate holds, disabled/impossible/DENY targets none. Join actual worker, bridge finally, plugin required settlement/reaper, original host acceptance/discard and operation.finish RETURN. Never-issued candidates settle only on original conclusive refusal or producer return. Preserve configured disposed-runtime MCP refusal and ordinary standalone/unqualified behavior. Genuine connected MCP/plugin RED/GREEN; positive live-lifecycle closed-capacity nesting separately from actual disposal refusal/drain; one named affected selection; final-byte/static/116+C106+c5 preservation; exact source-only commit. ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md; backlog/decisions/163-expanded-console-hook-runtime.md. Reason: complete the approved finite physical/source/host contract without a new permission grant or framework. Parent In Progress/AC unchecked; outer borrowing/public workflow remain closed.

D2B2b1 (approved complete command-only split): freeze the host-issued complete
potential event maximum before effects; any reachable MCP target preserves the
entire existing event path without a native claim. Covered source loss or maximum
change refuses before enqueue. Reuse finite event/target custody, original command
terminal evidence and plugin settlement; retain actual lifecycle publish, tool
checkpoint/transform and compaction ledger/prepare handoffs. Pre-retain authorized
SessionEnd operations while capacity is open, then seal and join original workers,
host acceptance/discard and operation-release return through repeated cancellation.
Verify actual Runtime/SQLite/commands with held source/publication/cleanup/host seams,
then one named affected selection and final-byte/static/preservation checks before
the exact source/test/docs/task commit. Local evidence stays ignored.
ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md
and backlog/decisions/163-expanded-console-hook-runtime.md.
Reason: qualify the existing finite source/physical/host contract. D2B2b2 must
complete MCP/nested preselection before capacity closure; full ordinary borrowing
and public workflow capture/coordinator remain closed. Parent AC stay unchecked.

**Task:** TASK-38531. **Coverage:** spec 2/4–8/11; A1–A11 native integration.

**Files:** Extend `Chat/console_workflow_runs.py`; modify `Chat/console_runtime.py`, `Chat/console_chat_controller.py`, `Chat/console_agent_bridge.py`, `Chat/console_automatic_submission.py`, `Chat/console_chat_models.py`, `Chat/console_fleet_wake.py`, relevant received-dispatch/lifecycle seams, and existing `Agents/agent_service.py` wiring. Create `Tests/Chat/workflow_rig.py`, `Tests/Chat/test_console_workflow_dispatch.py`, `Tests/Chat/test_console_workflow_scheduling.py`, `Tests/Chat/test_workflow_activity_admission.py`, `Tests/Chat/test_workflow_settlement_delivery.py`.

**Consumes:** all Task 1–7 contracts. **Produces:** `ConsoleWorkflowCoordinator`, lazily owned by ConsoleRuntime, with `start(request: WorkflowRequest, *, launch_id) -> WorkflowSnapshot`; `approve(decision: WorkflowDecision, *, command_id) -> WorkflowSnapshot`; `pause(workflow_id, *, expected_control_revision)`, `stop(workflow_id, *, expected_control_revision)`, `resume(workflow_id, *, expected_control_revision, command_id)`, `revise(workflow_id, *, expected_control_revision, instruction, command_id)` and `subscribe(callback) -> Callable[[], None]`. Controls act on explicit IDs, never the viewed tab by inference. Reuse existing controller submission/native origin, runtime shutdown and progress/settlement events.

Internal dependency 8D2B1 implements complete app-managed legacy event custody:
original source, one opaque event/non-child native owner, per-target operations
before existing enqueue, physical process/group cleanup plus original host
delivery/discard, retained postcommit retry and shutdown drain. It does not
complete Task 8. V2 planned events/MCP/nested inheritance/SessionEnd are mandatory
D2B2 next; outer ordinary-tree and initializer enqueue borrowing remain later.
ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md,
backlog/decisions/148-console-run-hooks.md and
backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md.
Reason: approved finite producer and no-journal settlement qualification.
Task 3's `settle(durable=True)` means all applicable existing required writes
and original host handoffs completed: an ordinary finite event's empty durable
payload obligation set does not invent a successful journal write. Physical
cleanup remains mandatory, and workflow durability/accounting is unchanged.

D2B1 review fix 1 qualifies native hook attachment and actual process-local phase
authority with one metadata-only platform gate before acceptance/effects.
Unsupported platforms retain ordinary granted hooks and passive workflow history;
the workflow flag cannot grant unqualified native execution. Public planning,
Start and Resume must reuse this gate before billable work when those consumers
land. ADR required: no new ADR; ADR285/148 record this compatibility boundary.
Policy simulation tests are not live Windows physical-cleanup evidence.

D2B2a (root-approved pre-engine producer split) owns the actual original plugin
hook preparation before an engine/lifecycle exists: captured preparation source
and selected immutable maximum, existing PluginService admission callbacks,
original NativeHooks acceptance/discard, and cancellation-safe close drain.
Capture/pin first-use native resources without construction I/O; acquire one
opaque finite owner and one operation for this sequential existing worker request
before publication. Bind trusted submission origin explicitly, seal before
capacity closure, and retain original storage/host scopes until actual return.
Preserve PluginRuntimeOwner's independent infrastructure lifetime, current
permissions, qualified-platform policy, and the no-plugin no-effect path.
Verify real PluginService/SQLite with held admission and host publication,
cancellation/replacement/revocation/close, no-target and reserved capacity cases;
then targeted affected consumers, final-byte/static/preservation and exact commit.
ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md
and backlog/decisions/163-expanded-console-hook-runtime.md.
Reason: complete the approved finite preparation producer using existing owners.
D2B2b remains mandatory for planned/individual/notify/lifecycle/command/MCP nested
events, host effects and pre-close SessionEnd. Parent AC and workflow capture,
planning, coordinator entry and ordinary covering-tree borrowing remain open.

- [ ] **1. Build failing integration journeys around real native owners.** In `Tests/Chat/workflow_rig.py`, provide `build_workflow_rig(tmp_path, monkeypatch, provider) -> WorkflowRig`, using existing isolated store/controller/gateway fixture construction and the Task 1 request factory. Its fields are `workflow`, `service`, `coordinator`, `controller`, `store`, `gateway`; methods are `wait_for_status(status, *, timeout=5)`, `close()`, `provider_release()` and `wait_for_provider_entry()`. Native owners must positively drain in cleanup. The test provider addresses roles/task IDs independently so concurrent agents cannot consume each other's scripted replies.
- [ ] **2. Run red:** `.venv/bin/python -m pytest -q Tests/Chat/test_console_workflow_dispatch.py Tests/Chat/test_console_workflow_scheduling.py Tests/Chat/test_workflow_activity_admission.py Tests/Chat/test_workflow_settlement_delivery.py`.
- [ ] **3. Admit read-only planning through typed custody and native per-session acceptance before billable preparation.** Persist the exact chosen existing chat and binding; an unsaved chat uses the current persistence owner before execution, not a second provisioning store. Plan review/approval validates exact control revision, target, inputs and remaining limits; repeated command IDs reuse receipts. Snapshot provider/preset/fallback choices and inherit current project-instruction/permission narrowing for each task.

```python
async def approve_same_decision_twice(rig, decision):
    first = await rig.coordinator.approve(decision, command_id="approve-1")
    again = await rig.coordinator.approve(decision, command_id="approve-1")
    assert first.id == again.id
    assert first.control_revision == again.control_revision
```

This helper is used by the real dispatch test with a held first task; additionally count physical provider entries, durable phase/task attempts and owner reservations. Identical snapshots alone cannot prove no duplicate launch.
- [ ] **4. Schedule only eligible tasks from the accepted phase.** Use actual completion events, full footprint acquisition and the Task 6 native launch seam with Task 5 authority. Missing capacity leaves a task unlaunched and visibly pending; release any unconsumed claims. Hold phase authority/primary slot through settlement, but do not spend lead generations polling. The workflow phase coordinator holds its existing primary admission slot, not a second whole-worktree footprint that would conflict with its own tasks. Persist typed workflow child membership, exclude those children from ordinary fleet-autowake synthesis, and preserve unrelated fleet delivery. Reject recursive delegation and cross-phase source leakage.
- [ ] **5. Wire activity admission for ordinary turns, Goals and survivor children too.** Unknown native root footprints claim whole admitted roots. Opaque-capable work acquires the runtime-wide opaque claim before any per-root claim, even when cwd is scratch. Resolve alias/nested overlap consistently. Acquire all claims without partial waits, before worker start; keep manual capacity reserves and existing explicit prompt-queue semantics. Ordinary parent/child trees share their captured reservation using `attach_child` rather than self-deadlocking on a second exclusive acquisition; survivor ownership keeps it alive after the parent returns. No ordinary run gets new budget or permission rules from this coordination. An active conflicting run receives the existing visible busy/refusal path, not a hidden persisted manual queue.
- [ ] **6. Implement control transitions and finalization.** Status vocabulary: `awaiting_plan`, `planning`, `awaiting_phase_review`, `running`, `pausing`, `paused`, `stopping`, `stopped`, `recovery_required`, `awaiting_result_review`, `completed`, `removed`. Pause stops new task launches and drains admitted work; Stop also revokes/cancels. No callback from an older revision/owner can publish completion, release replacement claims or launch the next phase. A partial phase resumes only against a new explicit validated decision, retaining successful fresh results and old spend. Failed/blocked dependents require a repair plan. Distinguish task execution, objective verification, human quality acceptance and cleanup recovery.
- [ ] **7. Run the new journeys plus** `Tests/Chat/test_fleet_runtime_admission.py`, `Tests/Chat/test_automatic_work_lineage.py`, `Tests/Chat/test_console_goal_dispatch.py`, `Tests/Chat/test_fleet_settle_fanout.py`, `Tests/Chat/test_console_runtime_lifetime.py`, `Tests/Chat/test_console_runtime_shutdown.py`, `Tests/Agents/test_execution_capacity.py`, and affected received/queued/voice dispatch cases selected from actual call-site changes. Retain held-worker Stop-before-dispatch, result-write failure and no-duplicate-wake controls. Commit `feat: schedule approved workflows through native Console owners` only after those pass.

**Internal direct-user RawCLI gate:** Internal D2 direct-user RawCLI (accepted source contract): capture original app/runtime/store/session/invocation before scratch allocation and nonexclusive worker enqueue; admit one manual non-child OPAQUE owner/tool operation. Bind only the real local_command row. Observe exact original partial-create/error and terminal row+canonical step outcomes; no-row refusal needs no fake row, failed/unknown writes retain original custody without replay/poller. Join actual process, admission/relay helper, original host delivery/discard and operation.finish return under original storage/scratch. Qualified model/Task7 calls hand off their same actual provider/runtime/operation; standalone/unqualified stays bounded and reports missing helper proof. Seal before capacity close and retain disposal drain through cancellation. Native Runtime/SQLite/process RED/GREEN, one named affected selection, final-byte/static/116+C106+c5 proof, exact-path commit and independent review. ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md and backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md. Reason: implement accepted source/physical/required-settlement boundary; unknown cleanup or DB outcome may retain original worker/storage indefinitely. Parent In Progress/all AC unchecked; Undo/ordinary covering tree/public workflow remain later. Original source/storage scopes, live original DB handles and scratch lease must extend through actual operation.finish RETURN before their retirement; the held-finish control must include disposed/tombstoned scratch.

**Internal confirmed Undo gate:** Bind the original runtime/feature issuer in the real provider recipe. Finite original OPAQUE preflight retires before actual confirmation; exact one-use confirmed single/multi-root selection admits complete OPAQUE manual nonchild owner/tool custody before original effects. Revalidate original rows/source/root/fingerprints; shared native provider cannot bypass. Preserve historical helper-capable Git using existing transport without D1 qualification. Native mutation attempt requires post-snapshot even on all-path failure; exact original reverted-path-union readback proves DB settlement, never arbitrary HEAD. Unknown required outcomes deliver honest partial results then retain original worker/storage/activity indefinitely. Capture only existing matching scratch generation, release before modal and reacquire that same lease after confirmation; tombstone/loss cannot become unmanaged fallback. Keep same-worker source/DB/shadow/scratch and original host acceptance/discard through operation.finish RETURN; seal starts and drain before storage close, permitting only already-started original obligations after disposal. Real Runtime/SQLite/Git/helper RED/GREEN, one named affected selection, final-byte/static/preservation checks and exact-path commit precede independent review. ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md and backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md; ADR089 card behavior preserved. Reason: accepted source contract implements existing native coordination and required mutating settlement; no new executor, queue, poller, journal or permission source. Parent remains In Progress/all AC unchecked.


**Internal complete ordinary integration:** Capture original selected metadata and callable maximum before reference/initializer/plugin/SSH effects or input consumption; acquire one complete claim/primary owner, consumed at the real bridge row. Wire every actual borrower before queue acceptance, attach genuine child owners before rows/clients/models/threads, observe canonical required writes, and reserve genuine late E/Stop/child/tool obligations before closure using existing required/optional policies. Preserve original sources/storage/host/finish return through disposal and gated A/S. Keep the original launch key; consume only original store committed-first-persistence publication for durable mapping. Actual accepted-turn attachment transfer moves behind asynchronous admission without changing synchronous intake API. Required bundles refuse atomically at small capacity; optional omissions stay observable. Genuine pinned early-effect RED, focused actual routes/maximum/child/borrow/late/SSH positives and negatives, one named affected selection, final-byte/static/preservation, exact-path commit and independent gate. ADR required: no new ADR. ADR paths: backlog/decisions/285-native-phased-agent-workflows.md; 163-expanded-console-hook-runtime.md; 084-change-review-consent-and-asynchronous-finalization.md; 181-ssh-remote-workspace-bindings.md. Reason: complete accepted ordinary ownership without a new capacity/executor boundary. Parent In Progress/all AC unchecked; wave/E/F/G and WHOLE remain mandatory.

<a id="task-9"></a>
## Task 9: Console setup, history, review and controls

**Task:** TASK-38532. **Coverage:** spec 10–11; A8/A10/A12 UI.

**Files:** Create `UI/Console_Modules/workflows.py`, `Widgets/Console/console_workflow_setup.py`, `Widgets/Console/console_workflow_status.py`, `Tests/UI/test_console_workflow_command.py`, `Tests/UI/test_console_workflow_controls.py`, `Tests/UI/test_console_workflow_navigation.py`. Modify `Chat/console_command_grammar.py`, `Chat/console_command_suggestions.py`, `UI/console_command_provider.py`, narrow handlers in `UI/Screens/chat_screen.py`, existing Console compose/status wiring, and source CSS/build manifest only if existing component patterns cannot express required states. Never edit the generated CSS bundle directly; rebuild it if sources change.

**Consumes:** Task 8 coordinator snapshot/subscription and Task 7 service reads. **Produces:** `/workflow new` launch review and `/workflow` this-chat history; setup/status/history modal and a compact native status line. Detail view constructors take explicit coordinator/workflow identity and read no active-tab execution authority. The finite setup uses canonical workflow Settings from Task 2. Histories/task lists are paginated at at most 50 rows.

- [ ] **1. Read** `backlog/docs/design-language.md`, `component-patterns.md`, `lessons-live-verification.md`, current Goal status/setup UI and existing mounted command tests. Add red native command and mounted tests using the Task 8 real rig, existing Textual App/Pilot and `Tests/UI/consolidated_css.py` patterns. Do not turn tests into method calls that skip real button/command paths.
- [ ] **2. Run red:** `.venv/bin/python -m pytest -q Tests/UI/test_console_workflow_command.py Tests/UI/test_console_workflow_controls.py Tests/UI/test_console_workflow_navigation.py`.
- [ ] **3. Wire late-bound commands and projections.** Show the current goal/binding, uncommitted-input baseline, frozen targets, task scopes/dependencies, limits and capture readiness before `Approve plan and start`. Never consume an unrelated draft or implicitly start from history selection. Views subscribe/unsubscribe to bounded snapshots and offload finite history/payload reads via the existing service; no timer or worker owns accepted execution.
- [ ] **4. Implement exactly the approved controls.** At boundaries offer `Approve next phase`, `Revise remaining plan`, `Stop`; during execution `Pause`, `Stop`, existing task inspection/steering; final review `Accept result` or requested repair. Initial-plan approval starts phase one. Resume required and Recovery required are distinct. Quality acceptance cannot resolve execution uncertainty. Default focus is not the approving action. Old controls refuse/refresh against exact revision; duplicate clicks return the existing receipt.

```python
def test_workflow_commands_are_not_dispatched_as_ordinary_prompts():
    from tldw_chatbook.Chat.console_command_grammar import (
        CommandParse, default_console_registry,
    )
    parsed = default_console_registry().parse("/workflow new")
    assert parsed == CommandParse("command", "workflow", "new")
```

The example uses the current `CommandParse` fields verified in `Tests/Chat/test_console_command_grammar.py`. Also test the full composer path through the native command provider and real launch review, preserving the draft/attachments.
- [ ] **5. Exercise mounted layout and lifecycle.** Wait for actual visible/enabled hit geometry, assert each Pilot click returned true, and inspect durable state/charges after decisions. Cover Tab/Enter/Escape, narrow/wide layouts, disabled reasons, pending tool decisions, tab switches, background attention, view detach/remount and reopened saved workflows. Reuse fleet and exact Changes links; display consent off/exclusions/missing checks honestly. Preserve token-backed focus/contrast and existing global keybindings.
- [ ] **6. Run new modules plus** affected existing command grammar/suggestion/composer tests, `Tests/UI/test_console_goal_controls.py`, `Tests/UI/test_console_goal_navigation.py`, `Tests/UI/test_design_token_governance.py`, `Tests/UI/test_component_pattern_governance.py`, `Tests/UI/test_css_bundle_sync_guard.py` if stylesheet integration changed. Update Console user guidance and commit `feat: expose native workflow phase controls in Console`.

<a id="task-10"></a>
## Task 10: Qualification and truthful documentation

**Task:** TASK-38533. **Coverage:** all spec sections and A1–A12.

**Files:** Add `Tests/Chat/test_workflow_native_journeys.py`, `Tests/DB/test_workflow_restart_process.py`, bounded actual-provider evidence under `Docs/superpowers/qa/2026-11-01-native-phased-agent-workflows/`; extend the Task 1–9 tests where the actual owner behavior belongs. Update `Docs/Architecture/console.md`, `Docs/Architecture/agent-runtime.md`, `Docs/Architecture/tool-catalog.md`, `Docs/Architecture/database-layer.md`, `Docs/Architecture/backup-recovery.md`, existing Console user/Settings guidance and task notes with the delivered contract. Add a lesson only if an actual incident yields reusable evidence.

**Consumes:** the real accepted Console path and exact retained outputs, native process results and source/evidence manifests. **Produces:** an A1–A12 receipt matrix stating platform, source revision, actual path exercised, expected/observed behavior and limits. No new benchmark/eval framework or full-suite substitution.

- [ ] **1. Run a deterministic native journey** with a goal that requires a small same-worktree correction, two independent tasks, an ordered dependent check and an independent review. Use actual SQLite and workers; count provider entries and effects. Keep a pre-existing user modification and assert it remains part of input and survives Stop.

```python
def assert_checked_process_result(evidence):
    assert evidence.outcome == "passed"
    assert evidence.exit_code == 0
    assert evidence.input_manifest
    assert evidence.verifier_sha256
```

Apply this to runtime-owned retained evidence, not model text or a fabricated fixture. Preserve a separate failed-command record (`failed`, nonzero actual code), the authorized correction, and a fresh passing record over the resulting exact inputs.
- [ ] **2. Hold the real boundaries.** Exercise duplicate/reordered approvals, old revision decisions, Stop before start, unknown/late worker completion, callback ordering and failing FULL result commits. Verify no second launch, no released claim while a worker lives and no ordinary fleet wake for workflow-owned children. Exercise simultaneous edit/patch races and runtime-wide opaque admission across unrelated chats/aliases.
- [ ] **3. Reopen fresh processes and actual backup candidates.** Cover prepared/accepted/result-commit crashes, read-only/stale bindings, source/verifier/config changes, logs off/truncated, missing/removed payloads, archive rebackup and inert restored history. A clean overnight review waits without active-time spend; unknown active intervals retain conservative charges. Preserve negative Goal/fleet timing controls.
- [ ] **4. Qualify a real configured provider in a private profile/workspace** through actual mounted controls: start, plan approval, phase approvals, failed check, authorized correction, fresh passing check, final review, view detach and restart Resume. Do not use real user databases or request provider secrets in task files. Retain exact commands/outcomes/receipts with sensitive source/credentials excluded. If no suitable provider is available, keep this task In Progress and report the missing qualification; scripted tests are not a live pass.
- [ ] **5. Run only the named affected modules from the earlier tasks and new journey modules.** Ruff lint/format the changed Python/test files using project tools, rebuild/check generated CSS only when affected, validate schema/recovery catalogs, links and scoped diff. Ask before any full sweep. For platform-specific filesystem/lock behavior, retain actual supported-platform results; modelled races are supplementary evidence.
- [ ] **6. Update docs and completion records.** Distinguish present behavior from later research/general-task scope. Keep the feature disabled by default until this qualification is recorded; enabling it remains the user's explicit Settings action. Self-review code, confirm each AC and exact receipts, record no full-suite claim, and commit `test: qualify native shared-worktree workflows and recovery`. Do not push, merge or start unrelated work as part of closeout.

## Coverage and handoff

| Spec acceptance | Implemented/qualified by tasks |
| --- | --- |
| A1 Shared execution and launch compatibility | 3, 6, 8, 10 |
| A2 Invocation scope and file races | 4, 5, 8, 10 |
| A3 Dependencies | 1, 3, 5, 8, 10 |
| A4 Coordination | 3, 8, 10 |
| A5 Approval/dispatch | 1, 2, 5, 6, 8, 10 |
| A6 Actual retirement | 3, 6, 7, 8, 10 |
| A7 Time/accounting | 2, 7, 8, 10 |
| A8 Freshness | 3, 7, 8, 9, 10 |
| A9 Durable results | 1, 7, 8, 10 |
| A10 Review truth | 4, 7, 9, 10 |
| A11 Recovery/storage | 1, 2, 7, 8, 10 |
| A12 Mounted/live flow and bounded reuse | 1, 6, 8, 9, 10 |

Before handing the plan to an implementer, verify all file owners against the
execution checkout, keep new APIs' names/types consistent across tasks, compile
the Python examples as syntax checks, validate all local links/anchors and
Backlog dependencies, scan placeholders, and check task IDs and whitespace.
These documentation checks do not establish product behavior or resolve the
historical bootstrap failure.

Planning ends with this document and To Do tasks. Execution requires the user's
next instruction. Offer either task-by-task inline execution using executing-plans
or, if explicitly chosen, subagent-driven execution with review gates. Preserve
the shared worktree in either case and never start implementation simply because
the plan exists.


Accepted-result lifetime qualification (Task 8D2B2b2): exact original live context
registration remains mandatory for capture, admission, worker start/currentness
and nested borrowing. After actual host return, operation release and positive
native owner drain have conclusively retired the original custody, passive
accepted-context revalidation may use its retained original selection without
requiring the normally spent per-input registration. A retired custody flag alone
is insufficient: the existing owner drain observation must be positive. Provider,
schema/catalog maximum, plugin/permission/runtime/store and activation checks stay
in force. Closed scopes and returned operations cannot dispatch or be borrowed;
no caller flag or replacement context grants authority. This preserves legitimate
later Send reuse of accepted runtime context without reviving launch permission.


I1 permission qualification (Task 8D2B2b2 fix1): passive context retains the actual
MCP provider's original tool, arguments, grant decision, profile and exact service/
permission-store sources. Refresh only that original permission policy before
passive input preparation, outside bookkeeping locks; never reactivate spent
work, dispatch, prompt or consume a grant. The existing resolved-path permission
store fence publishes one shared quiescent token across same-path instances;
policy mutation invalidates it before effects, and uncertain writes remain
unavailable. Fresh successful reads may preserve unrelated permission edits;
deny, kill switch, profile/argument/session-grant narrowing or read/source
uncertainty refuses. Final context publication rechecks only original identities
and the positively observed token. No new permission store/ledger or passive-mode
flag. Cost is bounded original-policy reads per passive preparation and
conservative refusal during concurrent mutation. ADR required: no new ADR;
existing ADR-285/163 source-revocation qualification governs this correction.


Task8D2 A effective project-authority qualification: for the original stock
capture, hook context identity is workspace, enabled state, the full selected
binding snapshot and complete options. Keep every binding/root/locator identity,
write flag and exclusion. Persisted choice/fingerprint and consent-notice
bookkeeping are redundant with that authority and do not invalidate unchanged
accepted context when normal first-folder adoption and independently validated
project consent record them. Custom/unknown readers retain their complete values.
Changed selection, disable and retarget still revoke; activation storage tuple
matching remains strict. This is not a new consent grant or ordinary physical
admission.


Task8D2 partition A review fix1 clarifies the same original-source contract:
MCP completion is published only by the actual guarded coroutine's outer
finally, including source-admission entry and unwind; conclusively unlaunched
refusals may retire after real return, while unknown physical/required settlement
still retains custody. Stock run-log qualification covers the concrete segment
path, append/write, manifest and close callbacks before their receiving calls;
a same-root writer with changed callbacks is not stock. Detailed logs remain
supplementary, including the existing disabled-log and standalone paths.


### Native SSH request custody (Task8D2 prerequisite S)

An actual AgentService model fs_* call retains its original native tool operation
through the selected stock provider/RemoteRoot/adapter/executor, every ping/request,
original cache/result host return and operation.finish return. Exact original
request identity, validated remote terminal cleanup and local request/IO retirement
are separate required facts. The same original worker/storage stays retained on
unknown cleanup; timeout, local SSH death and result text are not proof.

Finite native ssh -G and ssh -O checks reuse ProcessTreeController on the original
worker, preserving argv/configuration/timeout semantics. The existing ControlMaster
manager and session registry own healthy shared infrastructure: a private acceptance
observation binds original invocation/source and actual launcher/IO return to real
publication or verified exact adoption. Registration, path existence or zero exit
alone does not establish acceptance. Unlearned/ambiguous setup and unpublished
session starts require real cleanup or retain native uncertainty. Healthy shared
connections need not exit for a request to finish. No new owner, protocol, poller,
permission, binary restriction or master-lifetime service is introduced.

This qualifies existing operation/source retirement, not early ordinary worktree
coverage. Ordinary covering claims, separately scheduled recovery probes, remote
guidance and complete child/late integration remain mandatory. Task7 currently
accepts local roots only and is not SSH positive evidence. Native platform limits
remain; uncertainty may retain worker/storage/capacity and disposal indefinitely.

Task8D2 ordinary MAX declaration refinement: consume one inert original native effect table shared with the existing LocalToolSpec factory. Do not execute _default_specs, fallback services, handler binding or deep-search settings for early projection. Pin concrete native factory/handler/source identity, apply actual captured gates/profile/persona/Goal selection, and check final real composition containment. Explicit selected output/artifact roots contribute separately; unknown/custom effects stay OPAQUE. Cost: maintain declarations and their source qualification together. ADR285 records the source rationale.


Ordinary MAX Library continuation: original `_library_provider_for_app` selects direct descriptor tools or the RAG fallback from the captured Library authority. Projection must read that original selection and selected app service identities without invoking its factory. Reachable media rechunk with `reindex=true` enters shared RAG/indexing without a closed destination contract and remains OPAQUE. Native note save records Notes organization/publication intents; Sync v2 later copies them to its outbox, while the independent NotesSync observer can derive automatic file updates from note state. No closed original destination/handoff is provided here, so reachable note save remains OPAQUE; this does not classify native read/list the same way. Unknown provider factories/backends/callbacks remain OPAQUE before original construction. Native read/list uses its concrete original class/method/source identity and actual private DB or selected skills-root effect. Final actual provider and callable/source identity must be contained by the original selection; newly reachable services refuse. COST: finite native source records beside the original definitions and original selected service retention; no second catalog, backend inference, NotesSync redesign or new executor.


Ordinary virtual CLI remote consumer (controller accepted): actual VirtualCliProvider per-alias VirtualCliRegistry maps read commands onto the original remote adapter/executor. Retain exact original catalog owner/provider, selected registry/parser/projector/authority/adapter sources before the existing model enqueue; bind the actual selected registry before its execute. Reuse the same original model operation and S request-ID/typed-terminal/local-IO/helper/shared-infrastructure/finish-return proof; no fabricated LocalToolSpec, authority, owner or extra operation. Normal virtual permission/task/alias checks and remote-Git refusal remain. WHY: capture_model_ssh previously selected only fs_* LocalToolProvider despite real virtual remote reads. COST: one finite original virtual source association and receiving equality checks, with unknown cleanup retaining the same original worker/storage. Qualification uses raw source identity without invoking changed descriptors.


Ordinary primary host return (controller accepted): register the exact existing _run_maintenance_agent_call native-host obligation before its original enqueue. Outer submit cancellation/return seals new ingress and requests completion; it does not settle canonical writes while the shielded original native call still runs. Only both original native callback return and outer return permit final canonical readback/primary finish_root, once under either callback order. Existing member/tool/source holds still control physical claim retirement. Cost: one finite native-host receipt and original return callback; no new execution owner, operation, retry, journal or status-based proof. Failed required writes remain retained.


Original ordinary disposal completion (Task8D2): retain only actual admitted _OrdinaryTurn instances in the original runtime, published under its existing execution bookkeeping lock after original scratch admission; a disposal race conclusively retires the unused original admission outside the lock. One private completion observes the existing whole-tree lifetime. Its producer is _retire_scratch AFTER original lease exit and every settlement callback RETURN; the earlier election flag is not completion proof. Runtime disposal waits on these exact completions after original preparation drains and before store/gateway teardown, shielding repeated cancellation beyond grace. Unknown physical return or failed required canonical proof stays pending. Remove a registry entry only after completion. WHY: zero-grace held original operation.finish control showed bounded outer/controller return can precede original tool/child/snapshot/source returns; maintenance-task-only joining is insufficient. COST: one original completion and observer entry per admitted tree, no new owner/operation/executor/queue/capacity API, retry, permission or fresh source authority.


Admission cancellation publication (Task8D2): run_preparation_read delivers cancellation only after the original capture callback physically returns. A genuine owner may therefore already be registered before the awaiting submit observes cancellation. Publish that exact admitted tree into the original submission's finish custody in the await-finally path, including failed return, then let the existing outer finally finish only after original issuer scope unwinds. No unadmitted tree is published; actual source replacement still fails closed. This closes the orphan found by admission-return-red1 without a new owner, replay, return shortcut or early claim release.


MCP detached identity projection (Task8D2): add immutable mcp_definition_sources rows (opaque tool_id, original server_key, raw name, projected LLM name) alongside definition hashes. One private result from the existing original ordered HubTool capture carries both; the public dict-return API remains a wrapper. Use original llm_tool_name/dedupe_names, selected profile/persona and captured source, with no second source read/catalog composition. Native selected-profile DENY filtering already existed; this change closes Goal/final name identity and synchronous fallback selected semantics. Missing/custom structured metadata stays OPAQUE and proves no bounded Goal exclusion. Final native collision wrapper/provider entry must match original fields/hash/alias and captured selected maximum; ambiguous IDs or aliases refuse, never split IDs or remap authority. Goal/config revocation may narrow; changed naming/order requires fresh capture/refusal. COST: detached immutable source rows and one private capture result, no live capability, new owner/operation or alternate catalog.

MCP selection custody (Task8D2): the original received-turn source privately witnesses its exact native capture result and original MCP source, retained in the existing same-task received binding. Ordinary admission may use this witness only with the original receipt, runtime, session and unchanged detached MCP fields; caller-supplied values alone never prove exclusion. Capture failure and unqualified/custom sources remain OPAQUE, while exact native absent or killed sources may be bounded. This is a narrowing observation, not a permission or a live capability in the snapshot; actual provider gates and final identity checks remain authoritative. No second capture, registry or owner is introduced. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: implements the existing private ordinary-custody contract.

Common MCP capture witness (Task8D2): the original qualified native capture publishes one private task-context witness, inherited by its actual submit, for exact controller, session, runtime, MCP source and detached selected metadata. Admission verifies the original source and unchanged MCP definitions/names/profile/persona before using it to prove a bounded exclusion; received intake separately retains its original claim. Unknown or foreign snapshots remain OPAQUE. The witness retains at most one last source identity per live task and no active lease, permission or registry entry. This supersedes the receipt-only association above and preserves no authority in the public snapshot. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: original-source narrowing under the existing ordinary custody contract.

Deferred MCP source selection (Task8D2): custom or unknown manifest/governance callbacks are never invoked for early classification. A detached pending marker plus the one private original capture witness carries their exact source to the same admitted OPAQUE primary; its borrowed operation runs the single original selection through callback, worker and finish return before SessionStart or input transfer. Missing, changed or lost witness visibly refuses, never silently omits tools or reselects. Existing in-memory queued prompt/claim may retain the same original witness through return-to-head; serialization and restoration do not preserve it. Direct, received, queued, Goal, wake and voice use their original custody and source checks; no second capture, owner, registry or live public-snapshot authority is added. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: completes existing ordinary admission and source custody.

Cold deferred MCP permission publication (Task8D2): the original installed local-store binding supplies the prospective canonical permission path before selection; the actual native permission-owner publisher prepares exact candidate, installed binding, method and path proofs before its existing publication lock. At that fence, pure identity/slot checks publish that same candidate to the original service, capture and ordinary source. No callback, source I/O or other owner lock runs under the fence; unproven or replaced sources refuse. The original capture witness consumes only that exact publication proof after worker return. Existing ADR285 and source-custody contracts apply; no new ADR or owner.

Task 8D2 original queued receipt handoff: while the exact received owner is still live, seal only its already-completed selected metadata and original source/publication proof into the existing in-memory queued prompt witness. The same actual entry and coordinator-issued current claim must be checked at dispatch before that metadata can narrow the new ordinary tree; the new tree supplies its own real execution owner. Draft-only edits and return-to-head retain that original witness; changed configuration, drop, consume and restoration do not create or revive it. Source and trust-winner changes refuse, and no old execution capability, second catalog, registry or serialized witness is introduced. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md; reason: direct implementation of original-source custody.

Task 8D2 cold mounted fallback uses the existing original cold-skill preparation as private pending metadata until the actual ordinary custody record acquires its own OPAQUE tree and primary. The same original catalog and conditional trust producer then run once under that tree before input transfer or MODEL, with exact source and physical-return checks. Existing UI Guardian text, captured draft revision/commit, handoff, staged evidence, RAG and queue paths remain their original consumers. The existing in-memory queued witness may retain this pending preparation without claiming an unrun catalog returned. Missing or changed original source refuses; detached snapshots grant no authority. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md; reason: direct implementation of the existing original-custody contract.


**Original skill trust publication handoff (ADR-285 implementation).** A cold original skill preparation records the app/local trust slots at noninvoking capture. Its existing private ordinary witness carries that exact preparation to the original admitted producer. The original pinned synchronous bridge getter or existing asynchronous catalog initializer may publish one latest private local-service metadata receipt only after its real return, original source/factory/path checks, exact initial-slot match and a pure identity publication fence. Queue sealing and exact-claim consumption accept unchanged original slots or that exact native publication proof; an observed or caller-inserted receiver is not proof, and the receipt is neither detached configuration authority nor an execution owner. Existing trust-bearing prepared receipts keep their exact original winner checks. Draft-only edits and return-to-head retain the same private queue carrier; config replacement, drop and restore cannot mint the receipt. WHY: a legitimate first turn may publish the original cold trust service while a separately captured next send waits, but merely adopting a same-type receiver at queue seal is unproven. COST: two original captured slot identities and one latest private source-bound publication receipt; unknown or replaced sources refuse conservatively. The injected-receiver control is a focused 1F/5.08s RED; this record precedes behavior.

The original trust publisher association must survive the before/after race: a task-private scope around the exact original native operation records the candidate at the actual synchronous app getter assignment or asynchronous native build-lock publication, not from an observed slot. Only after the original local getter physically returns, validate the same captured source/factory/path outside bookkeeping locks and publish the one local-service metadata receipt through a pure identity fence. The final app/local winners must be that recorded actual candidate, or the exact already-published app winner captured at original preparation. A concurrent foreign same-type insertion with no recorded original publisher cannot mint the receipt. Cost: a bounded private in-flight candidate marker in addition to the one latest receipt; no producer replay, new store, execution owner or public snapshot authority.


**Original first-save Library provenance handoff (ADR-285 implementation).** Once the original Q1 identity-publication effect physically returns and its existing fingerprinted postcommit ledger marks that exact effect complete, the same controller verifies the exact original live store/session/holder, first-persist commit and binding, and its original Q1 frozen `new_session`/None Library policy against the actual published `durable`/revision-1 snapshot using the existing hook-parent full-value comparison. Only then may it retain one private immutable narrowing receipt on that existing live session; it holds the original commit/source/policy identities, never an execution owner or capability. At a later ordinary consumer, both queue-before-publication and queue-after-publication cases require the exact original common capture witness, actual current queue claim when queued, the unchanged originally captured ceiling, the same live session/store/holder/binding and the exact current published durable policy. Only source/revision provenance can then advance; both policy values still intersect with the frozen maximum, and the original durable transaction rechecks its revision. Detached/restored/no-witness input, unrelated commits, rebinds, changed policy/holder or equality without the original Q1 ledger cannot mint proof. Reroute reuses the same original witness. The receipt is one session-lifetime cell, not a persisted journal/map or general authority. WHY: original Q1 inserts the same captured values but advances their durable provenance; retaining the old source makes a normal queued first-save fail at durable acceptance. COST: one private first-publication receipt and a strict witness/claim/current-policy check. Genuine unsaved first-session RED is `new-session-queue-red1` 1F/7.55s.

Q1 original-result transport refinement: the native store publication retains its existing public `None` return. A private invocation-local carrier receives only the actual original repository read result/session/holder at the exact native holder assignment. After that store method physically RETURNS, the existing controller callback may place this exact value as a provisional session cell. Only after the original fingerprinted `identity_publication` effect has completed may the same cell become verified, with source/commit/session/holder/original snapshot checks repeated. If cancellation lands after ledger completion, an already-completed retry may verify the same retained provisional cell; failed/incomplete effects and a later observed holder pointer cannot authorize. This uses the one session cell, not an additional journal/map, and source checks occur before and after original methods.


### First-send Stop storage-only retirement (Task 8D2; approved before implementation)

WHAT: retain private paired native raw/normalized published configuration observations on the original agent source, and accept only the exact original successful onboarding transaction as the one-flag semantic successor. Bind this proof to the original runtime, transaction, cache identities and exact operation/event; all host and target returns and required writes must be proven before release. Use the original storage/lease only around closed native journal work, with notifications outside that storage context; no ordinary tree, execution owner, inherited waiver, new source, or new effect authority is installed. Source checks for admission, dispatch, borrow, worker and effects stay strict. Preserve the original claim and storage until the actual finish call returns. Unknown callback/source or any semantic/source mismatch remains retained/refused.

WHY: `stop-firstsend-red8` (1F/20.33s) shows the actual first-send onboarding publication increments the generation between physical Stop return and original operation retirement, causing strict source activation to retain the event indefinitely. Same-path equality alone cannot prove semantic continuity, and generic finish callbacks include activity notifications and possible later E admission.

COST: private full in-memory type-sensitive raw and normalized snapshots plus exact original publisher/cache proof; these may contain secrets and are neither logged nor serialized. The single native onboarding mutation must match both normalized flag projections, locked pre-write effective raw values, successful transaction and exact successor/cache identities. Failed writes, unrelated configuration/environment changes, unknown callbacks and new dispatches gain no exception. ADR required: no new ADR; existing ADR-285 supplies the original source/physical custody contract.


Native fresh-bootstrap retirement qualification (before behavior): the exact original native bootstrap can install `_first_run is True` in the raw cache only after its actual config-file creation returns; its locked persisted/effective snapshot omits this transient key. A proven same-path native first-send replacement may therefore compare the original FULL views to expected post-reload views with precisely raw `_first_run` and normalized `COMPREHENSIVE_CONFIG_RAW._first_run` absent. Require exact original bootstrap source/cache/path and creation provenance, True type/value, locked disk absence, and type-sensitive full comparison of every other field; a persisted marker, false marker or any additional change is not this transition. The full original views remain unchanged and new dispatch remains strict. The original app object and all source/permission/path/DB/scratch/lease guards remain exact. Do not require equality of the entire unrelated process environment: full paired raw/normalized publication views and the existing selected/direct source guards govern the original storage-only callback. The observed TIKTOKEN_CACHE_DIR change comes from the native lazy tokenizer and grants no authority. WHY: observations 4–6 show a real freshly created profile loses only these two ephemeral cache projections during native reload; COST: one exact original creation-provenance check and a finite expected-view transition, with no general key exclusion, registry, repin or effect authorization.


Original Stop receipt notification retry (before correction): trace4 shows the exact receipt notification reached the same native event while an earlier reconcile held its latch; the earlier reconcile then refused at original source entry and discarded its pending signal. Permit one same-event retry only for that recorded initial `check_memory` refusal before any source stack/body was entered, when the already-pending notification and current exact semantic receipt and complete original physical/member/required gates all still hold. No retry for activation unwind, body, finish, write uncertainty or other errors, and no new owner/operation/poller. WHY: otherwise the legitimate original receipt cannot wake the existing returned Stop. COST: one private same-call entry-refusal marker and one finite original-event reschedule. Trace5 1P/5.48s used an observer and a too-broad preliminary retry; it is diagnostic only and cannot qualify the narrowed implementation.


Source-entry clarification (before correction): passive trace1 identified the original pre-yield FINAL source check, after original ExitStack entry, as the refused point. Treat the whole original `activate` context entry through its yield as source entry. A same-event pending-receipt retry is eligible only if that exact final pre-yield exception object propagates after the original ExitStack actually and successfully unwinds, with no event body or finish entered; then recheck the current exact receipt and every original source/storage/member/physical/required gate before one retry. An unwind failure/replacement, body entry or any other exception never mints this marker or retry. WHY: the first-check-only marker conservatively misses a real in-flight original source-entry race. COST: an exact private same-call exception identity and positive unwind marker; no effect authority, new owner or generic retry.


Original Stop post-finish fence (before behavior): recheck the same original source, semantic receipt and physical/member/required gates after the actual operation.finish CALL RETURNS and before marking the target released. If a notification changes a pinned source during finish or any check/unwind fails, keep the original source-scope count, scratch lease and claim retained; do not treat finish return as authority to release against a replaced source. Native notification callbacks run without journal DB context and future dispatch remains strict. WHY: capacity notification occurs inside finish and can synchronously change selected state before its return. COST: one source recheck and conservative sticky retention on an unproven return, with no reaper or replay.


Original Stop retirement receiver/caller fence (before correction): before ANY operation pop, callback, DB context, source-scope or claim action, raw-inspect the exact original source instance/class and definition-time native retirement/check/activation/receipt methods including code/default/closure state, refusing instance overrides or replacement helpers. The private finish tuple is evidence only; require the immediate original native `_EventCustody._reconcile_active` frame, exact event/source/target/operation/owner relationship, and original membership/physical/required proofs. Unknown/foreign callers and replaced methods raise without invoking them or changing capacity. WHY: `stop-seam-red1` genuinely called the actual returned Stop operation from a foreign caller after replacing its source checker; current code returned instead of refusing. COST: finite original function records and noninvoking receiver/frame checks at the existing boundary; no registry, new owner or caller-issued authority.


Stop publication producer pin (before correction): the private initial-view capture and original fixed-key publication helper are native producer functions, so record their original function/code/default/closure state and raw-check them before and after the actual call, as well as at retirement consumption. A replaced producer cannot mint a receipt by restoring the consumer later. WHY: the receiving fence alone would not prove which function originally created its metadata. COST: two additional finite original function records and no added source read, capture, executor or authority.


Ruling: WHAT: apply the same noninvoking original receiver/function fence in the pending-notification exception path before calling any retirement method. WHY: an earlier strict source entry can begin before the receipt exists, so the later exception branch cannot assume its initial entry validated a newly available retirement receiver. COST: one repeated pure native identity check on that finite retry; changed receivers refuse before invocation and no new authority is introduced.


### Ruling: first-send publication after original source retirement (FIX1)

WHAT: Keep the app-owned onboarding flag immediately visible after a successful manual parent, but defer its one native persistence until all already accepted ordinary source lifetimes, queued source handoffs, physical tree members and required settlement have actually retired. One runtime-owned pending intent/task and short metadata-only admission/publication fence serialize the final quiescence election with new original source captures; no IO or callbacks run under bookkeeping locks. New captures wait before acquiring a lifetime, or synchronous intake refuses before input transfer. Existing source, permission, DB, root, lease and required-write checks remain strict. Parent response does not await children or late hooks. The original save runs outside locks and disposal waits for its actual return; failed or uncertain proof cannot establish retirement.

WHY: FIX_BASE actual runtime/received survivor RED (1F/20.08s) and native late SubagentStop RED (1F/11.46s) show the onboarding generation change occurs while original admitted physical work still needs its original config identity. Stop-only semantic retirement does not cover those other original consumers. The exact producer/carrier/consumer and cancellation table is recorded in `task-8d2-ordinary-integration-fix1-source.md`.

COST: one private runtime pending intent/task, finite capture/queue/tree bookkeeping and potentially delayed disk persistence; an abrupt exit before the deferred save can show onboarding again. ADR required: no new ADR; this corrects the lifecycle under existing ADR-285. Prior Stop successor text records the earlier approach; it does not grant a general generation waiver or new dispatch authority.

Ruling: WHAT: use the existing one-key `apply_settings_mutation_to_cli_config` transaction with its native locked identity precondition and existing original-function source pins; retain the in-flight metadata fence through actual worker RETURN, including cancellation. WHY: checking config identity before calling the ordinary save alone leaves a race with a concurrent settings transaction. COST: one pure original-source check in the existing config transaction; changed sources refuse and no generation is repinned.

Ruling: WHAT: retain the same original tree source until native activation DB/execution scopes and the original fleet worker's actual source DB cleanup return, checking original thread-local handle identities; uncertain cleanup retains custody, while conclusively unwound body exceptions preserve existing settlement. WHY: the native FIX1 controls exposed live original fleet-thread DB leases after the former tree retirement signal. COST: finite original-scope lifetime and handle proof only; no new owner, lease, reaper, generation waiver or change to non-tree execution.

FIX1 supersedes the earlier Stop semantic-successor implementation: source-closed deferred persistence makes its sole receipt producer unused. Remove paired snapshots, bootstrap receipt metadata and the private Stop finish/journal exception together; keep strict original activation and native config-function pins for the locked one-key mutation. Actual hook/worktree callbacks, storage checks and cleanup semantics remain. Replacement ordering is qualified by actual runtime survivor and held SubagentStop controls; the final native child tail beyond retained DB/execute scope is source-only implicit return/thread bookkeeping, not a universal OS-thread-join claim. ADR required: no new ADR; existing ADR-285.

FIX2 ruling: WHAT — extend the same first-send metadata counter to exact original RawCLI and confirmed-Undo pending membership. Gate native capture before selecting generation, atomically overlap capture/admission with pending count, and release/wake only after original physical/required/host/operation-finish/storage/scratch/root/claim return. Gate only trusted RAW classification through the SAME captured draft commit/start; normal slash/unknown/escaped routes retain their existing availability. WHY — actual unchanged FIX1 native transaction hold admitted a RawCLI call with the old generation (RED1F12.63s, normal original cleanup), and the directly connected Undo source has the same generation-sensitive pending lifetime outside ordinary trees. COST — reuse one existing per-runtime metadata count; no new owner/claim/registry/poller/waiver or source adoption. Lost original membership keeps its count conservatively. Unadmitted Undo confirmation metadata may become stale between user actions and then refuses before effects. Model/Task7 RawCLI keeps its existing single operation/tree. ADR required: no new ADR; existing ADR-285. Source/caller/physical table: task-8d2-ordinary-integration-fix2-source.md.


### Concurrent phase-wave capture (Task8D2; approved before behavior)

Follow the complete [ADR-285 wave contract](../../../backlog/decisions/285-native-phased-agent-workflows.md#concurrent-phase-wave-capture-task8d2-approved-before-behavior)
and ADR-084 amendment. This is one complete internal gate, not a partial interface.

1. Record genuine unchanged-behavior native RED using original Runtime/SQLite,
   current providers/permissions and two independent approved fs tasks.
2. Reuse finite preparation's original AUTOMATIC owner with reserved tool/OPAQUE
   admission before callbacks; qualify exact source/callable maximum and final
   invocation. No fake ordinary tree, capability pruning or new unsupported policy.
3. Implement same-Activity-mutex fence/member ordering and finite snapshot route;
   existing coordinator issues inert reservation, FULL active-window commit
   outside locks, then same-original scheduling. Preserve ready manual priority.
4. Add actual v28 window plus ONE ref table and exact 64-window/64-root/4096-path/
   32768-root-JSON/64-ref/65536-reserved-byte bounds in existing4MiB/128MiB accounting.
   Complete canonical migration/validation, retention/removal and archive/recovery.
5. Launch actual independent children concurrently after B return; bind first
   committed member. Require task physical/required settlement and E disposition;
   close at real gaps/no-commit, reopen fresh B. Keep original source counting and
   shutdown drain through actual native/storage/host/finish return.
6. Cover conclusive physical observation versus required publication failure;
   preserve completed B on E handoff failure (D1 M1), zero/no-run/late E/checkpoint
   immutability, manual races/aliases/OPAQUE and real archive/restore/recovery.
7. Run one named affected selection, scoped static/final-byte preservation and
   exact owned source commit; writer reports to controller for independent full
   SPEC+QUALITY gate. Preserve116/C106/c5 and protected five journeys for G.

ADR required: yes, amend existing decisions.
ADR paths: backlog/decisions/285-native-phased-agent-workflows.md;
backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md.
Reason: approved wave ordering, durable workflow-only window and original source
custody. No duplicate ADR. Task38531 stays In Progress; E/F/G and WHOLE remain.


Task8 D2 close refinement (source-confirmed): workflow preparation must not allocate the unused primary model lifeline. Only `run_primary` starts it, and that entry refuses workflow dispatch; actual children retain their unchanged native child model lifelines. Launched-task native frame teardown returns under a pre-reserved operation on that same original child before its physical drain. A no-launch frame may retire only the source-verified empty display/maps and native provider unbind metadata; any unexpected effectful callback or changed cleanup source refuses, retaining its source lifetime. There is no fresh cleanup owner or OPAQUE claim against the wave, and no generic cleanup exemption.


Task8 D2 hook construction refinement: the unchanged native workflow hook policy refuses configured effectful hooks and any existing legacy engine. `ConsoleAgentBridge._prepare_native_reply` must not invoke the ordinary lazy `ensure_run_hooks` constructor for an accepted workflow preparation after that check: it creates an otherwise unused engine even when unconfigured, invalidating its own preparation. Workflow child callbacks keep the captured absent engine; ordinary hooks and the original workflow refusal checks remain unchanged. This creates no newly supported hook execution.


Task8 D2 existing-empty-hook refinement: an earlier ordinary send creates the original app-owned legacy engine even with no configured handlers. Its mere presence is not an effectful hook. Workflow preparation may recognize only that exact RunHooksEngine and its existing _NativeLegacyHookSource attachment, original runtime/permissions/callback identities, no pending custody, and no published targets, while the saved hook configuration has no enabled handlers. Reuse the issuer capture/check; never fire engine callbacks. Configured handlers, custom/replaced engines or changed sources remain refused. This enables fresh workflow B after an ordinary gap without enabling workflow hooks.


Task8 D2 required-reference count refinement: the v28 window records a bounded snapshot_count alongside the ONE normalized ref table: NULL while required E is pending/unresolved, 0..64 atomically with conclusive E rows/refs, and zero for unused/removed windows. This is the necessary count witness distinguishing a genuine successful zero-row E from missing required references during restore validation; it carries no second mapping or authority. It remains inside the existing 65536-byte reservation and does not change v25–v27 catalogs.


Task8 D2 original native failure/host return refinement: a known failed native create/setup/thread start must finalize its original log and return its original frame under the pre-held close operation when no child thread was started. Unknown creation retains custody. Reserve one additional finite launch-host operation on the SAME child owner before dispatch callbacks, and finish it only after the original launch source/storage/execution scopes return; it prevents an early child or known failure root finish from releasing the original claim before host return. It adds one existing operation per admitted child, not an owner, scheduler or claim. Tool-worker automatic budget/authority checks must enter the existing owned_goal_storage scope before their first database read; the actual child model-loop coroutine must activate its exact original workflow source and owned_goal_storage before gateway checks. The unchanged original lifeline and pre-held child close/join retain protection through these real thread/source/storage returns. Live native archive evidence found AgentRunsDB leases on departed tool/model threads when those checks preceded the finite scopes; no cache reset or cross-thread close substitutes for their original return.


Task8 D2 approved owning-loop callback correction: the existing project-authority wrapper may invoke the SAME synchronous callback directly inside the original native Textual App context only when the captured dispatcher/context are original, current thread equals App._thread_id, and the actual running loop is App._loop. Worker callers continue through original App.call_from_thread; missing/mismatched loop or custom dispatcher/context preserves the existing route/refusal. This corrects the actual ordinary gap send reaching its authority callback on the UI thread without changing permission, source, recovery or activation checks; one positive native identity/thread/loop check, no replacement dispatcher or authority repin.
