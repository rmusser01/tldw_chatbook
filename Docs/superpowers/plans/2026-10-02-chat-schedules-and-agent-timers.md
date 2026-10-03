# Chat schedules and agent timers implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline, slice by slice, with review checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users and native main agents schedule bounded future responses in the same saved Console chat.

**Architecture:** AgentRunsDB owns definitions, grants, occurrences, operation receipts and accounting in one transaction boundary. The common scheduler projects due metadata and hands execution to the app-owned native coordinator. The composer and agent tools call one service, while process ownership, per-chat admission, permissions and physical settlement remain shared runtime boundaries.

**Tech Stack:** Python 3.12+, Textual 8.x, SQLite, existing Pydantic, croniter/zoneinfo and portalocker dependencies.

**Spec:** [Revised scheduling design](../specs/2026-09-09-chat-schedules-and-agent-timers-design.md).

ADR required: yes
ADR path: [Existing ADR-143](../../../backlog/decisions/143-local-chat-schedules-and-agent-timers.md)
Reason: Implements its durable authority, storage, runtime ownership and tool contracts; no duplicate ADR is needed.

## Global constraints

- Use Python >=3.12 and Textual >=8.0.0,<9.
- Execute local native schedules only while Chatbook is running; retain the main-agent scope.
- Do not dispatch server-owned rows, persist scratch locators, infer workspace roots or copy definitions into ScheduledTasksDB.
- Keep user drafts/attachments untouched; use current history and the approved provider/tool/resource capture.
- One shared automatic owner, atomic maintenance exclusion and one native primary per persisted conversation; unrelated manual chats in other app processes remain supported.
- No uncertain-effect replay, early ownership release, grant replenishment or automatic human-root body replacement.
- Apply live profile/persona policy, existing permission gates and deny-only hooks; UserPromptSubmit stays manual-only under ADR-148.
- Use F9 settings, token-backed UI classes, source CSS modules and the CSS builder; preserve terminal/global keybindings.
- Validate public inputs through Pydantic and `Utils/input_validation.py`; use existing `Utils/path_validation.py` authority checks and `DB/sql_validation.py` identifier rules. SQL values are always parameterized.
- The spec's policy table is the single source for numeric defaults, including daily windows, per-occurrence bounds and retained-data quotas.
- Use targeted verification only. A full sweep requires explicit user authorization.
- The first UI has calendar presets; advanced cron and chat Run-now controls remain deferred. The service's defined Run-now semantics still need qualification.

## Integration preflight

The owned native-goal branch predates current dev runtime prerequisites. Do this
before any runtime implementation; do not adapt new scheduling code around the
obsolete scratch/profile/hook wiring in this checkout.

- [ ] Inspect attached worktrees and reuse a suitable isolated checkout. Preserve the shared root's unrelated dirty files and the user-authored `wtf ` line in the original goal plan.
- [ ] Fetch current `dev`, record its exact SHA and integrate it into the owned PR branch without force pushing or bypassing hooks. Review goal/fleet semantic conflicts against both specs; rebuild generated CSS rather than hand-merging it.
- [ ] Re-sweep task and ADR IDs over all refs/worktrees. Reconcile the later web-search `TASK-32195` claimant from add commit `e3ce4d3f4c50a9fe513b42a872e63c2f902526c0`; preserve the earlier scheduling design from `627cb86f77692530c0a83eda955321f2e78c8236`. After both records exist in the integrated checkout, use Backlog CLI for the new record and move every later claimant reference. Never import a duplicate or guess a free ID from this plan.
- [ ] Read the current canonical ADRs by full filename: `079-workspace-assistant-defaults.md`, `082-console-per-chat-private-scratch-space.md`, `148-console-run-hooks.md`, `150-design-token-system-and-design-language.md`, plus ADR-069/131/134/135/141. Read `backlog/docs/design-language.md` and the testing, live-verification, Backlog and Console-wiring lessons.
- [ ] Run targeted existing goal/fleet/runtime ownership checks after integration and record new baseline failures separately from scheduling changes.

Create Backlog execution records through the CLI in dependency order when these
slices begin, with freshly swept IDs, measurable outcomes and a plan before code.
The numbered slices below are implementation units, not reserved Backlog IDs.
Each can be reviewed and committed independently. The existing all-session PR
continues to target `dev`.

## 1. Durable schedule lifecycle and authority

**Files:**

- Create `tldw_chatbook/Scheduling/chat_schedule_models.py`, `tldw_chatbook/DB/chat_schedules.py`, `tldw_chatbook/Scheduling/services/chat_schedule_service.py`.
- Modify `tldw_chatbook/DB/AgentRuns_DB.py` to expose `chat_schedules`, advance its actual current schema version and register guarded migration/audit entries together.
- Add the migration file for the current version transition only after integration; compare the migration manifest and version-table guard before naming it.
- Create `Tests/DB/test_chat_schedules.py`, `Tests/DB/test_chat_schedule_retention.py`, `Tests/Scheduling/test_chat_schedule_service.py` and `Tests/Scheduling/test_chat_schedule_cadence.py`.

**Interfaces:** Define these public types in `chat_schedule_models.py`:

| Type | Fields/meaning |
| --- | --- |
| `ScheduleTiming` | Pydantic discriminated union of delay(seconds), at(run_at), interval(seconds), calendar(cron, timezone); frozen normalized values. |
| `ScheduleCapture` | Provider/model IDs, bounded generation settings, allowed tool IDs, workspace ID, binding ID/fingerprint/access tuples, selected binding ID; no secrets/scratch. |
| `ScheduleMutation` | operation(create/update/pause/resume/retime/run_now/cancel), optional schedule ID/expected revision, instructions/title/timing/max_runs/end_at/follow-up choice. Tool schemas omit human-only capture/delegation fields. |
| `ScheduleOperation` | Runtime-issued source/run/generation/call identity and operation kind; never parsed from model overrides. |
| `ScheduleCallContext` | Trusted conversation/source/owner/run/grant identity, captured authority and exact reviewed-request hash where human task review is required. Resolve through live runtime records. |
| `ScheduleSnapshot` | Definition ID/revision/state, separate grant state, timing/cursor, root/grant IDs, remaining limits and safe provenance. |
| `ScheduleMutationResult` | Immutable snapshot/result, affected and blocked IDs, original committed revision, replay flag or structured refusal code. |
| `ScheduleOccurrence` | Stable ID, nominal due, definition/grant revision, accepted immutable snapshot, covered interval, separate post-acceptance skips, state and ledger references. |

`ChatScheduleService(db, authority_resolver, clock, on_queue_changed, policy)`
produces `mutate(request: ScheduleMutation, *, context: ScheduleCallContext,
operation: ScheduleOperation) -> ScheduleMutationResult`,
`get(schedule_id: str, *, context: ScheduleCallContext) -> ScheduleSnapshot`,
`list(*, context: ScheduleCallContext, cursor: str | None, limit: int) -> SchedulePage`
and `claim_due(schedule_id: str, *, now: datetime) -> ScheduleOccurrence | None`.
Define `SchedulePage(items: tuple[ScheduleSnapshot, ...], next_cursor: str | None)`
in the same models module. The service owns validation and post-commit notifications;
the store exposes transaction-scoped primitives without UI/provider imports.

- [ ] Write failing real-SQLite cases for duplicate claims, all one-time transitions, immutable human root tasks, finite shared descendant grants and mutation replay before CAS. Build a `schedule_case` fixture in the new service test module with a fake aware-UTC clock, actual AgentRunsDB, a live trusted authority resolver and helpers issuing `ScheduleOperation` identities.

```python
def test_committed_update_replay_precedes_revision_check(schedule_case):
    case = schedule_case
    saved = case.create_human("Report usage", {"kind": "delay", "seconds": 300})
    request = ScheduleMutation(
        operation="update", schedule_id=saved.schedule_id,
        expected_revision=saved.revision, instructions="Report disk usage",
    )
    operation, context = case.review_human(request, call_id="edit-1")
    first = case.service.mutate(request, context=context, operation=operation)
    replay = case.service.mutate(request, context=context, operation=operation)
    assert replay.replayed
    assert replay.snapshot.revision == first.snapshot.revision
    assert case.definition_revision(saved.schedule_id) == first.snapshot.revision
```

The fixture's `create_human` builds a validated create request, resolves a real
reviewed context and calls `mutate`; `review_human` issues that exact request's
hash, not a blanket permission bypass. `definition_revision` reads SQLite.

- [ ] Run `python -m pytest Tests/Scheduling/test_chat_schedule_service.py Tests/DB/test_chat_schedules.py -q`; confirm the new behaviors fail before implementation.
- [ ] Implement request normalization with UTF-8/body limits, integer/bool rejection, timezone/DST handling and aware UTC timestamps. Use one transaction for receipt lookup/hash comparison, source authority, new-call CAS, root/grant mutation and receipt. Match the spec's transition table, including retime refusal during accepted one-time work and separate completed-root grant controls.
- [ ] Use unique occurrence keys plus one-active-occurrence constraints. Transfer unaccepted nominal identity on body edits; supersede on timing edits. Record active-slot skips without changing accepted snapshots. Never refund consumed starts or mint descendant grants.
- [ ] Scope early Run-now to a fresh exact-task human start exception for that occurrence. Test that original grant first due time and descendant earliest-start rules do not move.
- [ ] Implement charged-state admission, pinned dependencies, receipt expiry by retired source authority, independent lifetime counters/day watermark and reserved settlement/control capacity. Fresh scoped pause/cancel/revoke remains available at a full quota; create/resume does not.
- [ ] Run all four new files and the existing AgentRunsDB schema/version and automatic ledger tests. Verify real old-file migration, failed-migration rollback and re-open idempotence.
- [ ] Review the slice against spec cases 2/3/7/10/13/15/16/17/21/22, document evidence in its Backlog record and commit only its files.

## 2. Runtime ownership and native scheduled execution

**Files:**

- Create `tldw_chatbook/Chat/console_execution_ownership.py`, `tldw_chatbook/Chat/console_scheduled_turns.py`, `tldw_chatbook/Chat/scheduled_history.py`.
- Modify `Chat/console_runtime.py`, `Chat/console_chat_controller.py`, `Chat/console_fleet_wake.py`, `Chat/console_goal_runs.py`, `Agents/automatic_work_runtime.py`, `Agents/automatic_work_budget.py`, `Agents/agent_service.py`, `Agents/run_context.py`, `DB/automatic_work.py`, `DB/AgentRuns_DB.py` and `app.py` under `tldw_chatbook/`.
- Create `Tests/Chat/test_console_schedule_ownership.py`, `test_console_scheduled_turns.py`, `test_scheduled_history.py` and `schedule_test_runtime.py` under `Tests/Chat/`.

**Interfaces:**

- `ConsoleExecutionOwnership(store_path: Path)` exposes `try_automatic_owner() -> bool`, `try_maintenance(*, exclusive: bool) -> ExecutionLease | None`, `try_primary(conversation_id: str) -> ExecutionLease | None` and `close() -> None`. `ExecutionLease` is a closeable context-managed handle. Shared maintenance precedes primary acquisition; handle ownership remains physical, not a row timestamp.
- `ConsoleScheduledCoordinator(runtime, service, clock)` exposes `offer(occurrence_id: str) -> None`, `notify_capacity() -> None`, `close_admission() -> None`, `shutdown() -> Awaitable[None]`. It restores the saved conversation via app-owned factories, acquires shared maintenance/primary leases, revalidates and accepts through the native controller, then settles through the existing ledger.
- `project_scheduled_notice(message: ConsoleChatMessage) -> dict[str, object] | None` in `scheduled_history.py` returns a labeled user-role provider payload only for versioned scheduled provenance. The controller retains the stored SYSTEM role and pairs it with the occurrence's reply.

- [ ] Write failing subprocess controls using a real temporary file DB and pipe/event synchronization. A holds a physical tool and native primary; B cannot recover/replace it or execute the same conversation, but can manually execute an unrelated conversation. A successful same-process control must prove lock plumbing is active.

```python
def test_secondary_cannot_recover_live_owner(schedule_process_pair):
    pair = schedule_process_pair
    pair.a.start_primary("chat-a", hold_tool=True)
    owner = pair.a.runtime_owner_id()
    assert pair.b.start_automatic_runtime() == "owner_busy"
    assert pair.b.runtime_owner_id() == owner
    assert pair.b.try_manual("chat-a") == "primary_busy"
    assert pair.b.try_manual("chat-b") == "completed"
    assert pair.b.recovery_count() == 0
    pair.a.release_tool()
```

Define `schedule_process_pair` in `schedule_test_runtime.py` using spawned
subprocesses, real runtime/store operations and explicit pipes, not sleeps or fake
owner rows. Each helper above reports actual acceptance/outcome/counts. Kill A in
a separate case and assert unknown work retains charges when B later recovers.

- [ ] Run `python -m pytest Tests/Chat/test_console_schedule_ownership.py -q` and confirm the new boundary fails before implementation.
- [ ] Implement strict portalocker-based automatic-owner, maintenance shared/exclusive and per-conversation locks. Reuse the existing instance-lock mechanics but preserve its separate advisory behavior. Gate schema/blanket recovery, remove constructor/view-open recovery, and perform shared automatic recovery once under exclusive maintenance. Keep unrelated manual chats usable.
- [ ] Add trusted `scheduled` origin and a scheduled policy dispatcher to all admission/accounting routes. Share capacity with goals/fleet/manual turns. Re-resolve current scratch/profile/persona/hooks and bindings before acceptance and after waits; saved selection is a ceiling, not a permission grant.
- [ ] Build the app-owned controller/bridge bootstrap before any screen attachment. Execute the saved task without draft/attachment consumption. Native and non-streaming fallback paths persist deterministic notices/replies and preserve provider request pairs through later history/compaction.
- [ ] Bind the hard approved wall deadline to preparation, approvals and actual workers. Cancel pre-dispatch approvals safely; abandoned/unknown effects retain leases and reservations and block their grant. Do not release on a timeout ToolResult, navigation or a cancelled waiter.
- [ ] Revalidate durable grant controls through active cancellation checks, including a secondary process's pause/cancel. Test A holding a tool, B cancelling its grant and A retaining leases until actual cleanup; an in-memory callback in B alone is insufficient.
- [ ] Run the new runtime/history files plus targeted existing runtime-lifetime, goal scheduling/recovery and fleet safety checks. Test a completed one-time root with live descendants and a timing edit racing settlement.
- [ ] Review against spec cases 4/8/9/11/14/18/19/20/21; commit after checking the actual controller/bridge path and durable transcript, not just service returns.

## 3. Queue projection and nonblocking dispatch

**Files:** Create `Scheduling/services/chat_schedule_projection.py`, `Scheduling/scheduler/handlers/chat_schedule_handler.py` under `tldw_chatbook/`; modify the Scheduling facade/queue/handler registry and `app.py`. Create `Tests/Scheduling/test_chat_schedule_queue.py` and `test_chat_schedule_dispatch.py`.

**Interfaces:** `ChatScheduleProjection(service).tasks() -> list[ScheduledTask]`
returns bounded local metadata without instructions/settings. The registered
`ChatScheduleHandler` uses the existing handler protocol, calls `claim_due`, offers
the occurrence to `ConsoleScheduledCoordinator`, and returns immediately. Pass
the existing `on_queue_changed` callback through service and handler construction;
do not add a second scheduler loop.

- [ ] Write the real-loop test below with a recording native runtime fixture from slice 2, fake time, the actual queue/projection and no forced reload.

```python
async def test_cursor_rearms_three_ordinary_slots(schedule_loop):
    case = schedule_loop
    saved = case.create_human("Report progress", {"kind": "interval", "seconds": 60})
    for _ in range(3):
        await case.advance_and_poll(seconds=60)
        await case.wait_for_physical_settlement()
    assert case.provider_call_count(saved.schedule_id) == 3
    assert case.manual_reload_count == 0
```

Define `schedule_loop` in `test_chat_schedule_queue.py`: `advance_and_poll` advances
the injected clock and lets the real `Scheduler.run` consume its normal reload
notification; it must not call `queue.load` or dispatch the handler directly.

- [ ] Run `python -m pytest Tests/Scheduling/test_chat_schedule_queue.py -q` and establish failure before changing queue integration.
- [ ] Implement local-only projection, durable claim/coalescence and post-commit reload notifications for cursor/retry/control changes. Tick never awaits generation. Use bounded retries/capacity notifications; avoid duplicate task submission on reload.
- [ ] Test a held scheduled tool beside a normal reminder and another ready chat; both progress. Exercise unavailable provider, three-minute active execution on a minute cadence, rollback, midnight and coalesced restart catch-up.
- [ ] Run both new files plus the existing scheduler/service/projection tests touching registration. Review cases 2/3/5/12/14 and commit.

## 4. Composer, Schedules and canonical settings

**Files:** Create `UI/Console_Modules/schedules.py`, `Widgets/Console/console_schedule_setup_modal.py`, `Widgets/Console/console_schedule_status.py`, `UI/Screens/settings_schedule_policy.py`, `css/features/_console_schedules.tcss` under `tldw_chatbook/`. Modify `UI/Console_Modules/wiring.py`, `UI/Screens/chat_screen.py`, `Widgets/Console/console_composer_bar.py`, `Chat/console_command_grammar.py`, `Chat/console_command_suggestions.py`, `UI/Screens/scheduling/schedules_workbench.py`, `UI/Screens/scheduling/task_detail.py`, `UI/Screens/settings_screen.py`, `config.py` and `css/build_css.py`. Create `Tests/UI/test_console_schedule_command.py`, `test_console_schedule_form.py`, `test_console_schedule_controls.py`, `test_console_schedule_settings.py`.

**Interfaces:** `ConsoleSchedulesController` is the thin screen adapter over
`ChatScheduleService`; `open_setup(instructions: str) -> None` opens the shared
review form. `ConsoleScheduleSetupModal` returns an immutable reviewed
`ScheduleMutation` and runtime-issued form operation/context to that adapter.
Its Save never submits a normal chat message. View hook declarations and wiring
must agree with `CONSOLE_VIEW_HOOK_SLOTS` and their viewless defaults.

- [ ] Add mounted command/form tests using the existing app factory, native-ready Console and a recording runtime; drive Paste through the app, not directly through TextArea.

```python
async def test_schedule_save_preserves_user_draft(schedule_console):
    app, pilot, case = schedule_console
    await case.type_composer("/schedule Report progress")
    await case.submit_command()
    await case.fill_schedule_form(kind="interval", seconds=60)
    await case.save_schedule_form()
    assert case.saved_schedule_count() == 1
    assert case.native_provider_call_count() == 0
    assert case.pending_attachments() == case.initial_attachments
    assert case.composer_draft() == case.draft_before_dialog
```

Define `schedule_console` and those helpers in `test_console_schedule_form.py`
over the existing `Tests/UI/app_factory.py` and native flow fixtures. Record the
draft when the dialog opens; query the actual TextArea, database and gateway.

- [ ] Run the new command/form cases first and confirm failure. Implement slash discovery, literal/paste grammar, exact Save freezing, unsaved-chat persistence, duplicate Save identity and validation recovery without clearing the user draft.
- [ ] Implement captured authority/limits/next-run preview, app-running and temporary-files explanations, follow-up consent, status/detail/history and root grant Pause/Resume. Scope fresh full-quota control authority to revocation; display `retime_required` and `occurrence_in_progress` without silently altering cursors.
- [ ] Add source/status semantic classes and defined rest/hover/focus/disabled styles with `$ds-*` tokens. Register the stylesheet in the builder and regenerate the bundle. Add no legacy settings or terminal/global keybindings; do not advertise deferred cron/Run-now controls.
- [ ] Run the four new UI files, `Tests/UI/test_console_runtime_ownership.py`, `Tests/UI/test_design_token_governance.py` and `Tests/UI/test_css_bundle_sync_guard.py`. Verify real modal/control focus and narrow/normal terminal widths, then commit the source CSS and generated bundle together.
- [ ] Review cases 1/4/10/13/19/21/22 with the actual mounted adapter, including paused master settings and navigation/remount.

## 5. Native agent timer tools

**Files:** Create `Agents/schedule_tool_provider.py` under `tldw_chatbook/` and `Tests/Agents/test_schedule_tool_provider.py`; modify tool catalog registration, `Agents/run_context.py`, native bridge wiring and protocol operation-identity propagation at the actual worker boundary.

**Interfaces:** `ScheduleToolProvider(service, context_supplier, operation_supplier)`
implements the current provider protocol: `list_catalog() -> list[ToolCatalogEntry]`,
`load_schema(tool_id: str) -> ToolSchema`, `invoke(tool_id: str, args: dict) -> ToolResult`.
The suppliers resolve live runtime records and the real parsed-call identity, not
arguments, global mutable conversation state or an untrusted `current_run_id` string.
Descriptors use `schedule:` IDs and exactly the five model names from the spec.

- [ ] Write missing-context, forged override, cross-chat and child-engine refusals, with a successful same-chat main-run control. Exercise actual native and fenced parsing plus worker execution, not provider.invoke alone.

```python
async def test_both_protocols_keep_mutation_identity(schedule_agent):
    for protocol in ("native", "fenced"):
        case = await schedule_agent(protocol=protocol)
        await case.model_create_timer(delay_seconds=300, max_runs=2)
        await case.lose_result_and_retry_saved_call()
        assert case.definition_count() == 1
        assert case.original_due_time() == case.current_due_time()
        assert case.operation_identity_before_retry == case.operation_identity_after_retry
```

Define the `schedule_agent` fixture in the new test file using the recording
native gateway and actual agent parsing/worker route. Lost-result injection occurs
after durable mutation, before result delivery; saved-call replay retains its
actual run/generation identity. Repeat for update/pause/resume/cancel and mismatched
arguments, not just create.

- [ ] Run the failing provider/protocol cases. Implement extra-field-forbid schemas, owned pagination, exact-request permissions, persona/profile advertising floors and PreToolUse guards. Enforce fresh human root-task review even where ordinary update permission was previously stored.
- [ ] Persist fenced ordinals before dispatch and scope native IDs to run/generation. Propagate immutable operation context into workers; resolve receipts before CAS and refuse retired/missing origins.
- [ ] Implement finite human-origin grants and inherited scheduled descendants. Goal/fleet without a grant can only propose. Root edits cannot mint budgets; descendant cancellation/recreation does not reset total, original first due time, spacing, deadline or daily reservations.
- [ ] Run the new agent tests plus targeted existing catalog, permission and run-context regressions. Review cases 6/7/8/15/16/17/21; document real model/tool outcomes and commit.

## 6. Combined qualification and PR readiness

**Files:** Create `Tests/Scheduling/test_chat_schedule_end_to_end.py`; update user-facing Console/scheduling documentation, execution Backlog notes and the design review's evidence links. Change CI only where the repository's existing scoped job needs the new test files.

- [ ] Run combined recording-provider tests through real runtime startup, scheduler, controller/bridge, permissions, transcript DB and automatic ledger. Cover both protocols, streaming/nonstreaming, cold viewless startup, navigation, restart and next manual/scheduled history replay.
- [ ] Inject crashes at claim, acceptance, provider completion, transcript commit and accounting commit. Assert no extra physical provider/tool calls and preserved unknown reservations. Hold an effecting tool beyond its wall deadline; no next occurrence starts while the worker/effect remains uncertain.
- [ ] Exercise all 22 spec cases, especially full-quota fresh revocation, root completion with live descendants, one-time edit/settlement races, two live processes and maintenance exclusion. Use a successful control for each refusal path.
- [ ] Run formatter/linter on changed Python files and the targeted tests collected by the six slices. Rebuild/verify CSS and generated architectural manifests affected by the dev integration. Do not claim passing skipped optional dependencies or request a full suite by default.
- [ ] Perform a real terminal UI check and one permitted live provider/tool scheduling round trip following the live-verification lessons; record exact build, origin, grant, occurrence and observed physical outcome. A recording harness is not a live-model claim.
- [ ] Self-review and use the requesting-code-review skill for each completed slice. Resolve meaningful findings, complete AC/notes/docs/ADR links and mark each actual Backlog execution record Done through CLI only with evidence.
- [ ] Update the existing all-session draft PR against `dev` with the final behavior, verification and integration limits; attach it to this chat. Push ordinary commits, preserve hooks and never force push. Readiness requires the task-ID collision resolved in the integrated tree; this plan alone does not resolve it.

## Requirement map

| Spec cases | Owning slices |
| --- | --- |
| 1 | 4 |
| 2, 3 | 1, 3 |
| 4, 8, 9 | 2, 4, 6 |
| 5, 12, 14 | 2, 3 |
| 6, 7, 15, 16 | 1, 5 |
| 10, 17 | 1, 4, 6 |
| 11 | 2, 6 |
| 13, 21, 22 | 1, 2, 4, 6 |
| 18, 19, 20 | 2, 4, 6 |

This plan is an execution contract, not evidence that runtime changes or their
tests already exist. Allocate final migration/task identifiers and pin integration
SHAs at execution time. Keep source capture and owner locks distinct from tool
permission grants, and preserve uncertainty through every frontend and protocol.
