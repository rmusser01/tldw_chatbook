# Completed tool-turn trace transition implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Root alone runs Python, tests, browser probes and Git writes; workers edit their assigned files and report test requests.

**Goal:** Capture the next saved prompt after a completed tool turn without losing history or weakening dispatch ownership.

**Architecture:** Extend the existing typed surface admission with a distinguished one-replacement/one-append shape. Reuse the existing SQLite transaction, ledger and exact response links; add owned pre-dispatch recovery and exact commit reconciliation at the existing gateway boundary.

**Tech Stack:** Existing Python 3.11+, SQLite and pytest stack; no new dependencies or migrations.

**Spec:** `Docs/superpowers/specs/2026-09-05-console-tool-turn-surface-transition-design.md` (including requested review corrections).

ADR required: yes, amendment of an existing ADR.
ADR path: `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`.
Reason: the compound surface and recovery handoff extend cross-module contracts. Amend the ADR before product changes; retain Canvas ADR-121 unchanged.

## Global Constraints

- `MAX_SURFACE_REPLACEMENT_SPAN = 256` is unchanged.
- Exactly one replacement and one append, in that order, under one verified predecessor.
- Cross-turn policy agreement compares all persisted disclosure settings, not freshly allocated opaque IDs; exact within-run and incoming reservation/retry policy identity remains required.
- Eligible next-send routes are `AGENT_FIRST` and `FRESH`; other routes gain no compound permission.
- No Canvas privilege, tool approval, conversation storage, synchronization or V2 change.
- No migration, additional dependency, raw payload logs, transcript-sized durable ID lists, or relaxed startup/performance budgets.
- Root executes only targeted tests through `../../.venv/bin/python -m pytest`; never run a full repository sweep. Workers must not launch Python, import application modules, run tests/browser probes, stage/commit, or spawn agents.
- Preserve the existing dirty controller diagnostic tests, worktree and recovery refs. Do not delete SDD evidence at closeout.
- All expected-error and transport fixtures use synthetic data and real persistence/gateway machinery. Never replace the factory under test with a permissive fake.

## Task 1: Verified compound admission and atomic persistence

**Files:**
- Modify: `tldw_chatbook/Chat/console_trace_final_values.py`
- Modify: `tldw_chatbook/Chat/console_trace_runtime.py`
- Modify: `tldw_chatbook/Chat/console_trace_service.py`
- Modify if a bounded query is needed: `tldw_chatbook/Chat/console_trace_repository.py`
- Test: `Tests/Chat/test_console_trace_runtime.py`, `Tests/Chat/test_console_trace_service.py`, `Tests/Chat/test_console_trace_final_values.py`
- Test: retained nodes in `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**
- Consume `ConsoleTraceBoundaryFactory.__call__`, `SurfaceDeltaAdmission`, `VerifiedSurfaceDelta`, `prepare_current_surface_delta`, `prepare_surface_provenance`, and `bind_and_mark_dispatch`.
- Produce a `CompletedToolTurnWitness` frozen content-free record in `console_trace_final_values.py`, identifying the previous origin/terminal calls and the exact assistant/user revisions. Admission and verified delta carry an optional witness; a witness is evidence rechecked by the service, not a standalone authority token.
- Ordinary callers retain their current signatures/default behavior. Internal signature extensions must use keyword-only optional arguments, with missing proof rejecting compound work. Task 2 consumes the resulting boundary/admission and existing immutable call identity, not a new persisted registry.

- [x] **Step 1: Prepare focused failing tests without product edits.** Preserve the two existing production-factory regressions; remove temporary traceback-frame introspection only after recording its evidence. Extend the calculator case with next-send `FRESH`, a fresh factory, exact before/after native reconstruction and a third completed tool turn. Capture hand-derived final message roles `['user', 'assistant', 'user']`. Add service/type cases for wrong owner/policy/response revision, incomplete terminal call, unrelated reservation, non-tool artifacts, changed prefix, extra new item, noncontiguous/oversized range and unsupported route. Test the observable refusal/unchanged database, not only the dataclass constructor.

  Existing executable regression selectors:
  ```sh
  ../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_chat_controller.py::test_two_saved_turns_keep_history_references_through_production_trace Tests/Chat/test_console_chat_controller.py::test_production_trace_factory_keeps_canvas_tool_loops_on_their_saved_turns
  ```
  The known RED is `unsupported_surface_change` before the second adapter call sequence. First-turn completion, successful calculator42/Canvas staging and drained settlements must be asserted before the failing next-turn expectation.

- [x] **Step 2: Root runs and records RED.** Workers report exact new node IDs and the product break each catches; wait for root's confirmed failure before implementation. Existing negative cases may already pass; distinguish new missing-behavior failures from preserved safeguards.

- [x] **Step 3: Implement the narrow shape.** Give admission/delta an explicit witness-bearing form requiring exactly two new message descriptors; preserve exclusive ordinary replacement/append validation. Derive the range only after matching the unchanged prefix and proving the terminal response link, prior run origin, active tool suffix and exact new saved-user ownership. Keep the existing original-turn/run continuation checks.

  The persistence order is fixed:
  ```text
  validate current owner + terminal call + predecessor + exact two values
  replace active bounded tool suffix with saved assistant revision
  append saved new-user revision
  persist header
  bind reserved call to final head
  advance call to dispatch_started
  commit caller-owned immediate transaction
  ```
  `persist_request` must receive the exact current reserved-call identity from `bind_and_mark_dispatch` to distinguish its own reservation from an intervening call. Do not accept arbitrary caller-supplied ignore lists. Witness validation runs during preparation and again in this transaction.

- [x] **Step 4: Handle structural projection and cold reconstruction.** Update final-value alignment, service-owned child binding, `_ProjectionRoot`, `_DescriptorRoot` and cold/native reconstruction consistently: the replacement is the first new node, never whichever new node is last. A new route or factory must reconstruct a verified parent from durable references rather than blessing the incoming prefix or failing solely for lack of a warm route-specific checkpoint. Ordinary single-replacement and continuation ordering must remain unchanged. Rollback invalidates tentative child/cache state.

- [x] **Step 5: Root runs focused GREEN and the existing trace type/service/runtime files.** Add pre-commit fault injection after each surface operation and call binding; assert neither partial operation survives and all prior native request projections remain identical. This task proves transaction rollback and happy-path next turns; user-facing Retry and ambiguous-commit recovery are Task 2, not claims for this checkpoint.

- [x] **Step 6: Independent task review, fix findings, commit verified Task 1.** Root stages only reviewed files. Record exact commands, results and warning qualifications in the report; do not mark TASK-31742 Done.

## Task 2: Owned retry handoff and commit reconciliation

**Files:**
- Modify only for the saved-turn unresolved-call existence query: `tldw_chatbook/Chat/console_trace_repository.py`; reuse the existing owner/turn index without schema or registry changes.
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/console_trace_runtime.py`
- Modify: `tldw_chatbook/Chat/console_trace_service.py`
- Test: `Tests/Chat/test_console_chat_controller.py`, `Tests/Chat/test_console_provider_gateway.py`, `Tests/Chat/test_console_trace_runtime.py`, `Tests/Chat/test_console_trace_call_lifecycle.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`, only to retain the exact accepted recovery's actor/chain before first-call identity allocation; subsequent tool calls retain that same recovered run.
- Modify: `tldw_chatbook/DB/base_db.py`, only to share the already-verified synchronous `operation_owned_connection(database)` cleanup context across its existing worker consumers; no changed registry or database-wide close policy.
- Modify only if required by the proven worker cleanup boundary: `tldw_chatbook/Chat/console_canvas_controller.py`, `tldw_chatbook/Chat/console_chat_store.py`, `tldw_chatbook/Chat/console_library_policy_coordinator.py` and the existing trace settlement worker module.

**Interfaces:**
- Consume Task 1's witness-bearing boundary and the existing `ConsoleTraceCallBoundary.identity`, reservation, preparation identity and gateway one-shot adapter-entry grant.
- Produce a narrow recovery path from the controller's existing `_trace_call_boundaries_by_preparation` ownership to the gateway/factory. The handoff carries the exact failed boundary and accepted frozen preparation, never a loose conversation ID or bypass boolean.
- Produce exact dispatch-outcome reconciliation inside the trace boundary/service, returning the existing `TraceCallRecord` on proven commit and preserving an owned persistence failure on proven rollback or unknown outcome.
- Release trace database connections owned by short-lived worker operations before their threads exit. Preserve pre-existing caller-thread connections and other same-file owners; no blanket registry sweep or changed database-wide close semantics.
- The measured real-flow cleanup also covers `_run_durable_db_call` and Canvas `_service_call`: synthetic creation stacks identify the saved-turn writer and Canvas tool threads as residual handle owners. Local synchronous `finally` scopes preserve borrowed transactions; do not introduce a Canvas dependency on trace implementation solely for cleanup.
- Keep chat-worker cleanup around the whole synchronous primary agent invocation, not each marker write. The same synthetic probe identifies the library-policy coordinator's `_run_repository_call` as another offloaded handle owner; close only its newly created thread-local handle without changing policy reads, publication, or memory-database execution.
- Deduplicate that exact no-acquisition/borrowed-handle/finally logic in the existing DB utility module, with first-use imports where needed. This is behavior-preserving extraction of the approved scopes, not a new ownership policy or lifecycle registry.

- [ ] **Step 0: Reproduce and correct short-lived worker handle ownership.** Task 1's root probe measured regular-file growth after completed and disposed operations: runtime compound +4, three-turn calculator +16, Canvas +18; forced per-test GC did not change it. The surviving handles belong to the exact test chat databases, and agent lifeline and settlement threads end after the run/runtime. Add exact registered-handle regression tests with real gateway/controller/settlement operations and an independently owned same-file observer. Root records RED before correction. Close only operation-created current-thread handles in the owning worker's `finally`, including error/cancel paths, while preserving pre-existing borrowed connections and active transactions. Reuse existing lifecycle seams; do not fix this by quiescing the entire database in fixture teardown. Re-run per-test resource probes and historical reconstruction. ADR required: no new ADR for this lifecycle bug fix; existing ADR-097 transaction/ownership boundaries apply. A new shared ownership policy would require a separate design decision.

- [ ] **Step 1: Write real recovery regressions first.** Inject a one-shot pre-commit failure after Task 1 replacement/append, drive the actual controller Retry action, and assert stable call ID/idempotency key, a single call-boundary event and one eventual adapter entry. Repeat failures twice before success. Negative cases substitute another preparation's boundary, mutate the saved request/route/destination, introduce an unrelated reservation, retain a stale child capability, and present a terminal `NOT_DISPATCHED` call; each refuses transport.

  Outcome assertions are independent literals, for example:
  ```python
  assert final_call.call_id == reserved_call_id
  assert final_call.idempotency_key == reserved_idempotency_key
  assert adapter_entries == 1
  assert matching_call_boundary_event_count == 1
  ```
  Root runs each new selector and confirms RED before the handoff is implemented.

  Cover the existing explicit recovery actions as well: `Send without capture` and `Cancel` after a proven pre-dispatch failure, for both FRESH and AGENT_FIRST. Capture Off requires the explicit user action and exact accepted-turn ownership; the original trace reservation must remain terminal if canceled/not-dispatched, never revived. Unknown delivery still blocks automatic resend. These controls implement the spec's existing blocked-capture UX, not a new bypass mode.

- [ ] **Step 2: Implement exact owned re-admission.** Thread the exact previous boundary through existing frozen continuation ownership to the gateway's private recovery path. Verify issuer/identity and the same accepted preparation, saved revision, request, route, destination and policy. Require durable `RESERVED`, no bound head/header or dispatch/response timestamps, unchanged predecessor and no unrelated later call. Retire the old capability, rebuild verification, and reuse the immutable reservation rather than calling ordinary allocation. Cold recovery lacking exact existing durable proof remains blocked; do not add a new persistence registry or revive terminal calls.

- [ ] **Step 3: Write post-commit fault regressions.** Inject an exception after the transaction successfully commits, including WAL-setting restoration; verify exact call and composed head/header are durable. Inject a reconciliation read error separately. Test both the original live invocation's unconsumed adapter grant and a new/cold invocation. Inject a wrapper error after adapter entry. Check cancellation cannot convert committed dispatch to `NOT_DISPATCHED`.

  Required observable outcomes:
  ```text
  rollback => owned reserved call remains retryable, zero partial surface writes
  known commit + original unconsumed live grant => same bound call, at most one entry
  known commit + new/cold invocation => no automatic entry, preserve uncertainty
  unreadable/inconsistent outcome => no entry or automatic retry, retain owned failure
  wrapper error after entry => no second entry
  ```
  Root records RED before reconciliation changes.

- [ ] **Step 4: Implement three-way exact read-back.** On write/cleanup exceptions, inspect the exact call identity and expected final surface/header under an owned usable connection. A matching committed boundary reconciles in-memory state without replaying writes. An unbound reservation plus unchanged predecessor establishes rollback. Anything else remains unknown. Retain the original exception's privacy boundary; no payload or exception-body logs. A proven committed original live call may continue only with its exact unconsumed gateway grant; do not issue a replacement grant to a new invocation. Preserve database lifecycle state through Retry/Cancel recovery.

  Cold recovery must not allocate another `AGENT_FIRST` for an existing actor/chain.
  Two legacy recreated-factory tests expect such duplicate first-call allocation;
  replace those conflicting expectations with duplicate-first refusal and legitimate
  cold `TOOL_LOOP` sequencing after a real completed response. Preserve the exact
  owned Retry path and do not broaden other route permissions. Test cancellation
  after commit without swallowing it to enter the adapter, and a transient verifier
  re-preparation failure without losing the surviving reservation's ownership.
  For `FRESH`, reject replay of the same saved turn while a prior call remains
  unresolved; retain completed ordinary replacements, new saved turns, and the
  existing explicit `RETRY` route used by Retry anyway. Verify the refusal with
  the actual cold gateway after unreadable postcommit state, not preparation alone.
  Check the requested saved turn even when another turn has a later call; a new
  gateway may reuse a warm factory, so a cold-factory replacement refusal alone
  is not sufficient recovery ownership evidence.

- [ ] **Step 5: Root runs new recovery cases and the four affected test files.** Also repeat Task 1's production calculator/Canvas controls to verify recovery integration has not changed healthy sends. Independently review the task and commit only verified changes.

## Task 3: Integration, growth evidence and protected PR completion

**Files:**
- Test: existing `Tests/Chat/test_console_trace_*` growth/lifecycle tests and `Tests/Canvas/browser/test_canvas_{native_flow,served_flow,zero_egress}.py`
- Update: `Docs/Canvas/V1_VERIFICATION.md`, the owning Backlog task and this plan's checkboxes.

**Interfaces:** Consume the public gateway and native-reader behavior from Tasks 1/2; produce evidence and reviewed PR updates, not a new runtime API.

- [ ] **Step 1: Strengthen measured growth coverage.** Extend the existing real-gateway fixture with repeated completed-tool/next-send transitions; assert hand-counted new surface nodes/events per transition and unchanged earlier reconstruction. Keep ADR-097 growth and latency gates unchanged. Root executes relevant existing tests; no full suite.
- [ ] **Step 2: Run affected trace, gateway/controller, Canvas and startup checks.** Preserve existing warning/baseline qualifications rather than raising budgets. Run all three mandatory Chromium Canvas browser files after runtime integration is stable; optional browser absences remain explicit skips, not claims of coverage.
- [ ] **Step 3: Broad review of this repair range and its integration with Canvas.** Reviewer gets the approved spec, complete diff since `c2d5aac3a`, reports and parked findings. Resolve load-bearing issues before merge readiness.
- [ ] **Step 4: Update ADR/task/evidence, run derived preflight, commit and publish.** Recheck actual dev, rebase if needed with exact force-with-lease and recovery ref retained; inspect overlapping changes and reverify affected behavior. Read Qodo feedback on the new published head and address every finding. Wait for protected current-head CI. Do not merge a known failing integration case even if CI passes.
- [ ] **Step 5: Merge PR2432 normally into dev and verify merge SHA.** No admin bypass or deletion of worktrees/recovery refs. Only after verified merge begin V2 brainstorming; V2 implementation is not part of this plan.

## Plan self-review

All spec sections map to Task 1 (shape, proof, persistence, reconstruction), Task 2
(retry/commit outcomes and adapter ownership), or Task 3 (growth, integration and
PR gates). Tasks 1 and 2 intentionally share runtime/service files and execute
serially. Controller tests shared across all tasks are owned by one worker at a
time. No task treats a response link or matching text as permission. Root retains
all executable verification and Git operations; intermediate task completion is
not merge readiness.
