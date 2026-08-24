# Task 13 brief — truthful automatic Library pre-dispatch gate

## Scope

Implement Task 13 from the approved plan. Replace the Console screen's current fail-open automatic-RAG ownership with one controller/store-owned preparation gate for eligible immediate and queued user-text sends. Use Task 12's immutable preparation model and sidecar contribution. Do not implement Task 14 durable acceptance/checkpoint atomicity, post-commit recovery, assistant terminal state, recovery UI, or later activity/UI tasks.

This is a direct implementation of the approved spec and ADR-079. Apply the brainstorming skill only to reconcile existing seams; do not reopen the approved product behavior. ADR required: no new ADR. ADR-079 owns admission, fixed source categories, scope, recovery actions, and fail-closed provider boundary.

## Authorities

Read completely before production edits:

- repository `AGENTS.md`;
- Task 13 plan section and the frozen ledger/interfaces;
- spec §§6.1–6.3 and §§11.3 plus the queue/temporary clauses that Task 13 consumes;
- ADR-079 automatic preparation decisions;
- Task 12 brief/report and the actual preparation/sidecar contracts;
- Tasks 8–11 context/authority/destination/provider reports;
- TASK-19900.3 and relevant testing/live/backlog lessons;
- current retrieval controller, Console controller/store/wiring, queue coordinator, active-scope resolver, staged evidence, and real submit/dispatch tests.

## Files and frozen interfaces

- Modify only the plan-listed production seams and necessary adjacent tests:
  - `tldw_chatbook/UI/Console_Modules/retrieval.py`
  - `tldw_chatbook/Chat/console_chat_controller.py`
  - `tldw_chatbook/Chat/console_chat_store.py`
  - `tldw_chatbook/UI/Console_Modules/wiring.py`
  - new `Tests/Chat/test_console_automatic_library_preparation.py`
  - relevant controller/retrieval/queue tests.
- Consume `ConsoleTurnExecutionContext`, Task12 preparation contracts, current RAG service/scope resolver, and sealed staged-evidence bundle.
- Produce controller `prepare_library_for_turn(preparation_id: str) -> ConsolePreparationOutcome` and store-owned `begin_preparation`, `compare_and_set_preparation`, `preparation_for_session`, `cancel_preparation` with exact preparation-ID/state CAS and session lifetime independent of mounted screens.

## Required behavior

### Admission and fixed authority

- Automatic retrieval runs only for admitted ordinary plain user text, manual or queued, when the frozen conversation policy is Automatic, no explicit evidence bundle is already staged, and the turn is not a one-shot bypass.
- Skip commands, approvals/tool responses, retry/regenerate/edit-resend/continue, wakes/machine/system input, attachment-only/ineligible kinds, explicit evidence, Never, and bypass. Pin every current send kind explicitly; default unknown kinds fail closed/skip rather than silently spend.
- Capture after execution admission/queue claim using the exact executed draft and Task8 final execution context. Query equals that immutable executed draft, never current composer text.
- Fixed categories are exactly notes/media/conversations and never read or mutate manual Search Library modal toggles/state.
- Active item scope narrows exact Note/Media IDs and excludes Conversations under existing scope semantics. No active item scope uses the fixed full category set.
- Retry preserves preparation ID, executed draft, frozen policy revision, scope, selector, provider intent/destination, attachment/evidence/prefill IDs, and queue ownership, but creates a new retrieval attempt ID. It never silently refreshes policy or destination.

### Outcomes and dispatch boundary

- Evidence found: seal one evidence bundle and attach that exact object/content to the exact prepared request that later dispatches; do not duplicate spend/context.
- Zero matches: advance to ready without evidence and attach one bounded `LibraryPreparationContribution` for atomic persistence by Task14; no query/source identity retained.
- Timeout/service failure: pause with retrieval kind before provider composition/call. Return only bounded error code/category, never exception text/log secret, and do not fall through.
- Retry re-runs the same frozen request under a new attempt. Bypass advances to ready with a bypass contribution, changes no durable/in-memory standing policy, and is offered only for retrieval pause. Cancel sends nothing.
- Provider composition/call must remain unreachable until preparation reaches the legal ready/commit flow. No failure or race may silently dispatch.
- Never goes directly to ready and performs/displays no Library preparation.

### Store ownership, cancel, queue, and races

- One live preparation per affected session, owned by `ConsoleChatStore`, survives screen replacement/navigation. Begin/read/CAS/cancel are thread-safe and exact by preparation ID/state; repeated/racing actions are idempotent with one winner.
- Manual Cancel restores/preserves the exact draft, attachments, explicit evidence, one-shot prefill, and removes only the transient optimistic echo.
- Queued Cancel releases the same claimed entry to pending, never copies it into the foreground composer, and prevents provider dispatch. Later queue advancement semantics must remain current and safe; Task15 owns durable post-acceptance recovery.
- Close/shutdown cancellation uses the same exact-once store path. Racing Retry/Bypass/Cancel/close/shutdown has one winning transition and zero provider calls before acceptance.
- Do not add recovery widgets in this task; tests may drive controller/store action methods directly. Task18+ owns presentation.

### Controller lifecycle boundary

- Supported clean teardown requires awaited `ConsoleChatController.shutdown()`
  to run on, or coordinate with, every submit task's live owner loop before
  that loop closes. This path cancels and awaits all non-current tasks to a
  terminal state and must emit no pending/destroyed-task or never-awaited
  diagnostic.
- `begin_shutdown()` after an owner loop is already closed is emergency
  fail-closed detachment only. It synchronously removes controller/store
  volatile ownership and exact exclusively owned preparation sidecars, cannot
  dispatch, and must not call closed-loop scheduling, cancellation, or await
  APIs. Public asyncio cannot make the abandoned pending Task terminal, so this
  path does not promise clean shutdown, recovery, durability, or suppression of
  Python's destroyed-pending-task diagnostic.
- Lifecycle tests must use public loop exception handlers, warning capture, and
  weak references. They must not set private asyncio Task flags or manually
  close a Task's coroutine to alter diagnostic behavior.

### Ownership migration

- Remove standing automatic-policy/config reads and fail-open automatic retrieval notices from the mounted screen retrieval controller. Preserve all manual search methods/behavior.
- Controller/wiring use only the frozen final context and app/store lifetime services. No screen-owned preparation truth, cached policy, manual toggle dependency, or provider/API-key inference.

## TDD and verification

1. Confirm clean head and TASK-19900.3 stays In Progress with all ACs unchecked. Append a Task13 plan subsection/notes only; Task17 owns Done.
2. Run and record a focused pre-edit baseline for existing controller, retrieval modal/controller, queue coordinator, Task8 authority, and Task12 preparation/sidecar tests.
3. Write all admission, scope, outcome, cancel, queue, and race tests before production edits. Prefer real `submit_draft`, real queue claim, real store, and a fake RAG boundary/provider fence over private-helper-only tests. Capture meaningful RED.
4. Implement the smallest compatible gate. No Task14 persistence/checkpoint schema or Task18 UI.
5. Run the new file plus full affected controller/retrieval/queue/context/preparation tests and scoped UI wiring tests. Run scoped Ruff and `git diff --check`; no full repo suite.
6. Required mutation probes:
   - remove the retrieval-failure early return and prove provider call occurs/test fails;
   - use current composer text instead of executed draft;
   - read manual modal categories;
   - allow explicit-evidence duplicate automatic retrieval;
   - refresh authority/policy/destination on Retry;
   - make Bypass mutate policy;
   - weaken CAS so two race actions win;
   - make queued Cancel copy into foreground or fail to release the exact claim.
   Restore every mutant and prove final GREEN.

## Governance and handoff

- Self-review all actual send kinds and every provider-call path. Search for the deleted fail-open/standing-config automatic path and prove manual search remains.
- Record a lesson only for a concrete reusable incident.
- Write `.superpowers/sdd/2026-08-22-console-library-controls/task-13-report.md` and append the shared progress ledger.
- TASK-19900.3 remains In Progress with all ACs unchecked; notes must state Task14+ durable acceptance/recovery and UI remain incomplete.
- Commit once as `feat(console): gate sends on Library preparation`; do not push. Report baseline/RED/GREEN/mutations/Ruff/diff/hash/files/warnings/concerns and leave a clean worktree.
