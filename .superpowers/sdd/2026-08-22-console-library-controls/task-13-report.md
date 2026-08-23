# Task 13 implementation report

## Outcome

Task 13 moves eligible immediate and queued ordinary user-text sends behind one
store-owned automatic Library preparation gate. The gate consumes the exact
Task-8 execution context and executed draft, always searches the fixed
notes/media/conversations categories, applies the frozen active Note/Media scope,
and makes provider composition/call unreachable until preparation is ready.

The store owns one immutable preparation per session with exact ID/state CAS,
navigation-independent lifetime, and a shared cancel path for manual, queued,
close, and shutdown handling. Evidence success seals one bundle used by the same
provider request; zero-match and bypass attach only the bounded Task-12
contribution; failure/timeout pauses with bounded codes and never dispatches.
Retry keeps the preparation ID and frozen request/authority/destination while
creating a new attempt. Bypass does not change standing policy.

The mounted retrieval controller no longer reads or performs automatic-RAG
send work and no longer emits fail-open/placeholder notices. Its explicit manual
Search Library launch/capture remains intact; the obsolete standing
`rag_auto_retrieve_on_send` modal, callback, and runtime/config ownership are
removed, leaving only the one-time compatibility migration input.

Task 13 uses volatile ACCEPTED/DISPATCH_STARTED transitions only as an in-memory
provider fence. It does not claim Task-14 durable acceptance/checkpoint atomicity,
post-commit recovery, assistant terminal state, or any recovery UI. A queued
retrieval pause returns the exact claim to pending, stops the current coordinator
advance, and does not auto-retry or spin later entries.

## Compatibility and security review

Every ready gateway must supply Task-9's typed, credential-free classified
destination. Missing destinations fail closed before preparation/provider
dispatch; Retry re-resolves and requires exact equality with the frozen
destination. There is no `unknown://unresolved` synthesis, so Task-8's Unknown
fail-closed boundary cannot be weakened by a compatibility fallback.

Automatic admission is a closed map: manual and queued plain user text are
eligible, including nonblank text with attachments; attachment-only, agent wakes,
commands, skill mentions, explicit evidence, Never, bypass, and unknown future
origins skip automatic spend. Explicit evidence and Never still use the hidden
commit preparation required by spec section 6.3, without retrieval or a visible
Preparing-Library projection. The success integration test proves the exact
sealed bundle content reaches the same provider request, while failure and race
tests fence all provider calls.

## TDD and verification

- Pre-edit affected baseline: 464 passed, 1 inherited
  `RequestsDependencyWarning`, 43.44s.
- RED: the complete new test file failed collection because
  `ConsolePreparationOutcome` did not exist, before production edits.
- Final affected command:

  `../../.venv/bin/python -m pytest Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_auto_rag_on_send.py Tests/UI/test_console_rag_settings_modal.py Tests/UI/test_console_retrieval_controller.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_turn_preparation.py Tests/Chat/test_library_preparation.py Tests/UI/test_console_controller_wiring.py -q`

  Result: 486 passed, 1 inherited warning, 34.90s.
- Scoped Ruff and `git diff --check` passed.
- Required restored mutations all failed their named ratchets: retrieval-failure
  fall-through, current-composer query, narrowed/manual categories, duplicate
  explicit-evidence retrieval, Retry refresh, Bypass policy mutation, weakened
  CAS (three apparent winners), and queued Cancel foreground copy.

An adjacent `test_console_runtime_ownership` check has a pre-existing isolated
fixture defect: its `object.__new__(TldwCli)` lacks `notes_sync_runtime_owner` and
raises before reaching Console shutdown. Task-13 production does not touch that
runtime owner and this warning was not expanded into unrelated scope.

## Fix round 1

Review-driven production-path REDs proved that Retry and Bypass resume the exact
already-admitted manual or queued send without a second submit, preserving its
draft, Task-8 authority, destination, attachment/evidence/prefill identities, and
queue owner. Queue recovery retains the exact claim, blocks later entries without
spin, and either completes that claim once or releases it on Cancel.

State transitions now follow the existing live boundaries: COMMITTING starts at
the acceptance attempt, ACCEPTED follows the established USER/assistant ownership
boundary, and DISPATCH_STARTED occurs immediately before the real agent/provider
call. Provider preflight refusal therefore settles an ACCEPTED volatile owner
without falsely claiming dispatch. Preaccept persistence/assistant/refusal and
queued-authorization exits roll back or cancel exact ownership; accepted owners
survive store session removal until their live turn settles. This is process-memory
state only and makes no Task-14 durability/checkpoint claim.

The final review gate was 578 passed with the inherited Requests warning in
49.07s. Runtime ownership companions were 24 passed; the UI ownership set was 10
passed/1 deselected with the independently reproduced fixture defect above. The
Task-13 production-path file was 34 passed. Nine restored mutations killed the
dead-end continuation, stranded refusal, attachment exclusion, fail-open evidence
probe, eager dispatch claim, unconditional accepted-owner close pop, legacy
toggle, missing-destination fallback, and outcome leak variants. A separate
RED/green boundary probe caught and fixed stranded USER-persistence failure.
Scoped Ruff, whole-production source scans, and `git diff --check` passed.

## Files

- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- `tldw_chatbook/Chat/console_runtime.py`
- `tldw_chatbook/UI/Console_Modules/retrieval.py`
- `tldw_chatbook/UI/Console_Modules/session.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/Widgets/Console/console_rag_settings_modal.py`
- `tldw_chatbook/config.py`
- `Tests/Chat/test_console_automatic_library_preparation.py`
- adjacent authority/retrieval ownership regression tests
- TASK-19900.3 plan/notes, this report, and the shared progress ledger

ADR required: no. ADR path:
`backlog/decisions/079-console-library-conversation-authority.md`. ADR-079 already
owns the implemented admission, authority, recovery-action, and fail-closed
provider-boundary decisions.

## Fix round 2

Recovered queued sends now keep their exact claimed entry until one coordinator
finalizer observes the resumed result. Accepted results acknowledge once and
advance normally; refusal or exception returns the exact entry to the pending
head, releases the reservation, pauses later entries, and never spins. Only the
recovered path defers registry settlement; the controller's volatile ACCEPTED
state still begins at the existing USER/assistant ownership boundary.

The live-only continuation now freezes the original attachment objects, resolved
prefill value and one-shot identity, and production staged-evidence launch at
admission. Retry and Bypass never reread current staged state, consume only the
original matching identities after acceptance, and leave attachments, evidence,
prefill, and composer changes made during the pause untouched. This remains
volatile Task-13 state and makes no Task-14 reconstruction or durability claim.

Recovery destination resolution, queue reclaim, submit exceptions, stale CAS,
and repeated action races return the stable `ConsoleSubmitResult` contract and
cannot strand READY or a claimed entry. `DISPATCH_STARTED` is now crossed only
immediately before `stream_chat` or `agent_bridge.run_reply`; direct request
preparation and agent setup remain ACCEPTED and settle without a false dispatch.
Session close synchronously removes only cancellable preparations; an accepted
owner remains until its cancelled live task's existing finalizer settles it.

Round-2 baseline was 578 passed with one inherited Requests warning. The complete
RED matrix produced 22 expected failures and 35 passes before production. Final
affected verification was 601 passed/1 inherited warning in 38.49s; runtime,
queue-registry/coordinator, and queue-UI companions were 100 passed/1 inherited
warning in 16.97s. Eight restored mutations were killed: removed recovered
finalization, live staged-input reread, resolver exception escape, loser
`KeyError`, early direct dispatch, early agent dispatch, eager close settlement,
and continuation/evidence leakage. Scoped Ruff, Ruff format check, source scans,
and `git diff --check` passed. The sole production occurrence of the old config
key remains ADR-079's required one-time v44→v45 migration seed; there is no
standing automatic-policy read or UI callback.

## Fix round 3

Post-accept queued recovery now distinguishes the exact coordinator-owned
`accepted_live_turn` from a preaccept refusal. A direct request-preparation or
agent setup exception after USER/assistant acceptance returns a stable accepted
`ConsoleSubmitResult`, retains the exact USER and failed assistant identities,
acknowledges the reclaimed entry once, and pauses later work without returning
that accepted entry to pending or creating a second USER on Retry/Bypass.

One volatile `_active_submit_tasks` fence now owns the complete submit lifecycle,
from before the first await through the submit finalizer. Shutdown tombstones new
work, cancels and awaits both submit and stream tasks, removes a pre-preparation
transient echo coherently, and rechecks immediately before the direct provider or
agent bridge call so a cancellation-resistant preflight cannot dispatch later.
Close cancels the exact submit but preserves COMMITTING/ACCEPTED preparation and
sidecars until that task's finalizer settles them. Self-shutdown never awaits or
cancels its own task. These owners remain volatile only and add no Task-14
checkpoint or restart claim.

One-shot prefill consumption now uses a store-owned monotonic opaque revision,
so re-arming identical text during an in-flight turn survives the older turn's
compare-and-clear. Explicit staged evidence is held as an exact live launch lease
plus its captured production result: context capture does not consume the launch,
acceptance releases only that identity, and preaccept failure, Cancel, or a newer
replacement leaves the appropriate launch staged.

Round-3 baseline was 601 affected tests and 100 runtime/queue/UI companions.
The bounded RED run produced 11 expected failures before teardown hardening, then
14 focused review probes passed. Final affected verification was 613 passed and
runtime/queue/UI companions were 100 passed, each with the inherited Requests
warning. Seven restored mutations were killed: ignored accepted-live ownership,
unawaited submit shutdown, missing direct and agent shutdown rechecks, early
COMMITTING close removal, equality/unconditional prefill clearing, and early
evidence release. Scoped Ruff lint/format, source/privacy scans, and diff checks
passed.
