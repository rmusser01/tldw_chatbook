---
id: TASK-42412
title: "The recovered-suites managed received acceptance test fails because its Send stops at trace reservation with a plain ValueError on the CI runner"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 21:17
updated_date: 2026-11-23 05:42
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/Performance/test_console_managed_received_acceptance.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Chat/console_send_diagnostics.py
- tldw_chatbook/Chat/console_trace_errors.py
- tldw_chatbook/Chat/console_trace_runtime.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `.github/`.

**Why.** `Tests/Performance/test_console_managed_received_acceptance.py::test_received_managed_demand_uses_one_catalog_and_real_cold_initializer` fails the required "PR lane (recovered suites)" job with `managed_received_prerequisite_timeout`, for reasons outside the PR under test. In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), it failed in 24 recovered-suites jobs, from 2026-11-20 04:52Z to 2026-11-22 20:13Z, including runs 44954347282, 44974735998, 44974742439, 44977408277, 44978286545 and 45050980929. Its body runs in its own child process, and it passes alone locally (16.7 s, TASK-33620.15.61's notes). TASK-34563.35 (In Progress) wrote it as its AC#2 qualification under ADR-230, and TASK-41564's notes mention it; neither tracks this failure.

**What the logs show.** In the five of those runs whose logs were read (44954347282, 44974735998, 44974742439, 44977408277, 44978286545), the signature is the same:
- the Send's `trace_reservation` stage fails with `exception_type=ValueError error_category=validation`, when the Send attempt is about 6.2 s old (`duration_ms=6241` in 44978286545; `duration_ms` counts from the attempt's start, `SendDiagnostic.started`, `Chat/console_send_diagnostics.py:46` and `:98`, not from the stage's start);
- the child then fails at script line 770, `await settle(entered.is_set)` (test file line 778 at this base), the wait for the controlled preparation read after the Send;
- the call takes about 133-135 s. In 44978286545 the stage failure is logged at 06:31:04 and the test fails at 06:32:58, which fits the 120 s `CUSTODY_HANG_BOUND` counted from the Send.

So raising the bound would not help. 767c15cffb (2026-11-19) already turned `settle()`'s 10 s deadline into that hang bound, and every listed run is later than it. The Send never reaches the read the test controls.

**Where the ValueError comes from.** `console_send_diagnostics.py:140-168` walks the cause chain and records the innermost exception's type. A `ValueError` whose single argument is a known trace-failure code would be recorded under that code, and a `TraceReservationRefused` or `TraceSurfaceChangeRefused` would be recorded under its own class name (`console_trace_errors.py:39-56`). So the root is a plain `ValueError` with some other message, raised inside `_reserve_trace_call` (`Chat/console_provider_gateway.py:3582-3671`), whose broad `except Exception` turns it into `TraceCallPersistenceError(reservation_status="unknown")`. The diagnostic records only the type, never the message, and the stage's `entered` INFO line is not in the child's captured output, so the CI logs cannot say more.

**Coordination.** If the fix touches `.github/workflows/derived-artifacts.yml`, coordinate with tldw-chatbook-30's open PRs (#3656, #3683, #3688) and its coming TASK-42281 lanes PR before editing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The plain ValueError raised during this test's trace reservation is identified: its message and the line that raises it are recorded in the task notes, from a local reproduction or a temporary CI diagnostic
- [x] #2 The root cause is fixed in product code, or, if the refusal is correct behaviour for that state, the test sets up the state the reservation needs; the notes say which and why
- [x] #3 A reproduction that fails the test before the change (a temporary slowdown or fault injection, reverted) passes after it
- [x] #4 The test passes in the recovered-suites job in at least three consecutive PR runs after the change, or in a local run of the job's step under parallel load
- [x] #5 Review found a second cause in the same test: under load its backend switch (`handle_runtime_backend_changed("local")`) fails fast with `FolderConflictError` ("database is locked") while creating the agent-lessons folder during the startup Actor Pack recovery. The lock holder is proven, the contention is fixed in product code, a forced lock hold fails before the fix and passes after it, and the test passes 20 times under heavy load
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: run the test alone, then after the files that precede it in the "Run the recovered suites" step (serial, `--storage-lease-guard --timeout=180`), and under load. Add a temporary diagnostic at `_reserve_trace_call`'s broad except (and the child's captured output) that prints the ValueError's message and raising line; revert it afterwards.
2. Classify the cause: product bug in trace reservation (fix in product code, red-then-green test) or state the test leaves out (fix in the test or the leaking file).
3. Show a forced reproduction (temporary slowdown or fault injection, reverted) failing before the change and passing after it.
4. Soak: 10 serial runs in the step neighbourhood, plus one whole recovered-suites step run; record counts and load.
5. Preflight, ruff F and the size ratchets for touched modules; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the test (AC#2's second branch). The refusal is correct behaviour for the state the test pressed Enter in; no product change.

**The ValueError (AC#1).** `ValueError("surface_replacement_checkpoint_unavailable")`, raised at `tldw_chatbook/Chat/console_trace_service.py:2927` (`prepare_current_surface_delta`, `if replacement_range is not None and bootstrap`). It is reached from `ConsoleTraceBoundaryFactory.__call__` (`console_trace_runtime.py:816`) on the agent's TOOL_LOOP call (`first_call=False`). The stack runs from `console_agent_bridge._run` through `Agents/activation.py:373` and `console_streaming_request` to `stream_chat` (`console_provider_gateway.py:6163`). `_rolled_back_refusal` (`console_trace_runtime.py:84`) re-raises it as `TraceReservationRefused("surface_replacement_checkpoint_unavailable")` from that ValueError, and `_reserve_trace_call`'s broad except catches it. Because the type is `TraceReservationRefused` and `boundary` is None, the reservation status is `not_established`, not `unknown` as the description assumed. The diagnostic shows `ValueError`/`validation` because its cause walk ends at the inner ValueError and that code is not in `_TRACE_FAILURE_CATEGORIES`. Recorded with a temporary log at that except (reverted).

**Why the Send got there (AC#2).** The test pressed Enter before the Console's arrival read of the emergency stop (`schedule_emergency_stop_refresh` -> `emergency_stop_hold.refresh`, an off-loop read after `_ui_ready`) had landed. While `_console_emergency_stop_path` is None, `received_send_held()` returns True (`emergency_stop_hold.py:137-142`, "doubt holds: the dispatch path decides"). `receive_console_visible_intent` then returns None (`UI/Console_Modules/wiring.py:485`), and the Send takes the ordinary `accept_turn` route. A temporary log at every `return None` in that function and in `capture_console_received_configuration_preparation` recorded `DECLINE wiring:486` (the instrumented line). On that route there is no received intent, so the test's observer never records `hook_ready` or the custody record, and all 11 monitored callbacks fail (KeyError, or AttributeError on a record whose `received_intent` is None). That includes the builder callback, which fails before `entered.set()`, so the trust build is never held. The agent run goes on: AGENT_FIRST reserves, and the TOOL_LOOP call is refused as above. The child then waits out `CUSTODY_HANG_BOUND` at `settle(entered.is_set)`. `Tests/UI/test_console_received_intent_feedback.py` and `test_console_configuration_worker_feedback.py` already wait for the same read ("sending then correctly takes the ordinary dispatch route"). The fix adds that wait, `settle(lambda: screen._console_emergency_stop_path is not None)`, after the attach-reconcile wait and before the draft is loaded. No assertion or bound changed.

**Natural reproduction.** In 1 of 18 local standalone runs (load 22-31, 6 run concurrently at times, a step run going in parallel): the same 11 observer errors, then AGENT_FIRST OK and TOOL_LOOP refused with `surface_replacement_checkpoint_unavailable`. That run came before the decline log, so its decline point rests on the identical signature and the forced run below.

**Red/green (AC#3)**, with a temporary env-gated `time.sleep(8)` in `emergency_stop_hold._read_default_stop`, reverted:
- Without the fix: `1 failed in 131.93s` and `1 failed in 149.28s`. The child printed `duration_ms=6482 error_category=validation exception_type=ValueError phase=trace_reservation`, then `AssertionError: managed_received_prerequisite_timeout` at `settle(entered.is_set)`, the CI signature (CI: `duration_ms` 6241/6405/8014).
- With the fix: `1 passed in 26.44s`, with all 11 phases in order and only the AGENT_FIRST reservation.

**Soak (AC#4), serial, step flags (`-p no:xdist --storage-lease-guard --timeout=180`).**
- Ten runs of the step's last 18 targets (held-pointer x3, stock-owned x14, this test): 10/10, 180/180 passed, load 24-46. This test took 14.4-35.4 s per run, and a clean whole-step run was going alongside from run 4 on.
- Whole "Run the recovered suites" step, first run (the managed child ran with the fix and clean source; the parent had imported the temporary diagnostics): 2006 passed, 1 skipped, 1 deselected in 1742.76 s; this test took 18.5 s.
- Whole step, three more runs on clean source, all serial (runs 3 and 4 kept their full output):
  - Run 3 (load 23-35): 2006 passed, 1 skipped in 1220.75 s; this test took 24.6 s.
  - Run 4, with the 10-run neighbourhood soak going in parallel (load 16-37): 2006 passed, 1 skipped in 1532.69 s. The parallel soak passed 10/10 again (180/180).
  - Run 2, alongside the first soak (load 24-46): 2 failed, 2004 passed. The two failures were this test and `Tests/Library/test_library_ingest_runner.py::test_real_pool_worker_exit_is_reported_for_owning_generation`. This test failed in under 12.2 s (it is missing from the 5 slowest durations, whose cutoff was 12.24 s), so this is not the 120 s `settle(entered)` hang fixed here. The step script kept only the last 15 lines, so the failure text was lost, and the run did not recur in runs 3 and 4 or in 20 soak runs. Likely cause, found by review: the agent-lessons seed race fixed under AC#5 below. That race fails the test at the backend switch, about 12 s into the child, and the reviewer's failing calls took about that long.

Out of scope here (proposed as follow-ups):
- Add the trace refusal codes the send diagnostic does not know (`surface_replacement_checkpoint_unavailable` and others) to `_TRACE_FAILURE_CATEGORIES`.
- Check whether an agent's next call after a failed first call should be refused by trace reservation.
- Grep-audit the other read-then-write `db.transaction()` sites (no `immediate=True`) on shared ChaChaNotes handles; any of them fails the same way under a concurrent writer.

Files: `Tests/Performance/test_console_managed_received_acceptance.py` (one wait, 8 lines); `backlog/docs/lessons-testing-evidence.md` (one entry).

**Second cause, found by review (AC#5).** Under heavy load (16 busy loops plus 3 lanes, load 31-78), review saw 6 of 91 runs fail fast at `assert await app.handle_runtime_backend_changed("local") is True` (script line 662). The error was `FolderConflictError("Folder changed during mutation.", reason="busy")` from `sqlite3.OperationalError: database is locked`. The chain is `handle_runtime_backend_changed` -> `_wire_notes_sync_services` (`app_service_wiring.py:649`) -> `initialize_agent_lessons_folder` (`Notes/agent_lessons.py:337`) -> `LocalNoteFolderRepository.create_folder(cursor=...)` INSERT. The failure comes before the emergency-stop wait, so it was on base too, and it is a product defect: a user's backend switch during startup recovery fails the same way.
- *The holder.* A temporary probe logged every ChaChaNotes transaction begin and end (per connection, thread and caller), plus the SQLite code at the failure. It caught a natural failure on base code (1 failure in the roughly 21 runs before the fix was applied, at load 20-50; times in seconds since the child started):
  - 12.4551: the UI thread's DEFERRED transaction begins at `agent_lessons.py:270` and reads.
  - 12.4600: the startup Actor Pack recovery thread (`asyncio_0`, `ensure_actor_pack_recovery` -> `PersonaBuddyLibrary.ensure_builtin` -> `publish_review`) runs `BEGIN IMMEDIATE` at `Persona_Buddy/library.py:247` to insert its `buddy_profiles` row.
  - 12.4676: the recovery commits.
  - 12.4719: the UI thread's INSERT fails with `SQLITE_BUSY_SNAPSHOT` (517).
  The reviewer's thread dumps were taken in the error handler, after the recovery thread had already moved on to `persona_visual_participants.files`.
- *Why neither the busy handler nor the retry absorbs it.* The seed reads and then writes in a DEFERRED transaction. If another connection committed after the read, SQLite returns SQLITE_BUSY_SNAPSHOT (517); if it still holds the write lock, it returns SQLITE_BUSY (5). Both come back at once, and SQLite never calls the busy handler on a read-to-write upgrade, so the 15 s `busy_timeout` never applies. This is a refusal, not a hold longer than the timeout. `_retry_owned_contention` would retry 517, but only for a transaction the repository owns, and here `create_folder` ran in the caller's transaction (`cursor=` passed).
- *Fix (product).* `initialize_agent_lessons_folder` opens its transaction with `db.transaction(immediate=True)`. The write lock is taken before the first read and waits on the busy timeout, which is the repo's existing rule for read-then-write transactions (`TransactionContextManager` docstring; TASK-34000.28's `BEGIN IMMEDIATE` in the folder repository). Both callers (the local seed on backend change, and `NotesOrganizationSyncService.initialize_agent_lessons_seed`) get it. The recovery's own write transaction is a few milliseconds, so the wait is short.
- *Red/green, with forced contention.* Two new tests in `Tests/Notes/test_agent_lessons_seed.py`:
  - `test_seed_waits_for_a_writer_another_connection_holds`: another connection holds `BEGIN IMMEDIATE` for 1 s. Base: `AssertionError: FolderConflictError('Folder changed during mutation.')`, SQLite code 5 (SQLITE_BUSY) while the holder was alive.
  - `test_seed_survives_a_write_committed_between_its_read_and_its_write`: another connection commits a `buddy_profiles` row between the seed's read and its write, the captured order. Base: the same assertion, code 517.
  - With the fix, both pass. The seed file ran 16 passed.
  - A 6 s env-gated hold inside `publish_review` did not overlap the backend switch in unloaded runs (2/2 passed at base), so it is not used as evidence. It was reverted.
- *Soak.* All on fixed code, serial per lane, with the step flags, all passing:
  - 16 busy loops + 3 lanes: 30/30 at load 18-38.
  - 24 busy loops + 3 plain lanes x 15: 45/45 at 1-min load 38-62.
  - 2 probe lanes x 10 at the same time: 20/20, with no `FolderConflictError` and no folder-mutation error logged. The probe's timings of the seed's `BEGIN IMMEDIATE` (2-64 ms) cannot be attributed to the lock, since they also include connection setup and scheduling; the forced-hold tests above are the proof that the seed now waits.
  - Note: a second startup caller, `_deferred_wire_notes_sync_services` (`app_service_wiring.py:1988`), runs the same seed, and the shared fix covers it.
- *Whole "Run the recovered suites" step* at bc8ba15f47, serial with the CI flags (`--storage-lease-guard --timeout=180 --tb=short`) and full output kept: 2006 passed, 1 skipped, 1 deselected in 1048.14 s, load 46 falling to 11. This test took 11.84 s.
- *Covering tests* (`-n 4`): the seed file plus the Sync_Interop/Agents/Notes files that import `agent_lessons` ran 156 passed, 2 failed. The 2 (`test_notes_organization_app_wiring.py::test_real_per_user_notes_owner_rolls_back_mutation_and_intent_together` and `test_agent_runtime_preparation.py::test_agent_lessons_guidance_recomputes_from_each_effective_schema_set`) fail with `RecoveryRequired: raw_source_selection_changed`, the known local trap, and fail the same way with base `agent_lessons.py`.
- *Lane.* `Tests/Notes/test_agent_lessons_seed.py` now runs in the "Dreams tracking tests and unlaned riders" job (derived-artifacts.yml), after the three Workspaces riders and before `Tests/Tool_Packs`. It is sandboxed, and the job's pinned order keeps Tool_Packs last because Tool_Packs' bootstrap_profile nodes would break config reads for any test after it. `DREAMS_TRACKING_RIDERS` in `Tests/CI/test_ci_queue_pressure_contract.py` pins it. The job runs no `--storage-lease-guard` and no marker filter; its flags are serial `--timeout=180 --tb=short`. Cost: 16 tests, 12.5-12.8 s serially at load 11, so about 33 s at CI's 2.6x factor. Over 15 PR runs to 2026-11-23 02:31Z the job took median 8.4 / max 9.65 min of its 20, so about 10 min of headroom remains. Local step run and Tests/CI results: the whole step, serial with its flags, ran 753 passed in 429.17 s at load about 10, and this file passed 16/16 in place. `Tests/CI` at `-n 4` ran 1823 passed, 2 skipped.
- Files: `tldw_chatbook/Notes/agent_lessons.py` (immediate transaction + comment), `Tests/Notes/test_agent_lessons_seed.py` (two tests), `Tests/Performance/test_console_managed_received_acceptance.py` (`settle()` takes a label, and the stop-read wait fails as `managed_received_emergency_stop_read_timeout`, since `refresh()` logs a failed read only at DEBUG), `backlog/docs/lessons-testing-evidence.md` (one new entry, and the first entry corrected).
<!-- SECTION:NOTES:END -->
