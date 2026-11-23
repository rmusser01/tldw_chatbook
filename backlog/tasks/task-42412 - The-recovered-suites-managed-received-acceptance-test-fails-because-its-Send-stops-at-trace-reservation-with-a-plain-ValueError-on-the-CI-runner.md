---
id: TASK-42412
title: "The recovered-suites managed received acceptance test fails because its Send stops at trace reservation with a plain ValueError on the CI runner"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 21:17
updated_date: 2026-11-23 04:28
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
  - Run 2, alongside the first soak (load 24-46): 2 failed, 2004 passed. The two failures were this test and `Tests/Library/test_library_ingest_runner.py::test_real_pool_worker_exit_is_reported_for_owning_generation`. This test failed in under 12.2 s (it is missing from the 5 slowest durations, whose cutoff was 12.24 s), so this is not the 120 s `settle(entered)` hang fixed here. The step script kept only the last 15 lines, so the failure text was lost, and the run did not recur in runs 3 and 4 or in 20 soak runs. Cause unknown.

Out of scope here (proposed as follow-ups):
- Add the trace refusal codes the send diagnostic does not know (`surface_replacement_checkpoint_unavailable` and others) to `_TRACE_FAILURE_CATEGORIES`.
- Check whether an agent's next call after a failed first call should be refused by trace reservation.

Files: `Tests/Performance/test_console_managed_received_acceptance.py` (one wait, 8 lines); `backlog/docs/lessons-testing-evidence.md` (one entry).
<!-- SECTION:NOTES:END -->
