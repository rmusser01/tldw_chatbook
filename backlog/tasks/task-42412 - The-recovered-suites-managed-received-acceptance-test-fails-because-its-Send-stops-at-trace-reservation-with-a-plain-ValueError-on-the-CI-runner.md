---
id: TASK-42412
title: "The recovered-suites managed received acceptance test fails because its Send stops at trace reservation with a plain ValueError on the CI runner"
status: To Do
assignee: []
created_date: 2026-11-22 21:17
updated_date: 2026-11-22 21:17
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

**Why.** `Tests/Performance/test_console_managed_received_acceptance.py::test_received_managed_demand_uses_one_catalog_and_real_cold_initializer` fails the required "PR lane (recovered suites)" job with `managed_received_prerequisite_timeout`, for reasons outside the PR under test. It failed in at least six PR runs between 2026-11-21 21:59Z and 2026-11-22 17:42Z: 44954347282, 44974735998, 44974742439, 44977408277, 44978286545 and 45050980929. Its body runs in its own child process, and it passes alone locally (16.7 s, TASK-33620.15.61's notes). No task owned it; TASK-41564's notes only mention it.

**What the logs show.** In the five of those runs whose logs were read (44954347282, 44974735998, 44974742439, 44977408277, 44978286545), the signature is the same:
- the Send's `trace_reservation` stage fails after about 6.2 s (`duration_ms=6241` in 44978286545) with `exception_type=ValueError error_category=validation`;
- the child then fails at script line 770, `await settle(entered.is_set)` (test file line 778 at this base), the wait for the controlled preparation read after the Send;
- the call takes about 133-135 s, which is the 120 s `CUSTODY_HANG_BOUND` plus setup.

So raising the bound would not help. 767c15cffb (2026-11-19) already turned `settle()`'s 10 s deadline into that hang bound, and every listed run is later than it. The Send never reaches the read the test controls.

**Where the ValueError comes from.** `console_send_diagnostics.py:140-168` walks the cause chain and records the innermost exception's type. A `ValueError` whose single argument is a known trace-failure code would be recorded under that code, and a `TraceReservationRefused` or `TraceSurfaceChangeRefused` would be recorded under its own class name (`console_trace_errors.py:39-56`). So the root is a plain `ValueError` with some other message, raised inside `_reserve_trace_call` (`Chat/console_provider_gateway.py:3582-3671`), whose broad `except Exception` turns it into `TraceCallPersistenceError(reservation_status="unknown")`. The 6.2 s before it suggests a wait inside the reservation (a lock or a cold initializer) that ends in a refusal on the slower runner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The plain ValueError raised during this test's trace reservation is identified: its message and the line that raises it are recorded in the task notes, from a CI log or a local reproduction under a slowed reservation
- [ ] #2 The root cause is fixed in product code, or, if the refusal is correct behaviour for that state, the test sets up the state the reservation needs; the notes say which and why
- [ ] #3 A reproduction that fails the test before the change (a temporary slowdown or fault injection, reverted) passes after it
- [ ] #4 The test passes in the recovered-suites job in at least three consecutive PR runs after the change, or in a local run of the job's step under parallel load
<!-- AC:END -->
