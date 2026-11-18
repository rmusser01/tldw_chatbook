---
id: TASK-41517
title: "test_production_settings_inspects_private_and_deleted_metadata[size0] fails locally on base on a 5 s provenance wait"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-18 08:36'
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- backup
priority: medium
references:
- Tests/Backup_Recovery
dependencies:
- TASK-41122
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the pc stream (round 15). The `[size0]` parameter fails on base locally on a fixed 5 s provenance wait, so a local run of the Personal Context custody tests is red before any change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The provenance wait is a condition wait with a bounded deadline, and the node passes 10 of 10 locally at `-n 4`
- [x] #2 The notes record the load
<!-- AC:END -->

## Implementation Plan

1. Find which `until` (5 s, `pilot.pause(0.01)` poll) in the node runs out under load and what it waits for (worker-thread provenance read, a lost press, or a recompose).
2. Make the wait a bounded condition wait sized for loaded runs that names the unmet condition, polling without starving the reader thread; keep the shared helper's contract for its other callers.
3. Red before under a forced delay in the provenance read (reverted); green after; 10 of 10 at `-n 4` (temporary `rep` parametrize, since the node is `@private_profile_test`) with load recorded.

## Implementation Notes

Test-only. The shared provenance `until` now has a 30 s wall-clock deadline and fails naming the wait; the node's text waits also report what the panel showed.

**Cause: the test's 5 s budget, against a product lease that drops slow reads.**
- `until` (`Tests/UI/test_personal_context_provenance.py`) wrapped a `pilot.pause(0.01)` poll in `asyncio.timeout(5)` and failed with a bare `TimeoutError`.
- `PersonalContextProvenanceDetails._read` calls `settings_provenance` on a thread. A probe (temporary, reverted) timed 8 reads at load ~15: 0.25-0.65 s each, all of it in `settings_provenance` (the service loader took ~0).
- `_finish` drops a result whose read outlived the 2 s lease (`_LEASE_SECONDS`, counted from the read's start) and shows "Provenance unavailable."; the 1 s tick then reads again. Reads slow down under load, so at load ~85 the 5 s bound ran out before a read landed.
- The plan's lead that the `pilot.pause` poll starves the reader thread did not hold: with an `asyncio.sleep` poll the reads took the same time (mean 0.54 s against 0.51 s). The poll stays `pilot.pause`.

**Change.** `until(pilot, predicate, *, state=None, seconds=30.0)` polls to a wall-clock deadline and raises `AssertionError("<file>:<line of the predicate> still false after 30s; showing <state()>")`. It is shared by the provenance, proposal-review and Settings Personal Context tests, which all poll this widget's reads. The node passes `state=lambda: rendered(detail)` to its four text waits.

**Red/green.** A temporary control (reverted) made the first four reads sleep 2.1 s, so each outlived the lease, and later reads ran normally. The old test failed 2 of 2 (both sizes) at `test_settings_personal_context.py:2084` with a bare `TimeoutError`; the new test passed 2 of 2 (load 17-20). A second control slowed every read by 1.5 s: old and new both failed, the new one with ":2086 still false after 30s; showing '▼ Recorded provenance\nProvenance unavailable.'". No wait can pass while every read takes 2 s or more; that is the product limit below.

**Runs.**
- `[size0]` 10 of 10 at `-n 4`, through a temporary `r16` parametrize on the test (the node is `@private_profile_test`, so a repeat plugin cannot reach its child), since removed. Load 19.4 before, 20.4 after.
- `test_personal_context_provenance.py`, `test_personal_context_proposal_review.py` and `test_settings_personal_context.py`: 132 passed at `-n 4` (load 14-17).

**Proposed follow-up (product).** The lease counts the read's own duration, so on a host where one `settings_provenance` read takes 2 s or more, Settings shows "Provenance unavailable." for as long as that lasts.

**Fix round 1 (review r16wt-rev1).** `until` took its location from `predicate.__code__`, so the three callers that pass `reader.entered.is_set` would have reported `threading.py:529`. It now names its own call site (`sys._getframe(1)`). Check: a temporary test that awaited `until(pilot, Event().is_set, seconds=0.2)` got "test_personal_context_provenance.py:330 still false after 0s", the line of that call; the test was removed. The timeout now prints with `:g`, so 0.2 no longer reads as 0.

**Files.** `Tests/UI/test_personal_context_provenance.py`, `Tests/UI/test_settings_personal_context.py`.
