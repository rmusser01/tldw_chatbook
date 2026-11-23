---
id: TASK-42471
title: "The Console send diagnostic logs most trace refusal codes as ValueError/validation because its category set knows 15 of them"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 20:23
labels:
- follow-up
- console
- diagnostics
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_send_diagnostics.py
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_runtime.py
- Tests/Chat/test_console_trace_surface_refusal_diagnostics.py
- Tests/UI/test_console_send_diagnostic_flow.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42412 and its review. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** When a Send fails, `Chat/console_send_diagnostics.py:139-168` walks the error's `__cause__` / `__context__` chain to the innermost exception and logs its type and a category. A plain `ValueError` whose single string argument is in `_TRACE_FAILURE_CATEGORIES` (lines 21-37) is logged under that code; any other `ValueError` is logged as `validation`. The set has 15 codes. `Chat/console_trace_service.py` and `Chat/console_trace_runtime.py` raise 86 distinct coded `ValueError("<code>")`s at the stack top (counted with `grep -ohE 'ValueError\("[a-z_]+'` over the two files). So most refusals log only `exception_type=ValueError error_category=validation`. Codes missing from the set include `surface_replacement_checkpoint_unavailable`, `trace_policy_mismatch`, `surface_delta_alignment`, `surface_replacement_target_inactive`, `trace_tool_chain_source_mismatch`, `trace_child_owner`, `trace_child_segment_unavailable`, `trace_current_transform_owner`, `live_native_custody`, `live_native_owner` and the `trace_recovery_*` codes.

**Evidence.** TASK-42412's CI failure logged exactly `ValueError`/`validation` at `phase=trace_reservation`. Its code, `surface_replacement_checkpoint_unavailable` (raised at `Chat/console_trace_service.py:2927`), was found only with a temporary log; a CI log carrying the code would have named it on the first day.

**A second gap.** `_rolled_back_refusal` (`Chat/console_trace_runtime.py:84`) re-raises a coded `ValueError` as `TraceReservationRefused(code)` from it. The cause walk records the innermost exception, so the diagnostic never shows that the Send was refused at reservation.

The set is code-owned on purpose (its comment, line 20: never derive a token from arbitrary error text), so the fix keeps codes exact and logs no free text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every coded `ValueError` the two trace modules raise is either logged under its own code by the send diagnostic, or listed in the notes with the reason it stays `validation`
- [x] #2 A test fails when a trace module raises a coded `ValueError` the diagnostic does not know, so the set cannot drift again; it is shown red with one code removed from the set
- [x] #3 A Send refused at trace reservation logs both the refusal (`TraceReservationRefused`) and its code; a test shows it and fails on the current code
- [x] #4 No free error text reaches the diagnostic, and the existing send-diagnostic tests pass unedited
<!-- AC:END -->

## Implementation Plan

1. Keep the category set exact and code-owned (the module's rule: no token from free error text), and extend `_TRACE_FAILURE_CATEGORIES` to every literal `ValueError("<code>")` in `Chat/console_trace_service.py` and `Chat/console_trace_runtime.py`, plus the literal field names `_required_string` raises.
2. Drift guard: an AST scan of the two modules in the fast-lane rider `Tests/Chat/test_console_trace_missing_revision_diagnostic.py` fails on a literal code missing from the set, and on any non-literal `ValueError(...)` outside `_required_string`. Show it red with one code removed.
3. Cause walk: stop at the first exception that names a trace refusal: the typed refusals `TraceReservationRefused` / `TraceSurfaceChangeRefused` (always), or a plain `ValueError` whose code is in the set. Log its type and code, so a reservation refusal reads `exception_type=TraceReservationRefused error_category=<code>`.
4. Red/green: a real refused reservation through the gateway's `_reserve_trace_call`, inside a send-diagnostic scope, logs both; red on base (it logs `ValueError`).
5. Run the existing send-diagnostic tests unedited, plus the covering trace tests.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The send diagnostic now names every coded refusal the two trace modules raise, and a reservation refusal logs as `exception_type=TraceReservationRefused error_category=<code>`.

**Approach.**
- *The set.* `_TRACE_FAILURE_CATEGORIES` (`Chat/console_send_diagnostics.py`) stays an exact, code-owned list, because the module's rule is that no token is ever derived from error text. It now holds all 86 literal `ValueError("<code>")` codes in `Chat/console_trace_service.py` and `Chat/console_trace_runtime.py`, plus `provider_name` and `model_name` (88 in all). Those two are the literal field names that `_required_string` raises as `ValueError(field_name)`. No code stays `validation` (AC#1).
- *The drift guard* is `test_every_coded_trace_refusal_has_a_send_diagnostic_category`. It AST-scans both modules for every `ValueError(<str literal>)` and every `_required_string(..., "<field>")`, and fails on a code the set lacks. It also fails on any `ValueError(...)` whose argument is not a literal (outside `_required_string`), since nothing could check that one. A floor (`> 80` codes) keeps an empty scan from passing. I chose a test-time derivation over a runtime one: a runtime parse of the 6.7k-line module would cost every first failure, and a regex match would derive tokens from text.
- *The cause walk.* It stops at the first exception that names a trace refusal: a `TraceReservationRefused` or `TraceSurfaceChangeRefused` (always), or a plain `ValueError` whose argument is a known code. Before, it went on to the innermost exception. A typed refusal's argument is logged only when it is a known code; any other argument logs `validation` (AC#4).
- *Side effect, intended.* A coded `ValueError` raised `from` another error now logs its code instead of the inner error's type. For example, `surface_value_unavailable` or `request_header_component_invalid` raised from a `json.JSONDecodeError` used to log `JSONDecodeError`/`validation`. Neither module raises a code inside a `sqlite3` handler, so no `sqlite_code` is lost.
- *Import.* `console_trace_errors` is a dependency-free leaf that is already boot-resident (`ui_ready_modules.txt`), so the new import adds no module to the first-send warm-up.

**Red/green.**
- AC#2, the parity test:
  - Red at base: `assert ['available_b...y_value', ...] == []` (73 codes missing).
  - Red with only `surface_replacement_checkpoint_unavailable` removed from the set: `assert ['surface_rep..._unavailable'] == []`.
  - Green with the full set.
- AC#3, `test_a_refused_reservation_logs_the_refusal_and_its_code`: a real refused reservation (`trace_revision_unavailable`) through `ConsoleProviderGateway._reserve_trace_call`, inside a send-diagnostic scope. It is red at base (`assert 'ValueError' == 'TraceReservationRefused'`) and green now. It also asserts that neither the draft text nor the opaque revision id reaches the recorded fields.
- Live check, using TASK-42412's forced reproduction: a temporary 8 s sleep in `_read_default_stop`, with the test's stop-path wait removed. Both edits were reverted. The child's line read `error_category=surface_replacement_checkpoint_unavailable exception_type=TraceReservationRefused phase=trace_reservation`; base logged `ValueError`/`validation`.

**Runs (`-n 4`, load 15-33).**
- New file + `Tests/Chat/test_console_send_diagnostics.py` + `Tests/Chat/test_console_trace_surface_refusal_diagnostics.py`: soaked 20 times, 20/20 runs and 400/400 tests passed at load 15-22. The last two files are unedited (AC#4).
- Trace runtime/service, submit blocker, capture triggers, durable-commit diagnostics and the other `record_send_stage` callers' test files: 341 passed.
- Known local failures, all red at base with the same command (base `console_send_diagnostics.py` swapped in): `Tests/UI/test_console_send_diagnostic_flow.py` (plain: `RecoveryRequired`; under a temporary `@private_profile_test`: `phase=ui_action status=awaiting_review` and `assert 0 == 1`, in both trees) and `Tests/Performance/test_console_native_pause_probe.py` (`Tests.Performance.console_native_pause_seams` was never committed). Neither file is in a PR lane.

**Scope note.** Codes raised by the other `Chat/console_trace_*.py` modules (about 137 more, e.g. `console_trace_continuation`, `console_trace_final_values`, `console_trace_repository`) can also be wrapped into `TraceReservationRefused` at reservation. They now log `exception_type=TraceReservationRefused error_category=validation`. Adding them would be one more entry in the scan's module tuple plus their codes.

**Files.** `tldw_chatbook/Chat/console_send_diagnostics.py`; `Tests/Chat/test_console_trace_missing_revision_diagnostic.py` (two tests; this file is already a PR Fast Lane rider, so no lane edit is needed: 1.4 s and 0.1 s locally).
<!-- SECTION:NOTES:END -->
