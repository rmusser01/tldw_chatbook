---
id: TASK-42471
title: "The Console send diagnostic logs most trace refusal codes as ValueError/validation because its category set knows 15 of them"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
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
- [ ] #1 Every coded `ValueError` the two trace modules raise is either logged under its own code by the send diagnostic, or listed in the notes with the reason it stays `validation`
- [ ] #2 A test fails when a trace module raises a coded `ValueError` the diagnostic does not know, so the set cannot drift again; it is shown red with one code removed from the set
- [ ] #3 A Send refused at trace reservation logs both the refusal (`TraceReservationRefused`) and its code; a test shows it and fails on the current code
- [ ] #4 No free error text reaches the diagnostic, and the existing send-diagnostic tests pass unedited
<!-- AC:END -->
