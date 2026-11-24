---
id: TASK-42471
title: "The Console send diagnostic logs most trace refusal codes as ValueError/validation because its category set knows 15 of them"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 05:28
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
**AC amendment (2026-11-23, round-24 review).** AC#1 and AC#2 now cover every `Chat/console_trace_*.py` module, and the codes the typed refusals carry themselves. Why: review found that `_rolled_back_refusal`'s own fallback code, `trace_reservation_refused`, was missing from the set, because the scan matched only bare `ValueError` calls. The coordinator also chose to burn down the other trace modules' codes here, rather than file them as a follow-up; they reach the reservation through the same `TraceReservationRefused` wrapper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every coded `ValueError`, `TraceReservationRefused` or `TraceSurfaceChangeRefused` that a `Chat/console_trace_*.py` module raises is either logged under its own code by the send diagnostic, or listed in the notes with the reason it stays `validation`
- [x] #2 A test fails when any trace module raises a coded `ValueError` or typed refusal (including a `code or "<literal>"` fallback) the diagnostic does not know, so the set cannot drift again; it is shown red with one code removed from the set
- [x] #3 A Send refused at trace reservation logs both the refusal (`TraceReservationRefused`) and its code; a test shows it and fails on the current code
- [x] #4 No free error text reaches the diagnostic, and the existing send-diagnostic tests pass unedited
<!-- AC:END -->

## Implementation Plan

1. Keep the category set exact and code-owned (the module's rule: no token from free error text), and extend `_TRACE_FAILURE_CATEGORIES` to every literal `ValueError("<code>")` in `Chat/console_trace_service.py` and `Chat/console_trace_runtime.py`, plus the literal field names `_required_string` raises.
2. Drift guard: an AST scan of the two modules in the fast-lane rider `Tests/Chat/test_console_trace_missing_revision_diagnostic.py` fails on a literal code missing from the set, and on any non-literal `ValueError(...)` outside `_required_string`. Show it red with one code removed.
3. Cause walk: stop at the first exception that names a trace refusal: the typed refusals `TraceReservationRefused` / `TraceSurfaceChangeRefused` (always), or a plain `ValueError` whose code is in the set. Log its type and code, so a reservation refusal reads `exception_type=TraceReservationRefused error_category=<code>`.
4. Red/green: a real refused reservation through the gateway's `_reserve_trace_call`, inside a send-diagnostic scope, logs both; red on base (it logs `ValueError`).
5. Run the existing send-diagnostic tests unedited, plus the covering trace tests.
6. (Review, AC#1/#2 amended.) Widen the scan to every `Chat/console_trace_*.py`, to `builtins.ValueError`, and to the typed refusals' constructors, including `code or "<literal>"` and module-constant arguments. List every non-literal raise site it cannot check, with a reason. Add the new codes, including `trace_reservation_refused`, and set the floor to the measured count.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The send diagnostic now names every code-shaped refusal that a `Chat/console_trace_*.py` module raises: 210 codes from 22 modules (209 when written; dev's #3670 added `replacement_chain_link_inactive`). A reservation refusal logs as `exception_type=TraceReservationRefused error_category=<code>`.

**Approach.**
- *The set.* `_TRACE_FAILURE_CATEGORIES` (`Chat/console_send_diagnostics.py`) stays an exact list written by hand, because the module's rule is never to make a token from error text. It holds:
  - every code-shaped literal (`[a-z][a-z0-9_]*`) passed to `ValueError`, `builtins.ValueError`, `TraceReservationRefused` or `TraceSurfaceChangeRefused`, including the `code or "<literal>"` fallback (`trace_reservation_refused`) and module-constant arguments (`custom_pii_invalid_batch`);
  - the literal field names passed to `_required_string` (`provider_name`, `model_name`).
- *The drift guard* is `test_every_coded_trace_refusal_has_a_send_diagnostic_category`, with `_trace_refusal_codes()` in the same file. It reads every `console_trace_*.py` file's syntax tree (AST) and:
  - fails on a code the set lacks;
  - fails on any raise whose argument is neither a literal nor the exact argument expression its `_UNCODED_TRACE_VALUE_ERRORS` row pins (`field_name`, `name`, `capture_error`), so a new `ValueError(some_var)`, even in a listed function, fails;
  - fails when a row is no longer used by any raise (the used rows must equal the table);
  - counts the raises that carry no code against pinned totals in `_UNCODED_TRACE_RAISE_COUNTS`: 9 formatted messages, 45 prose sites (42 distinct messages), 2 argument-less raises. A new one fails until it is reviewed;
  - accepts the `or` form only as `<Name> or "<literal>"`;
  - fails below a floor of 210 codes, the count measured on 2026-11-24, so a scan that finds nothing cannot pass.

  I chose a test-time derivation over a runtime one: parsing the modules at runtime would cost every first failure, and matching a regex would make tokens from text.
- *The cause walk.* It stops at the first exception that names a trace refusal: a `TraceReservationRefused` or `TraceSurfaceChangeRefused` (always), or a plain `ValueError` whose argument is a known code. Before, it went on to the innermost exception. A typed refusal's argument is logged only when it is a known code; any other argument logs `validation` (AC#4). The code is now held in `trace_code`, so the name `code` is used only for the sqlite error code (review M3).
- *Side effect, intended.* A coded `ValueError` raised `from` another error now logs its code instead of the inner error's type. For example, `surface_value_unavailable` raised from a `json.JSONDecodeError` used to log `JSONDecodeError`/`validation`. No trace module raises a code inside a `sqlite3` handler, so no `sqlite_code` is lost.
- *Import.* `console_trace_errors` is a dependency-free leaf that is already loaded at boot (`ui_ready_modules.txt`), so the import adds no module to the first-send warm-up.

**What stays `validation`, and why (AC#1).**
- *Non-literal raise sites*, listed in `_UNCODED_TRACE_VALUE_ERRORS`:
  - `console_trace_service._required_string`: its callers' literal names are scanned as codes.
  - `console_trace_maintenance._validate_utc_timestamp`, and `console_trace_repository._validate_token`, `_nonempty`, `_json_object` and `_canonical_json_value`: each raises the name of a malformed field of an internal record. Some callers pass that name through a variable (for example a loop at `console_trace_repository.py:566`), so no scan can prove it.
  - `console_trace_redaction.__init__` and `redact_pii_value`, and `console_trace_regex_worker.__post_init__`: each raises a constructor bound's name taken from a literal `(value, name)` table.
  - `console_trace_projection._decode_legacy_row`: a legacy capture row's field name, from a read path, not a Send.
- *Formatted messages* (f-strings, 9 sites in `console_trace_models` validators): free text, never a code.
- *Two bare `raise ValueError`* (`console_trace_regex_worker.py:406` and `:426`): no argument, so no code.
- *42 prose literals.* A message with spaces cannot be a log token: the persistent schema's token rule would write it as `invalid`. They are: `available provider shadow is content-bearing`; `checkpoint unavailable`; `content_kind must not be empty`; `credential_filter_version must not be empty`; `duplicate flags`; `invalid revision`; `masked key collision`; `omitted_count must be a non-negative integer`; `pii_ruleset_revision_id is required when PII redaction is enabled`; the eight `promoted call/aggregate ...` messages; `reason must be a TraceOmissionReason`; `replacement range must be ordered`; the 16 `report ...` messages; `route must be a ConsoleRequestRoute`; `sanitized key collision`; `settlement timestamps`; `source must be a TraceProvenanceSource`; `trace lineage cycle`; `trace lineage unavailable`; `transform must be a TraceTransformKind`.

**Red/green.**
- AC#2, the scan:
  - First version (two modules): red at base, `assert ['available_b...y_value', ...] == []`, 73 codes missing. Red with only `surface_replacement_checkpoint_unavailable` removed from the set.
  - Widened version: red against the first version's 88-code set, 121 codes missing, among them `trace_reservation_refused` and `custom_pii_invalid_batch`. Red with only `trace_reservation_refused` removed: `assert ['trace_reservation_refused'] == []`.
  - Green with the full set. Each of the ten `_UNCODED_TRACE_VALUE_ERRORS` rows is used: emptying the table lists 17 unchecked sites.
  - Second review (each is a temporary edit, reverted, and red):
    - In the test tables: an extra row no raise site uses fails the used-rows check; a row repinned to `other_name` fails `_nonempty`'s raise site as unchecked; a count pinned to `bare: 1` fails the counts check.
    - In trace source: a new `ValueError(f"temp {x}")` makes formatted 10 against 9; a bare `raise ValueError`, bare 3 against 2; `ValueError("a new prose message")`, prose 46 against 45; `ValueError(a or b)` is unchecked (`console_trace_models:671`); and `raise ValueError(value)` inside `_nonempty`, a pinned function whose row is `field_name`, is unchecked (`console_trace_repository:322`).
- AC#3, `test_a_refused_reservation_logs_the_refusal_and_its_code`: a real refused reservation (`trace_revision_unavailable`) through `ConsoleProviderGateway._reserve_trace_call`, inside a send-diagnostic scope. Red at base (`assert 'ValueError' == 'TraceReservationRefused'`), green now. It also asserts that neither the draft text nor the opaque revision id reaches the recorded fields.
- Live check, using TASK-42412's forced reproduction (a temporary 8 s sleep in `_read_default_stop`, the stop-path wait removed, both reverted): the child's line read `error_category=surface_replacement_checkpoint_unavailable exception_type=TraceReservationRefused phase=trace_reservation`. Base logged `ValueError`/`validation`.

**Runs.**
- *Soak (`-n 4`).* The changed file + `Tests/Chat/test_console_send_diagnostics.py` + `Tests/Chat/test_console_trace_surface_refusal_diagnostics.py`, 20 times each round:
  - First version: 20/20 runs, 400/400 tests, load 15-22.
  - After review: 20/20 runs, 400/400 tests, load 41-95.
  - After the second review: the same files with `Tests/Chat/test_console_project_instruction_traces.py` and `Tests/Chat/test_console_thinking_capture.py`, 20/20 rounds, 2320/2320 tests, load 8-16.
  - The last two files are unedited (AC#4).
- *Covering Chat trace and diagnostic files:* 341 passed (first version).
- *Fast lane.* "Run fast PR contract", serial with its flags, through the fixed `r19-step.sh`: 4739 passed, 4 skipped. This file is a rider of that step.
- *Known local failures, red at base with the same command* (base `console_send_diagnostics.py` swapped in). Neither file is in a PR lane.
  - `Tests/UI/test_console_send_diagnostic_flow.py`: plain, it hits `RecoveryRequired`. Under a temporary `@private_profile_test` it fails with `phase=ui_action status=awaiting_review` and `assert 0 == 1`, in both trees.
  - `Tests/Performance/test_console_native_pause_probe.py`: `Tests.Performance.console_native_pause_seams` was never committed.

**Files.** `tldw_chatbook/Chat/console_send_diagnostics.py`; `Tests/Chat/test_console_trace_missing_revision_diagnostic.py` (two tests plus the scan helper). That file is already a PR Fast Lane rider, so no lane edit is needed.
<!-- SECTION:NOTES:END -->
