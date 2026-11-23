---
id: TASK-42468
title: "The character presentation cadence test's cancelled case is red on dev: a cancelled refresh lets the old display fingerprint pass as current"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 20:49
labels:
- follow-up
- console
- characters
- tests
- dev-red
priority: medium
references:
- Tests/Backup_Recovery/test_character_presentation_cadence.py
- tldw_chatbook/UI/Console_Modules/character_context.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23): TASK-42426's covering runs found it red 1 of 3 parameters, on base and branch alike, twice each. Re-run on the round-22 stack before its rebase (142e821116) at `-n 4`: still red. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches). Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**The failure.** `Tests/Backup_Recovery/test_character_presentation_cadence.py::test_direct_refresh_outcome_cannot_certify_an_old_display[cancelled]` fails at line 472 with `AssertionError: old fingerprint certified a refresh that never successfully published`. On `ConsoleCharacterContextController` (`UI/Console_Modules/character_context.py`), `_successful_refresh_generation` is None and `_generation` is 2, yet the old fingerprint is still accepted. The test's other two parameters pass. The file last changed in 12c803b093 (2026-10-16).

**Why it matters.** If this is a product defect, a cancelled character refresh lets a display that was never republished be treated as current, so the Console could show stale character context as fresh. It may instead be a test whose expectation drifted; that is not known. The file is in no PR lane (not in `scripts/ui_pr_gate_census.txt` or a `derived-artifacts.yml` step), so nothing reports it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is recorded: test drift after a named product commit, or a product defect in how a cancelled refresh certifies the display
- [x] #2 A defect is fixed in product code (a cancelled refresh never certifies the old fingerprint), with the `[cancelled]` case as its red-first test; drift is repaired in the test with its assertion kept
- [x] #3 All three parameters pass, alone and in a `-n 4` run of the file
- [x] #4 The file runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the `[cancelled]` failure at base: what state the cancelled direct refresh leaves, and why the next display check keeps the old key (95ca4bf47b's phase guard).
2. Decide drift vs defect from what the user would see (a panel left on "Refreshing..." whose display check never runs again).
3. If a defect: fix it in `ConsoleCharacterContextController.refresh()` so a cancelled refresh that still owns its generation leaves the panel idle and drops the old fingerprint for the next check to reload; keep the test's assertion and add one for the stuck phase; red at base, green after.
4. Run all three parameters alone and at -n 4; lane the file in an existing non-UI PR step (the unlaned-riders job) with its CI pin, timed serially with that step's flags.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause (AC#1): a product defect, from 95ca4bf47b (2026-11-14).** That commit made `refresh_presentation_if_scope_changed` return False at once unless the controller's phase is IDLE, so a display check never runs beside an operation. `refresh()` sets the phase to REFRESHING in `_begin` and nothing set it back when the refresh was cancelled. A cancelled refresh therefore left the panel on "Refreshing local character chats…" with the old fingerprint and the old presentation key in place, and every later display check returned "unchanged" without reading anything, until some other operation (a search, a Retry) began. Before 95ca4bf47b the next check ran anyway and its own refresh republished, so the test passed. The test's assertion (`_presentation_scope_key is None or` a successful refresh at the current generation) caught the old key surviving; the stuck phase is the user-visible half.

**Fix (AC#2).** `ConsoleCharacterContextController.refresh()` (`UI/Console_Modules/character_context.py`) now catches `CancelledError` around everything after `_begin`. If the refresh still owns the generation and the phase is still REFRESHING, it calls `invalidate_scope(_supersede=False)`: the panel goes IDLE, `loading` clears, the old fingerprint is dropped, and the next display check reloads and certifies a fresh publication. It then re-raises. A newer operation's state is left alone. The fix is in `refresh()`, so it also covers a display check cancelled during its own inner refresh (`_supersede=False`); a trace during the covering run saw the handler fire there once (generation 0) as well as in the red test.

**Red/green.**
- At base: `[cancelled]` fails at the existing assertion ("old fingerprint certified a refresh that never successfully published"; key from generation 1, `_successful_refresh_generation` None, generation 2).
- The test gained one assertion: after the display check, the phase is IDLE and not loading. With a temporary key-only fix (the phase guard clearing `_presentation_scope_key`, reverted), the old assertion passed and the new one failed ("a refresh that never published left the panel refreshing"), so it pins the stuck panel the key check alone misses.
- With the fix: all three parameters pass.

**Runs (AC#3).** Each parameter alone (serial): 3 of 3 passed. The file at `-n 4`: 47 passed (load 46). The 18 files that cover the controller at `-n 4` (load about 20): 430 passed, 12 failed; all 12 were checked against base. 10 fail at base too (`test_character_refresh_finite_batch` x2, `test_character_scope_check_one_helper` x2, `test_character_display_current_callers::test_actual_full_reuses_successful_character_display`, `test_console_character_context_geometry` x5). The other two pass at base in that run: `test_console_character_avatar.py::test_character_geometry_epoch_consumes_only_one_followup` failed 3 of 10 serial runs at base (`(22, 12) == (23, 12)`), and `test_actual_character_resume_declines_custom_and_refuses_queued_source_change` passed on rerun; a trace showed the new handler never fired in either. A rerun of the four mounted character files plus this file (220 tests) recorded every firing of the handler (two) and gave 219 passed, the one base red.

**Lane (AC#4).** The file is `bootstrap_profile`, so it joins the unlaned-riders job ("Dreams tracking tests and unlaned riders", `.github/workflows/derived-artifacts.yml`) after `Tests/Tool_Packs`, never before a sandboxed target; `Tests/CI/test_ci_queue_pressure_contract.py`'s `DREAMS_TRACKING_RIDERS` pin lists it last. Serially with the step's flags (`--timeout=180`): 47 passed in 63 s and 72 s (load 14-18). The whole step through `r19-step.sh` (`-p no:xdist --timeout=180`): 801 passed in 780 s at load 15-20. The CI contract files that pin the job passed (812 at `-n 4`).

**Files.** `tldw_chatbook/UI/Console_Modules/character_context.py`, `Tests/Backup_Recovery/test_character_presentation_cadence.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, and `Docs/User_Guide/console/sessions-tabs-workspaces.md` (an interrupted reload clears the "Refreshing" line).
<!-- SECTION:NOTES:END -->
