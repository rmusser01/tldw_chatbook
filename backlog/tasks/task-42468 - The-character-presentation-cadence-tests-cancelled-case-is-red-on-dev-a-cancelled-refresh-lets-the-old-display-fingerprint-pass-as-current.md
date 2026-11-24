---
id: TASK-42468
title: "The character presentation cadence test's cancelled case is red on dev: a cancelled refresh lets the old display fingerprint pass as current"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 00:25
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
- [x] #5 A refresh that publishes nothing for any other reason (its owner or scope moved during the read, or it raised) while it still owns the panel leaves it idle, with its fingerprint dropped so the next display check reloads; a test red at base covers the stock route and the original loop
- [x] #6 A cancelled search leaves the panel idle with a Retry for its query instead of "Searching local character chats…"; a test red at base covers it, and the search's other exits that publish nothing take the same guard
- [x] #7 Each sibling path that does not stick is recorded in the notes with its evidence
- [x] #8 The touched test files pass 20 soak runs at `-n 4` under load
- [x] #9 Opening a character chat (OPENING) that is cancelled, or ends without publishing, while it still owns the panel leaves a state the next check or Retry recovers from; a test red at base, which confirms the phase while the operation is blocked, covers it (or the notes record why the path does not stick)
- [x] #10 Each repair operation (REPAIRING: preparing a repair, opening the unavailable detail) that is cancelled, or ends without publishing, while it still owns the panel shows its failure with a Retry and never claims success; a test red at base, which confirms the phase while blocked, covers each (or the notes record why the path does not stick)
- [x] #11 The character file passes 20 soak runs at `-n 4` under load after these changes
<!-- AC:END -->

AC #9-#11 added 2026-11-23 (round 24, coordinator request): OPENING and REPAIRING are the controller's other loading phases. The IDLE-only display check strands them the same way it stranded REFRESHING and SEARCHING, so they finish this defect class before review.

AC #5-#8 added 2026-11-23 (round 24, coordinator request): the stuck phase is the same defect class as the `[cancelled]` red. 95ca4bf47b's IDLE-only display check also strands a refresh that returns without publishing (owner or scope moved; the code's own comment says "its next check reloads") and a cancelled search, so the fix belongs with this task rather than in a follow-up.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the `[cancelled]` failure at base: what state the cancelled direct refresh leaves, and why the next display check keeps the old key (95ca4bf47b's phase guard).
2. Decide drift vs defect from what the user would see (a panel left on "Refreshing..." whose display check never runs again).
3. If a defect: fix it in `ConsoleCharacterContextController.refresh()` so a cancelled refresh that still owns its generation leaves the panel idle and drops the old fingerprint for the next check to reload; keep the test's assertion and add one for the stuck phase; red at base, green after.
4. Run all three parameters alone and at -n 4; lane the file in an existing non-UI PR step (the unlaned-riders job) with its CI pin, timed serially with that step's flags.
5. (AC #5-#8) Write red-at-base tests for a refresh that publishes nothing after its owner or scope moved (the stock route a run's display check takes for a changed open chat, and the original loop) and for a cancelled search.
6. Replace the refresh's cancellation handler with one guard the operations share, in the controller that owns the phase: on any exit, if the operation still owns the generation and its phase, settle the panel idle (a refresh drops its fingerprint; a search keeps its query and offers Retry).
7. Record the exits that do not stick with their evidence, soak the touched files 20 times at -n 4, and re-run the round's check list.
8. (AC #9-#11) For activation and the two repair operations: a test that blocks the operation (activation in its activation step, the repairs in their scope read), confirms the phase, then cancels it, red at base; route each through `_settle_unpublished` with the copy its own failure path already shows, keeping the fingerprint (a moved scope is still found by the next check); soak and re-run the check list.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause (AC#1): a product defect, from 95ca4bf47b (2026-11-14).** That commit made `refresh_presentation_if_scope_changed` return False at once unless the controller's phase is IDLE, so a display check never runs beside an operation. `refresh()` sets the phase to REFRESHING in `_begin` and nothing set it back when the refresh was cancelled. A cancelled refresh therefore left the panel on "Refreshing local character chats…" with the old fingerprint and the old presentation key in place, and every later display check returned "unchanged" without reading anything, until some other operation (a search, a Retry) began. Before 95ca4bf47b the next check ran anyway and its own refresh republished, so the test passed. The test's assertion (`_presentation_scope_key is None or` a successful refresh at the current generation) caught the old key surviving; the stuck phase is the user-visible half.

**Fix (AC#2).** First commit: `ConsoleCharacterContextController.refresh()` (`UI/Console_Modules/character_context.py`) caught `CancelledError` around everything after `_begin`; the sibling-path work below replaced that handler with the shared `_settle_unpublished` guard in a `finally`, which keeps this behaviour. If the refresh still owns the generation and the phase is still REFRESHING, it calls `invalidate_scope(_supersede=False)`: the panel goes IDLE, `loading` clears, the old fingerprint is dropped, and the next display check reloads and certifies a fresh publication. It then re-raises. A newer operation's state is left alone. The fix is in `refresh()`, so it also covers a display check cancelled during its own inner refresh (`_supersede=False`); a trace during the covering run saw the handler fire there once (generation 0) as well as in the red test.

**Red/green.**
- At base: `[cancelled]` fails at the existing assertion ("old fingerprint certified a refresh that never successfully published"; key from generation 1, `_successful_refresh_generation` None, generation 2).
- The test gained one assertion: after the display check, the phase is IDLE and not loading. With a temporary key-only fix (the phase guard clearing `_presentation_scope_key`, reverted), the old assertion passed and the new one failed ("a refresh that never published left the panel refreshing"), so it pins the stuck panel the key check alone misses.
- With the fix: all three parameters pass.

**Runs (AC#3).** Each parameter alone (serial): 3 of 3 passed. The file at `-n 4`: 47 passed (load 46). The 18 files that cover the controller at `-n 4` (load about 20): 430 passed, 12 failed; all 12 were checked against base. 10 fail at base too (`test_character_refresh_finite_batch` x2, `test_character_scope_check_one_helper` x2, `test_character_display_current_callers::test_actual_full_reuses_successful_character_display`, `test_console_character_context_geometry` x5). The other two pass at base in that run: `test_console_character_avatar.py::test_character_geometry_epoch_consumes_only_one_followup` failed 3 of 10 serial runs at base (`(22, 12) == (23, 12)`), and `test_actual_character_resume_declines_custom_and_refuses_queued_source_change` passed on rerun; a trace showed the new handler never fired in either. A rerun of the four mounted character files plus this file (220 tests) recorded every firing of the handler (two) and gave 219 passed, the one base red.

**Lane (AC#4).** The file is `bootstrap_profile`, so it joins the unlaned-riders job ("Dreams tracking tests and unlaned riders", `.github/workflows/derived-artifacts.yml`) after `Tests/Tool_Packs`, never before a sandboxed target; `Tests/CI/test_ci_queue_pressure_contract.py`'s `DREAMS_TRACKING_RIDERS` pin lists it last. Serially with the step's flags (`--timeout=180`): 47 passed in 63 s and 72 s (load 14-18). The whole step through `r19-step.sh` (`-p no:xdist --timeout=180`): 801 passed in 780 s at load 15-20. The CI contract files that pin the job passed (812 at `-n 4`).

**Files.** `tldw_chatbook/UI/Console_Modules/character_context.py`, `Tests/Backup_Recovery/test_character_presentation_cadence.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, and `Docs/User_Guide/console/sessions-tabs-workspaces.md` (an interrupted reload clears the "Refreshing" line).

**Sibling paths (AC #5-#8, added at the coordinator's request).** The same IDLE-only display check strands any operation that keeps its loading phase. Each path, with a test red at base (product file at b31e3e9241, every red at the stuck-phase assertion, and the phase at the blocked read confirmed REFRESHING or SEARCHING):
- **A refresh whose open chat or data profile moved during its read (sticks).** `test_refresh_that_publishes_nothing_leaves_the_panel_idle[conversation|database]`. During a run, a display check for a changed open chat goes straight to the stock `refresh()` (`known_changed` in `refresh_if_scope_changed`). If the chat or the database accessor moves again while `_refresh_recent_batch` reads, `readers.ambient_current()` is False and the refresh returns without publishing, as designed ("its next check reloads"), but the panel stayed REFRESHING and the next check refused to run.
- **The original refresh loop with a custom owner (sticks).** `[direct]`: `refresh(_presentation_is_current=...)` whose owner stops being current returns at `not presentation_is_current()` after its read, also REFRESHING.
- **A cancelled search (sticks).** `test_cancelled_character_search_does_not_leave_the_panel_searching`: cancelled during its scope read, the panel stayed on "Searching local character chats…".

**Fix: one guard, where the phase is owned.** `ConsoleCharacterContextController._settle_unpublished(generation, phase, *, reload=False, error="")` settles the panel IDLE unless a newer operation owns the generation or the operation already published (its phase is no longer the one it began). `refresh()` and `search()` call it from a `finally` around everything after `_begin`, which replaces the earlier `CancelledError` handler, so cancellation, a raised error and every unpublished return take the same path. A refresh drops its fingerprint (`invalidate_scope(_supersede=False)`), so the next check reloads. A search keeps its query and fingerprint and shows "Could not search local character chats · Retry"; the widget's Retry re-runs the search for that query. Dropping a search's fingerprint was rejected: the next check's reload publishes a browse state without the query, which would hide the Retry while the search box still holds the text.

**Paths that do not stick (AC #7), with evidence.**
- Exits where a newer operation took the generation (`generation != self._generation` in refresh and search): the newer operation publishes its own phase. The guard checks the generation; `[superseded]` passes.
- A refresh with no owner (Retry, the first load): `presentation_is_current()` is always True, a moved scope `continue`s, and the loop ends by publishing "Local character chats changed · Retry". No unpublished return remains, apart from cancellation or a raise.
- Search's moved-scope exits `continue` and end by publishing the same Retry. What remains is a cancellation or raise, and the database-unavailable branch when the scope moved; both now take the guard.
- The display check's merged read (`_check_recent_batch`) publishes without `_begin`, so it never sets a loading phase. In this task's first test draft the barrier fired in that read with the phase IDLE, so that draft took the run route instead.

**Red/green.** At base: the character file gives 5 failed (`[cancelled]` and the four new cases), 46 passed. With the guard: 51 of 51.

**Other runs with the guard.**
- The 18 covering files at `-n 4` (load about 50): 435 passed, 11 failed. Ten are the base reds listed above. The eleventh, `test_console_character_section_collapsed_render.py::test_send_with_the_character_section_collapsed_renders_nothing_there` (a census entry; a private-profile child), failed at the child's deadline (`private_profile.py:410`, "still running after …"), not at an assertion.
- That file is load-flaky at base as well. With `test_console_character_context.py` and `test_character_context_async_ownership.py` at `-n 4` (load 54-73), base failed in 3 of 3 runs: `test_opening_after_a_collapsed_refresh_shows_its_chats_without_refreshing` 3 times, `test_widget_query_validation_preserves_search_and_clear` once. With the guard, 3 of 4 runs failed (`test_send_with…` twice, `test_opening_after…` once, the query-validation test once), and one run passed 92 of 92 (load 60).
- In 3 of those guarded runs, a temporary file write (reverted) recorded every time the guard settled the panel, the private-profile children included: zero times. With no firing, the try/finally leaves those tests' behaviour unchanged. `test_send_with…` alone, serially: 3 of 3 passed with the guard, 3 of 3 at base.
- The first 47 tests kept their results.

**Soak (AC #8).** The character file (51 tests), 20 whole-file runs at `-n 4` as two concurrent loops of 10, beside other agents' load (1-minute load 54-102): 20 of 20 passed, 51 of 51 each.

**Lane.** The riders job's note now gives the file's 51 tests: 125-134 s serially with the step's flags at load 59-79 (the first 47 took 63-72 s at load 14-18). The whole step through `r19-step.sh` (`-p no:xdist --timeout=180`): 805 passed in 1174 s at load 60-100.

**Files (sibling paths).** `tldw_chatbook/UI/Console_Modules/character_context.py` (`_settle_unpublished`, plus `finally` blocks in `refresh()` and `search()`), `Tests/Backup_Recovery/test_character_presentation_cadence.py` (4 tests), `.github/workflows/derived-artifacts.yml` (lane note), `Docs/User_Guide/console/sessions-tabs-workspaces.md` (an interrupted search offers Retry; a moved chat or data profile clears "Refreshing").

**OPENING and REPAIRING (AC #9-#11, added at the coordinator's request).** All three stuck. Each new test blocks the operation, checks the phase while it is blocked, then cancels it, and is red at base on "still OPENING/REPAIRING":
- **Opening a chat (`activate`).** `test_cancelled_character_chat_opening_does_not_leave_the_panel_opening`. The test blocks `_activate_target`. `activate` turns any `Exception` into a FAILED result and its Retry copy, but a `CancelledError` went past that, so the panel kept OPENING (the row's "opening" state) and the IDLE-only display check never ran again.
- **Preparing a repair (`repair_unavailable` → `_prepare_unavailable_repair`).** `test_cancelled_character_repair_does_not_leave_the_panel_repairing[repair]`, blocked in its scope read (`pairs_return`). Cancellation sticks. So does the database-unavailable branch when the scope moved meanwhile, which returns without publishing.
- **Opening the unavailable chat's Library detail (`open_unavailable`).** `[open_detail]`, blocked in the same read. Its `except Exception` publishes a failure, but cancellation got past it.

**Fix: the same guard.** Each operation now wraps everything after `_begin` in `try/finally: _settle_unpublished(generation, phase, error=...)`. The copy for each is the one that operation already shows for a failure. Unless a newer operation owns the panel or this one already published, the guard publishes IDLE with no busy row and that copy:
- **Opening a chat: "Could not open character chat · Retry"** (the FAILED copy). The fingerprint stays. If the open moved the scope (a cancellation can land after `_activate_target` committed), the next display check finds the new open chat through the ambient or owner key and reloads. If nothing moved, the panel keeps the list it showed before, with Retry, exactly as after a failed open.
- **Preparing a repair: "Could not refresh Library details · Retry"** (its DB-failure copy).
- **Opening the detail: "Could not open Library details · Retry"** (its scope-failure copy).
- Neither repair navigates after a settle: both navigate only after their own successful publish. The test records `_navigate_repair` and `_navigate_inspection` and checks they were never called, so a cancelled repair never claims success.
- After each settle, the next display check reads again (`pairs >= 1`).

**Red/green.** The three new nodes fail at base and at the previous branch tip (1538958185), each at the stuck assertion. With the guards, the character file passes 54 of 54.

**Other runs with all guards.**
- The 18 covering files at `-n 4` (load about 55-65): 437 passed, 12 failed. Ten are the base reds listed above. The other two are load flakes:
  - `test_console_character_section_collapsed_render.py::test_send_with_the_character_section_collapsed_renders_nothing_there` (the child deadline, as above).
  - `test_console_character_context.py::test_opening_search_result_escape_preserves_query_rows_and_focus`, the focus assertion at line 1582. In that test the activation returns `CANCELLED_PRECOMMIT` and publishes, so the guard is a no-op. Serially it passed 6 of 6 with the guards and 6 of 6 at base. A temporary file write (reverted) recorded zero guard firings over two `-n 4` runs of its file (64 of 64 each).
- Soak (AC #11): the character file (54 tests), 20 whole-file runs at `-n 4` as two concurrent loops of 10 (load 47-82): 20 of 20 passed, 54 of 54 each.
- The riders job's note now gives 54 tests: 138-140 s serially with the step's flags at load 37-64. The whole step through `r19-step.sh`: 808 passed in 1026 s at load 37-82.

**Follow-up candidate (not fixed here).** `Tests/UI/test_console_character_section_collapsed_render.py` (a UI census entry) is load-flaky at base: its private-profile child tests miss their deadline under load (`private_profile.py:410`, "still running after …").
- `test_opening_after_a_collapsed_refresh_shows_its_chats_without_refreshing`: 3 of 3 base runs at `-n 4`, load 54-73.
- `test_send_with_the_character_section_collapsed_renders_nothing_there`: seen in runs with the guards; the guard fired zero times in those runs.

**Files (OPENING/REPAIRING).** `tldw_chatbook/UI/Console_Modules/character_context.py` (`finally` guards in `activate`, `_prepare_unavailable_repair` and `open_unavailable`; the guard's docstring now names every operation), `Tests/Backup_Recovery/test_character_presentation_cadence.py` (3 tests), `.github/workflows/derived-artifacts.yml` (lane note), `Docs/User_Guide/console/sessions-tabs-workspaces.md` (an interrupted open or repair shows its Retry; a repair opens nothing).
- Checks on this code: `Tests/CI` at `-n 4` gave 1823 passed, 2 skipped. `Tests/Architecture` serially (the product file is scanned) gave 1492 passed, 5 skipped. `preflight.sh`, `ruff --select F` and `ruff_gate.py` passed.
<!-- SECTION:NOTES:END -->
