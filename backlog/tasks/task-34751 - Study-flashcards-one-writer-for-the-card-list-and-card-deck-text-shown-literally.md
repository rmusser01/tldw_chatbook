---
id: TASK-34751
title: >-
  Study flashcards: one writer for the card list, and card/deck text shown
  literally
status: Done
assignee:
  - '@claude'
created_date: '2026-10-10 16:28'
updated_date: '2026-10-19 22:58'
labels:
  - study
  - ui
  - flaky-test
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Study > Flashcards rebuilt #card-list from several writers that could be in flight at once (create/delete/move card, Refresh, initialize_view, and the deck switch's own Select.Changed rebuild worker), each clearing the list and appending row by row, so overlapping rebuilds interleaved: on dev 75159f8843 a 4-card deck re-listed showed 7 rows, two overlapping refreshes showed 8. The same race made the gated UI-lane contract test test_real_service_create_deck_select_deck_and_add_card_in_local_mode flaky on a loaded runner, and a rebuild still running when the test app shut down raised MountError at teardown, blocking unrelated PRs in the merge queue. Separately, card fronts/backs, deck and quiz names and workspace names are user text but reached Label/Static/Select as Textual markup: '[new]' vanished as a style tag and '[/b]' raised MarkupError, which exits the app.

Since this task was filed, dev took most of that in PR #3057 (commits tagged TASK-33007: one #card-list rebuild at a time behind a lock, Flashcards actions that end quietly when the sub-view is swapped out mid-await, literal card rows, review panel, deck/quiz pickers, dashboard and quiz session), and dev's notification helper now passes markup=False. Still broken on dev ce4aa4a231 (2026-10-19): the contract test still flakes (the deck switch's rebuild can queue behind create_card's and replace the row the test waits to see painted, or resume after the app quits); a card-list rebuild resuming while the app quits raises MountError from its worker; and a workspace name shaped like markup crashes the app when the Study scope banner first paints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overlapping card-list rebuilds (deck switch, re-list, refresh, create/delete/move) leave exactly one row per card, in order, and current_cards matches the rows
- [x] #2 The gated contract test test_real_service_create_deck_select_deck_and_add_card_in_local_mode no longer flakes: it passes repeatedly, including with extra event-loop turns injected before its card-list assert and with create_deck's own rebuild slowed
- [x] #3 A card-list rebuild still in flight when the list leaves the app (view switch, app shutdown) ends quietly: no MountError or NoMatches escapes its worker, so nothing fails at teardown; a MountError while the Study window is still attached still raises
- [x] #4 Card fronts/backs, deck names, quiz names/questions/answers and workspace names containing markup-like brackets render literally everywhere Study shows them (card and question rows, review panel, status lines, deck/quiz pickers, dashboard, quiz session panel, the deck-created toast) and never raise MarkupError
- [x] #5 Each new or changed test fails on origin/dev for the stated reason and passes with the fix; a test that pins behaviour dev already has fails under the mutation it guards against
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Re-planned 2026-10-19, after PR #3057 landed most of the original scope on dev.
1. Re-baseline on dev: keep only what is still red there; drop this branch's own card-list writer rule, the notification-sink change and the literal-text edits dev now carries.
2. Contract test: read #card-list only once no study-* worker is pending (checked after a pause, and again after a second), after the first load, the deck switch and create_card.
3. Shutdown: _quiet_when_swapped_out ends an action quietly on NoMatches or MountError once the Study window has left the app (App._exit set); a MountError while attached still raises.
4. Scope banner: the workspace name Static renders without markup.
5. Pin the deck/quiz pickers' type-to-search over markup-shaped names (rich Text prompts; Content prompts are not searchable).
6. Verify: red on dev / green on branch per test; an induced-ordering sweep of the contract test; Study test files compared with dev by test id; preflight; live tmux run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Scope after the rebase (2026-10-19, onto dev ce4aa4a231).** PR #3057 had landed most of the original scope while this PR waited, under commits tagged TASK-33007: one `#card-list` rebuild at a time (`_refresh_cards_lock`), `_quiet_when_swapped_out` on the Flashcards actions, literal card/question rows, review and quiz-attempt Statics, dashboard and quiz session, deck/quiz picker prompts as rich `Text`, and a picked move target that stays picked. Dev's notification helper and dispatch fallback now pass `markup=False`, and the Scheduling `notify` fakes take `markup`. This branch's own versions of those (a generation-claim single writer with one `extend`, its `markup=False` edits, the sink change) did not survive the rebase. What remains is what was still red on dev. AC #1 is met by dev's lock; the lock serialises rebuilds, so the last one to run writes the whole list.

**1. Contract test (AC #2).** `_wait_for_study_workers` pauses, then requires no pending `study-*` worker across two pauses. `_open_flashcards` waits for the view's first load. The test reads `#card-list` only after the deck switch's rebuild has finished, and again after create_card's. Dev's lock serialises rebuilds but does not order them: when create_card's rebuild takes the lock before the deck switch's worker reaches it, the worker runs last and replaces the row the test is waiting to see painted.

**2. Shutdown (AC #3).** Once `App._exit` is set, `is_attached` is False for every widget, so a rebuild resuming after its fetch appended into a list "not mounted" and raised `MountError` from its worker. `_quiet_when_swapped_out` now ends quietly on `NoMatches` or `MountError` once the window is detached, and still raises a `MountError` while it is attached. A view switch was already quiet on dev: mounting into a pruned list is a no-op, and `NoMatches` after a swap is handled. The guard stops crashes, not every write: after a swap out and back, a rebuild that outlived its panel still sets `current_cards` and writes the review status and lifecycle controls onto the new panel, and so do its `create_deck`/`create_card` callers. That predates this branch and is now AC #4 of TASK-34755. The quit test asserts that the held rebuild really resumed past its fetch, so it cannot go green on a Textual that cancels the worker first. Two unit tests are the negative control: a `MountError` with the window attached raises, whether Flashcards is shown or swapped out, and a detached window is quiet for both errors.

**3. Scope banner (AC #4).** `#study-scope-workspace-name` is `markup=False`. `_sync_scope_banner`'s own `except Exception` swallowed the `MarkupError` from `update()`, so the crash came later, when the compositor first painted the banner. No in-app path enters workspace scope on dev (`enter_workspace_scope` has no caller outside tests), so this is verified in Pilot only.

**4. Picker pin (AC #4).** Typing a first letter in the open deck or quiz picker selects a markup-named entry, and its label paints literally. Textual 8.2.8's `SelectOverlay._find_search_match` reads only `Text`/`str` prompts, so `Content` prompts render correctly but cannot be found by typing.

**Red/green, on dev ce4aa4a231 vs this branch:**
- `test_study_flashcards_quit_mid_rebuild.py`: RED on dev with `WorkerFailed: MountError("Can't mount widget(s) before ListView(id='card-list', ...) is mounted")`; GREEN.
- `test_workspace_name_with_markup_brackets_paints_literally_in_the_scope_banner`: RED on dev with `MarkupError: closing tag '[/b]' does not match any open tag`, raised from the compositor; GREEN.
- `test_a_mount_error_while_the_window_is_attached_still_raises` (both cases) pins behaviour that is the same on dev and the branch, so it is GREEN on both. With the guard mutated to swallow every `MountError`, both cases fail ("DID NOT RAISE") while the quit test stays green. That is the gap this negative control closes. The mutation was restored by Edit.
- `test_deck_and_quiz_pickers_find_a_markup_named_entry_by_typing` pins dev's `Text` prompts, so it is GREEN on dev. It fails with the deck prompts built as `Content` ("typing 'z' to pick the Zebra deck") and with the quiz prompts as `Content` ("typing 'g' to pick the Graph quiz"). Both mutations were restored by Edit.
- Contract test under loaded orderings: a scratch probe (deleted) ran the test unchanged under 17 orderings: the deck switch's rebuild started 0.01-0.4 s late or 1-16 loop turns late, every card fetch slowed 50 ms (create_deck's included), and extra loop turns added after each of the test's waits, before its asserts. Dev's version failed 6/17 with the CI signatures: `WorkerFailed(MountError)` at teardown (4 occurrences), "timed out ... the card row to be painted" (2), `assert [] == ['No cards in this deck.']` (1), and a row region of zero, meaning the row was replaced (1). The branch passed 17/17. An earlier 7-case sweep (wall-time delays only) gave the same split: dev 4/7 failed, branch 7/7 passed.

**Study tests compared with dev by test id** (one file per command, host load 60-100 on 14 cores):
- `test_study_flashcards_real_service_contract.py`: 2/2 on both.
- `test_study_controls_geometry.py`: 6/6 on both. Dev also had one teardown error ("Native app workers remain uncertain; config retirement refused" plus a TimeoutError). The branch did not; this is not claimed as a fix.
- `test_study_flashcards_screen.py`: 29 identical reds on both, all environmental `RecoveryRequired`. The branch had one more red, `test_a_picked_move_target_stays_picked_and_enables_move` (Move still disabled after one pause, load 97); it then passed 3/3 on the branch and 3/3 on dev.
- `test_study_screen.py` (11 reds), `test_study_dashboard.py` (7 reds + 1 error) and `test_study_quizzes_screen.py` (15 reds): identical by test id on both, all `RecoveryRequired`.
- `Tests/UI/test_mount_io_off_pump.py`: 10 errors on both, identical by test id, all `RecoveryRequired`.
- Architecture tests (`test_study_handler_service_keywords`, `test_backwards_select_option_guard`, `test_tier2_evals_media_study_guards`): all pass on both.
- New files on the branch: `test_study_user_text_literal.py` 2/2 in 27.9 s of pytest at load 66; `test_study_flashcards_quit_mid_rebuild.py` 1/1 in 25.3 s at load 70.

**Census.** Both new files are in `scripts/ui_pr_gate_census.txt`. `MINIMUM_FILES` goes from dev's 327 to 329 (dev 8136af6f6f; it was 326 to 328 before the last rebase).

**Live** (2026-10-19 17:28-17:39, `launch.sh` at 160x45, isolated profile, null keyring; evidence in `uxrev/evidence/bd5/R64/final/`):
- Created a deck named `r64f Bio [/b] [new] deck`. The picker and the toast showed it literally.
- Added cards `What does [/b] do?`, `r64 Q2 [x]` (its back is `[@click=app.quit]x[/]`) and `r64 Q3 [bold]plain?`. Each was listed once, with `[new]` shown.
- Pressed Refresh six times in a row; still 3 rows.
- Pressed Refresh twice, switched to Dashboard at once, then back to Flashcards; still 3 rows.
- Created `Alpha r64 deck`, opened the deck picker and typed `r` then Enter. The `[/b]` deck was selected, with its label literal and its 3 cards listed.
- Review showed the front and back literally.
- Pressed Refresh twice and quit at once. The profile log has 0 lines matching `unhandled_exception|MarkupError|MountError|WorkerFailed|Traceback`. This is a smoke check, not evidence for the quit fix. The local fetch is fast, so nothing shows a rebuild was still in flight at quit, and dev would very likely pass the same steps. The evidence for the quit fix is the Pilot test, which holds the fetch until the app is exiting.

**The 2026-10-10 review's items:**
- IMPORTANT 1 (the Scheduling `notify` fakes) and IMPORTANT 2 (`Content` prompts break type-to-search) are on dev already. The second is pinned here.
- MINOR 3 (`is_attached` during teardown): the guard now runs after the exception is raised. A mount into a pruning list is a no-op, and `is_attached` turns False exactly when `App._exit` is set, which is the case the guard handles.
- MINOR 4 (callers writing the review panel after `refresh_cards` stopped): each caller is itself wrapped in `_quiet_when_swapped_out`, which stops crashes but not the writes. After a swap out and back, those writes land on the new panel. That is AC #4 of TASK-34755, alongside the overwritten "Deck created." status.
- MINOR 6: the worker wait pauses before its first check, and the quit test holds the fetch on Events.
- MINOR 7: follow-ups TASK-34752 (the quizzes question list), TASK-34753 (doubled toast title) and TASK-34755 (Deck-created status) are filed. Their ids were checked as unused on every ref and in every worktree. The move-target draft (34754) was dropped because dev pins that behaviour.

**The 2026-10-19 review's follow-ups (verdict: ship, nothing blocking):**
- The picker test's quiz half now waits for the Study workers, as its deck half does, so the Quizzes view's initialize worker cannot outlive the test.
- The quit test asserts that the rebuild resumed past its held fetch.
- Two unit tests now pin the attached `MountError` negative control.
- AC #3 is reworded to what the guard guarantees. The stale writes are AC #4 of TASK-34755.
- The test docstrings cite dev ce4aa4a231, where the red evidence was taken.
- The live notes no longer suggest that the quit run exercised the fix.
- The teardown error at high host load (the executor join timing out, "Native app workers remain uncertain") is filed as TASK-35663. It is environmental and also hits dev's unchanged tests. The id was checked as unused on every ref and in every worktree; the highest seen was 35662.

**Out of scope.** `#review-next-intervals` stays markup-on. It shows scheduler interval strings, not user text.

**Docs.** Study has no User Guide page, and the visible change is a crash fix, so no guide edit was made. A new lesson in `backlog/docs/lessons-backlog-hygiene.md` covers this overlap: a flaky gated test gets fixed inside every PR it blocks, so search dev by code, not by task id.

**Merge evidence (2026-10-19 22:05-22:58, rebased onto dev 8136af6f6f, which had moved 73 files; host load 40-86 on 14 cores).** The census conflict was resolved as dev's floor plus this branch's two files (327 -> 329), keeping both dated comments. Branch and a detached dev worktree, one file per command under the plain pytest command:
- New tests RED on dev 8136af6f6f with the stated causes: the quit test fails with `WorkerFailed: MountError("Can't mount widget(s) before ListView(id='card-list', classes='flashcard-list') is mounted")`, and so do both detached guard cases (dev re-raises the `NoMatches` while Flashcards is shown, and does not catch `MountError` at all); the banner test fails with `MarkupError: closing tag '[/b]' does not match any open tag`. The picker test and both attached negative-control cases pass on dev, as they pin dev's behaviour.
- New tests GREEN on the branch: quit file 5/5, literal file 2/2, contract file 2/2 (dev's contract file also 2/2 in this run).
- Mutation re-checked: with the guard returning on an attached `MountError`, both attached cases fail with "DID NOT RAISE" (restored by Edit, scratch worktree removed).
- Existing files, reds identical by test id: `test_study_screen.py` 11, `test_study_dashboard.py` 7 + 1 error, `test_study_quizzes_screen.py` 15, `test_mount_io_off_pump.py` 10, all `RecoveryRequired: raw_source_selection_changed`; the three Architecture files pass on both.
- `test_study_flashcards_screen.py`: 29 identical `RecoveryRequired` reds. The move-target test failed once more on the branch (Move still disabled after one pause); it then passed in 2 more whole-file runs on each side and 6/6 alone on each side. Filed as TASK-35790 (not gated).
- `test_study_controls_geometry.py`: 6 bodies pass on both; `[size0]` ended as the TASK-35663 teardown ERROR on the branch twice and on dev once (dev passed once at lower load). Added to TASK-35663.
- `test_boot_css_byte_budget.py` passes. `test_ui_ready_module_census.py` fails identically on dev and the branch (1032 resident modules against a limit of 1031, the same 22 new and 12 shed modules, none of them Study); this branch adds no module and no CSS.
- No module-size ratchet row covers `flashcards_handler.py` or `Study_Window.py`. `./scripts/preflight.sh` passes (329 gated entries, floor 329).
- Live, on the rebased head (evidence `uxrev/evidence/bd5/R64/rebased/`): a deck named `r64rb Bio [/b] [new] deck` shows literally in the picker; a card `What does [/b] do? r64rb` lists literally with `[new]`; Refresh twice then Ctrl+Q exits with 0 matching lines in the 297-line profile log. A smoke check only, as above.

**Files:**
- `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`
- `tldw_chatbook/UI/Study_Window.py`
- `Tests/UI/test_study_flashcards_real_service_contract.py`
- `Tests/UI/test_study_user_text_literal.py` (new)
- `Tests/UI/test_study_flashcards_quit_mid_rebuild.py` (new)
- `scripts/ui_pr_gate_census.txt`
- `scripts/check_ui_pr_gate_census.py`
- `backlog/docs/lessons-backlog-hygiene.md`
- `backlog/tasks/task-34752`, `task-34753`, `task-34755`, `task-35663`, `task-35790`
<!-- SECTION:NOTES:END -->
