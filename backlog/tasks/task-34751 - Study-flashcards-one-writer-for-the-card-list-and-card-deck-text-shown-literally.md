---
id: TASK-34751
title: >-
  Study flashcards: one writer for the card list, and card/deck text shown
  literally
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-10 16:28'
updated_date: '2026-10-10 16:35'
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
Study > Flashcards rebuilds #card-list from several writers that can be in flight at once: refresh_decks restores or picks the deck through Select.value, which posts Select.Changed, and StudyWindow.handle_deck_select_changed rebuilds the list in a study-refresh-cards worker while the caller (create_deck, delete_selected_deck, initialize_view) rebuilds it too; Create/Delete/Move Card and the Refresh button are further writers. Each rebuild cleared the list and appended row by row with an await per row, so overlapping rebuilds interleaved. Evidence on dev 75159f8843: a 4-card deck re-listed through initialize_view showed 7 rows out of order; two overlapping refresh_cards calls showed 8 rows; the investigation also saw two 'No cards in this deck.' rows on the create path. The same interleave makes the gated UI-lane contract test test_real_service_create_deck_select_deck_and_add_card_in_local_mode flaky on a loaded runner (assert [] == ['No cards in this deck.'] read between the deck switch's clear and append) and the still-running worker then raises MountError in run_test teardown, which blocks unrelated PRs in the merge queue. Separately, card fronts/backs and deck names are user text but reach Label/Static/Select as Textual markup: the '[new]' queue state on every row is swallowed as a style tag, and a front such as 'What does [/b] do?' raises MarkupError in the compositor, which exits the app (reproduced in a Pilot run: Refresh button -> row mounted -> reflow -> MarkupError); a deck name with '[/b]' raises from create_deck itself.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overlapping card-list rebuilds (deck switch, re-list, refresh, create/delete/move) leave exactly one row per card, in order, and current_cards matches the rows
- [x] #2 The gated contract test test_real_service_create_deck_select_deck_and_add_card_in_local_mode no longer flakes: it passes repeatedly, including with extra event-loop turns injected before its card-list assert and with create_deck's own rebuild slowed
- [x] #3 A card-list rebuild still in flight when the list leaves the app (view switch, app shutdown) stops without writing, so no MountError at teardown
- [x] #4 Card fronts/backs, deck names, quiz names/questions/answers and workspace names containing markup-like brackets render literally everywhere Study shows them (card and question rows, review panel, status lines, deck/quiz pickers, dashboard, quiz session panel, the deck-created toast) and never raise MarkupError
- [x] #5 Each new or changed test fails on origin/dev for the stated reason and passes with the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on origin/dev with a real-app Pilot regression file: initialize_view re-list (7 rows for 4 cards), two overlapping refresh_cards (8 rows), markup-like card front / deck name (MarkupError).
2. Single-writer rule in StudyFlashcardsController.refresh_cards: each call claims a generation; after every await a superseded or detached rebuild stops without writing; rows are built first and mounted with one extend. Keep callers' explicit refresh_cards (they must work even when no Select.Changed fires) -- the guard makes the redundant Select.Changed rebuild harmless instead of relying on knowing which events fire.
3. Contract test waits for Study workers to settle before reading #card-list (the deck switch's rebuild is legitimately still running when create_deck returns).
4. Render user text literally: markup=False on card rows and the review/quiz Statics that show user text; deck/quiz picker prompts passed as Content.
5. Verify: red on dev / green on branch per test; contract test repeated and under induced delay; other Study files vs dev by test id; module-size ratchet; preflight; live tmux run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Single writer for `#card-list`.** `refresh_cards` is the one function that writes the list, and the most recent call owns it: each call claims a generation (`_claim_card_list`), and after every `await` (clear, fetch, mount) a call that a newer claim superseded -- or whose list left the app (`is_attached`: a sub-view switch, or `_exit` at shutdown) -- stops without writing (`_owns_card_list`). `handle_scope_changed`, which empties the list, claims too. Rows are built first and mounted with one `extend` (one checkpoint instead of one per row).

Rule chosen over "callers stop calling `refresh_cards` when a `Select.Changed` rebuild will run": whether `Select.Changed` fires depends on Textual internals (`set_options` resets the value, so a restored selection posts it; an empty deck list may not), and the list has other writers no worker group can serialise (Create/Delete/Move Card, Refresh, the deck-switch worker, `initialize_view`). Guarding the writer covers every interleave; the price is one redundant fetch per programmatic deck switch, and `await create_deck()` no longer implies the list is final -- the deck switch's own rebuild may still be running.

**Contract test.** `test_real_service_create_deck_select_deck_and_add_card_in_local_mode` now waits until no `study-*` worker is pending before reading `#card-list`. The code fix alone does not close the flake: in an induced-delay harness (0/1/2/4/8/16/32 `asyncio.sleep(0)` between `create_deck()` and the assert, with and without `create_deck`'s own rebuild slowed 50 ms), the original assert failed on dev at 0-4 yields with exactly the CI signature (`[] == ['No cards in this deck.']` + `WorkerFailed(MountError)` in teardown) and still read the legitimate in-flight rebuild at 1-4 yields on the fix (no MountError). With the wait, all 28 cases pass on both, and the fixed contract test passed 9/9 repeated runs.

**Literal user text.** Card rows, question rows and the Study Statics that show user text (`#review-status/-front/-back/-next-intervals`, the quiz attempt status/question/history, the workspace-name banner, the dashboard's scope/recent decks/recent quizzes/source status, the quiz session panel) are `markup=False`; deck and quiz picker prompts are `Content`; the dashboard's Resume label is `Content`. The queue state (`[new]`) now shows on every row.

**Deviation: the notification sink.** The live run on the branch found one more crash after every widget above was fixed: `LocalStudyService.create_deck` raises a "Local study deck created: ..." toast through `NotificationDispatchService`, and `App.notify` defaults to markup, so a deck named `bd9st Bio [/b] deck` exited the app the moment it was created. No Pilot test could see it: `run_test` disables notifications unless `notifications=True`, and the real-service builder leaves the dispatcher unwired. Fixed at the sink (`Utils/NotificationHelper.show_notification` and the dispatch fallback pass `markup=False`); dispatched titles/messages are composed from data everywhere (study, quizzes, watchlists, reminders, research, media, evaluations) and none carries intended markup. The toast text itself still repeats its title ("Local study deck created: Local study deck created: ...") -- untouched.

**Tests.** New `Tests/UI/test_study_card_list_single_writer.py` (4: re-list 7->4 rows, two overlapping refreshes 8->4, slowed create-deck 2->1 empty rows, rebuild in flight when the view switches -> `NoMatches` on dev) -- added to the UI PR gate census (16.5 s pytest / 19.8 s wall; `MINIMUM_FILES` 164). New `Tests/UI/test_study_user_text_literal.py` (5, all `MarkupError` on dev; outside the lane at ~22-27 s). Updated `Tests/Subscriptions/test_notification_dispatch_service.py` (+1) and `Tests/Scheduling/test_reminder_handler.py` (fake `notify` takes `markup`, asserts it is off). Mutation checks: dropping the post-fetch ownership check re-fails the slowed create-deck test; dropping `is_attached` re-fails the in-flight test; re-enabling markup on `#review-next-intervals` re-fails the surfaces test.

**Verified** against dev `75159f8843` on 2026-10-10: 13 Study-related, 15 notification-sink and 7 other Study-referencing test files show identical outcomes by test id on dev and branch (the reds are environmental `RecoveryRequired: raw_source_selection_changed` and pre-existing dev failures). `./scripts/preflight.sh` passes; the Textual worker contract reports no new post-await lookups. Live (isolated profile, null keyring, 160x45): deck `bd9st Bio [/b] deck` created with its toast shown literally, cards `What does [/b] do? [new]` / `bd9st Q2 [x]` / `bd9st Q3` listed once each after three rapid Refresh presses and a Dashboard -> Flashcards re-list, review front/back literal, zero unhandled exceptions in the profile log.

**Follow-ups not in scope.** `quizzes_handler.refresh_questions` has the same clear-then-append-per-row shape as the old `refresh_cards`; a move-target value set programmatically while its dropdown was open read back as blank, plausibly via `handle_move_target_changed` -> `_sync_move_target_options` -> `set_options` resetting it (seen once while writing the deck-name test; not reproduced by hand, not investigated).

Files: `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`, `UI/Study_Modules/quizzes_handler.py`, `UI/Study_Window.py`, `Widgets/Study/study_dashboard.py`, `Widgets/Study/quiz_session_widget.py`, `Utils/NotificationHelper.py`, `Notifications/notification_dispatch_service.py`, the four test files above, `Tests/UI/test_study_flashcards_real_service_contract.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->
