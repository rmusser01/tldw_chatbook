---
id: TASK-40313
title: "A refused automatic or manual mixin attempt leaves an empty accepted reply row that reads as a pending response"
status: Done
assignee: ['@claude']
updated_date: 2026-11-13 05:42
created_date: 2026-11-12 19:36
labels:
- follow-up
- console
- persistence
priority: medium
references:
- tldw_chatbook/Chat/console_prompt_mixin_service.py
- tldw_chatbook/Chat/chat_persistence_service.py
- tldw_chatbook/Chat/console_conversation_hydration.py
- tldw_chatbook/Chat/assistant_generation_state.py
dependencies:
- TASK-39159
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream, TASK-39159). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** A refused turn should leave no trace. Today it leaves an empty assistant row in the DB, and reopening the conversation probably shows it as "Response accepted; waiting for dispatch." (inferred from the code, not run), which reads as a send that is still pending or stuck.

**Evidence.** `console_prompt_mixin_service.py:695-705` raises `mixin_attempt_owner_changed` after the save and before `adopt_saved_pending_assistant`. The saved row (`chat_persistence_service.py:1302-1325`) is empty with `assistant_generation_state='accepted'`. The manual path, `accept_attempt` (`:593-604`), leaves the same row. Hydration keeps any row whose generation state is set (`console_conversation_hydration.py:258-263`), and the copy for an accepted row is at `assistant_generation_state.py:25`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a refused automatic or manual attempt, the saved reply row is settled to a terminal state or removed
- [x] #2 Reopening the conversation shows no pending reply for the refused attempt
- [x] #3 One test covers both refusal paths and is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce both refusals on base 26b2e677dd in the wake rig (`_controller_rig`): the automatic path (a wake's `save_pending_reply`) and the manual path. Find which manual turn saves the row through the attempt's `pending_assistant`. Read the DB row and the reopened (hydrated) tree.
2. Write one test parametrized over both paths that rebinds the chat during the held save, then asserts the reopened tree has no row for the refused attempt and no `accepted` assistant row. Show it red.
3. On each owner-changed refusal, tombstone the row this save created, through the captured persistence's existing `delete_message_subtree`, then raise as before. Best effort: the turn is refused either way.

## Implementation Notes

**Premise confirmed on base 26b2e677dd** (a probe, since deleted). A plain `submit_draft` commits its reply through the native durable commit, so its attempt save passes no `pending_assistant`. A `regenerate_message` does pass one, and that makes it the manual path that saves the row. After a rebind during the held save, both paths left `('', 'accepted', deleted=0)` in `messages`. The in-memory row was failed and unadopted, so `mark_message_failed` never wrote it. Reopening the conversation (`get_conversation_tree` -> `console_messages_from_conversation_tree`) kept the row with generation state `accepted`. Per `Docs/User_Guide/console/chat-basics.md` that reads "Response accepted; waiting for dispatch." and offers **Retry response**, which would have sent the refused turn after all.

**Fix** (`console_prompt_mixin_service.py`). A new `_discard_refused_reply` runs the captured persistence's existing `delete_message_subtree` on the same owned DB call as the save (`_run_owned_save`). That method is the Console's message-delete writer. It soft-deletes the row and releases recovered-media references, which a media-free row has none of. `accept_attempt` calls it only when its save created the row (`pending_assistant is not None`), and `save_pending_reply` always does, just before each raises `mixin_attempt_owner_changed`. A failed discard only logs. `chat_persistence_service.py` sits exactly at its size budget, so nothing new went there.

The close case of the existing `test_a_wake_whose_owner_moves_during_its_reply_save_is_refused` takes the same refusal, so a chat closed during the save also loses the row. That test's raw-row reads now select `deleted` and pin `(session.id, "", 1)` for all three moves (binding, persistence, close); fix round 2 made that change.

**Test.** `Tests/Chat/test_automatic_wake_recovery.py::test_a_refused_attempt_leaves_no_pending_reply_when_reopened[automatic|manual]`, in a file the recovered-suites lane runs whole. It holds the save, rebinds the chat, releases it, checks the refusal copy and that the row was not adopted, then reopens the tree. Red on base for both cases: `'<id>' not in {'<id>': 'accepted', ...}`. Green after: the file passes 12/12 (`-n 4`).

**Covering run** (`-n 4`, load 60-75): the wake, fleet-wake, goal-dispatch, fork census and semantic-mutation inventory files, all `test_console_prompt_mixin_*`, `Tests/DB/test_console_prompt_mixins.py`, `Tests/UI/test_console_prompt_mixins.py`, regenerate branching and variant stream: 601 passed, 4 failed.
- The three `Tests/Chat/test_fleet_wake_retry.py` nodes fail the same way with the base service (that file has no profile marker).
- `Tests/UI/test_console_prompt_mixins.py::test_browse_mixins_read_does_not_hold_navigation[close]` timed out in `app.workers.wait_for_complete()` once, then passed twice on rerun with the change.

Diagnostic inventory: preflight reported `console_prompt_mixin_service.py` at 1 -> 2 diagnostic calls. `check_persistent_diagnostic_inventory.py --statements ... --since bf9182a4f6` shows one added statement, `logger.warning("Console refused reply row could not be discarded")`, a constant with no interpolation. The pin was regenerated with `--write`.

**Fix round 2 (review r14con-rev1).**
- *Semantic-mutation inventory.* The first round's note wrongly said this writer was already in the inventory. The census scanner sees only direct calls and the runners in `_BOUND_METHOD_RUNNERS`, so the `partial(self.persistence.delete_message_subtree, ...)` it hands to `_run_owned_save` was invisible. `_bound_method_argument_actions` in `Tests/Chat/test_console_semantic_mutation_inventory.py` now reads `partial(<receiver>.<mutator>, ...)` (and `functools.partial`) as a call of that mutator. The new route `console_prompt_mixin_service.py::ConsolePromptMixinService._discard_refused_reply::call:persistence:delete_message_subtree` is classified visibility/ownership-only, in both the test's expected routes and `Docs/Development/console-semantic-mutation-inventory.md`. The doc's counts move to 66 boundary identities and 14 visibility/ownership-only, and its soft-delete owner row names the discard. New `test_partial_of_a_mutator_counts_as_its_call` checks that a mutator partial counts, and that a non-mutator partial or an unrelated receiver does not.
  - Red without the scanner change: `test_partial_of_a_mutator_counts_as_its_call` (`set() == {...}`) and `test_public_mutation_boundary_calls_are_classified` (`stale=[...the new route]`).
  - Green with it: the whole file passes 136/136, and no other route appeared repo-wide.
- *Raw-row pins.* Under the base service, all three moves read `('<id>', '', 0)` (3 failed). With the fix, the file passes 12/12.

Files: `tldw_chatbook/Chat/console_prompt_mixin_service.py`, `Tests/Chat/test_automatic_wake_recovery.py`, `Docs/security/production-diagnostic-inventory.json`, `Tests/Chat/test_console_semantic_mutation_inventory.py`, `Docs/Development/console-semantic-mutation-inventory.md`.
