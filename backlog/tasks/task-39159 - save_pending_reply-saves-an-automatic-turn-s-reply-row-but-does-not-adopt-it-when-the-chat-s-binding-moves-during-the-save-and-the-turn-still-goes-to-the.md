---
id: TASK-39159
title: "save_pending_reply saves an automatic turn's reply row but does not adopt it when the chat's binding moves during the save, and the turn still goes to the provider"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:19
labels:
- follow-up
- console
- persistence
priority: low
references:
- tldw_chatbook/Chat/console_prompt_mixin_service.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_chat_store.py
dependencies:
- TASK-39119
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (review r12p-rev2 of TASK-39119, round-12 branch fu/r12-cprod). Read on the round-12 stack 9091820bf2, 2026-11-08; not reproduced. This code exists only in round 12, so it is named by symbol, not line number.

**The new save.** TASK-39119 gave a wake or goal iteration, which has no mix-in anchor, a pre-provider save of its reply row: `ConsolePromptMixinService.save_pending_reply`. The controller's automatic-turn path calls it when `current_user is None and wake_authorization is not None`.

**What it does when the binding moves.**
- It snapshots `binding = (session.incarnation_id, session.conversation_binding_revision)` and runs `persistence.save_pending_assistant` to its end through `_run_owned_save`.
- It then adopts the row (`store.adopt_saved_pending_assistant`) only if the store still holds the same session object with the same binding.
- Otherwise it returns normally, and the controller goes on to the provider.

**What the manual path does.** Manual turns take the mix-in attempt path, `accept_attempt`, in the same class. It checks the same session object, the binding plus `persisted_conversation_id`, and that `store.persistence` is unchanged. If any differ it raises `MixinSnapshotUnavailableError("mixin_attempt_owner_changed")`, so dispatch is refused.

**The result.** If the binding moves mid-save, the row is written under the old binding and not adopted, but the turn proceeds. The reply then streams into a row whose saved copy is not linked to it. `save_pending_reply` also does not compare `persisted_conversation_id` or the persistence object.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test moves the binding, or replaces the persistence, while `save_pending_reply` is saving, and shows what happens to the turn and to the saved row; it is red before the change if the turn still reaches the provider
- [x] #2 A moved binding refuses the automatic turn before the provider, as `accept_attempt` refuses a manual one; if it must not, the notes give the reason with evidence
- [x] #3 The test runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Test: hold a wake's `save_pending_assistant` while the chat is rebound (or its persistence replaced), and record at provider entry whether the reply row was adopted; also check the saved row and the refusal.
2. `save_pending_reply`: compare the session object, binding (incarnation, binding revision, persisted conversation id) and `store.persistence` after the save, as `accept_attempt` does, and raise `mixin_attempt_owner_changed` instead of returning.
3. Controller: catch it at the automatic-turn call site and refuse before the provider (`_block_context_preflight`), or return the closed-session result when the chat was closed.
4. Put the test in a file a PR lane already runs (not the fast contract); three timings; size ratchet row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** `ConsolePromptMixinService.save_pending_reply` now checks after its save what `accept_attempt` checks: the same session object, the same `(incarnation_id, conversation_binding_revision, persisted_conversation_id)`, and the same `store.persistence`. If any moved it raises `MixinSnapshotUnavailableError("mixin_attempt_owner_changed")` and does not adopt the row; a failed save stays best effort, as before. The controller's automatic-turn call site catches it and refuses the turn before the provider with `_block_context_preflight` ("The chat changed while the reply was being saved; response was not sent."), or returns `_session_closed_result` when the chat was closed meanwhile, so a plain close does not read as an ownership change (the TASK-39155 shape).

**What happens to the row.** The reply row saved under the old binding stays in the database (empty, `assistant_generation_state='accepted'`) and is not adopted by the live session, exactly as a refused manual attempt's pending row is.

**Test** `test_a_wake_whose_owner_moves_during_its_reply_save_is_refused[binding|persistence]` in `Tests/Chat/test_automatic_wake_recovery.py`: that file is a whole-file target of the `dreams-boundary` job in `.github/workflows/derived-artifacts.yml` (a PR lane, not the fast contract), so no workflow edit. It reuses `test_console_fleet_wake`'s rig, holds the save on a thread event, and moves the binding (rebind to None and back) or replaces `store.persistence`. Red on base ae9f9e8d21: `binding` -> "a wake reached the provider with its reply row unadopted: [False]"; `persistence` -> "the moved wake was not refused before the provider: ... COMPLETED 'Response complete.'". Green after. Three timings (`-n 4`): 1.44 s, 1.49 s, 1.48 s call (load ~71).

**Covering runs.** `Tests/Chat/test_automatic_wake_recovery.py` 9 passed. `Tests/Chat/test_console_fleet_wake.py` + `test_console_goal_dispatch.py` + `test_console_chat_start.py` (`-n 4`, load ~65): 229 passed, 2 failed (`test_repeated_start_cannot_reprovision_an_accepted_attempt`, `test_goal_authorization_cannot_be_forged_or_moved_to_another_session`); both fail the same way with this change reverted (a storage `PermissionError` at `controller_submit` in this sandbox), so they are not this change's.

**Size.** `Tests/Architecture/test_module_size_ratchet.py`: controller row 29176 -> 29183 (+7, exact; owner waiver).
<!-- SECTION:NOTES:END -->
