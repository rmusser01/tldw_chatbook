---
id: TASK-39159
title: "save_pending_reply saves an automatic turn's reply row but does not adopt it when the chat's binding moves during the save, and the turn still goes to the provider"
status: To Do
created_date: 2026-11-08 06:54
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
- [ ] #1 A test moves the binding, or replaces the persistence, while `save_pending_reply` is saving, and shows what happens to the turn and to the saved row; it is red before the change if the turn still reaches the provider
- [ ] #2 A moved binding refuses the automatic turn before the provider, as `accept_attempt` refuses a manual one; if it must not, the notes give the reason with evidence
- [ ] #3 The test runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
