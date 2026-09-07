---
id: TASK-1130
title: Restored pending_skill_install card is dead-but-clickable
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 15:20'
updated_date: '2026-07-28 06:40'
labels:
  - console
  - approvals
  - resume-state
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TaskResumeState.from_dict restores pending_skill_install across screen navigation, but skill rounds live in the controller's request_id-keyed registries and every navigation builds a fresh ChatScreen/ConsoleChatController — so the restored card can mount with no live round behind it; clicking it strict-match no-ops (fail-closed, never auto-approves). TASK-1051 established this chain and deliberately documented the script-side asymmetry (pending_skill_script is dropped); the install side keeps the hazard for round-trip data-fidelity reasons pinned by TASK-910 tests. Either stop restoring pending_skill_install too (mirroring the script decision, updating the fidelity tests) or build a real reconnection path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A restored ChatScreen never shows a skill-install confirm card whose decision cannot reach a live round.
- [x] #2 Never-auto-approve and round-identity invariants unchanged.
- [x] #3 TASK-910's round-trip fidelity tests updated coherently with the chosen branch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify TASK-1051's precedent still holds at HEAD: fresh ConsoleChatController per navigation (chat_screen.py _ensure_console_chat_controller), no new reconnection seam, TASK-1143's navigation guard now also denies busy rounds on teardown.
2. Drop pending_skill_install symmetrically with pending_skill_script in TaskResumeState.from_dict (chat_screen_state.py), citing both TASK-1051 and TASK-1130 in the docstring/comments.
3. Sweep chat_task_cards.py and chat_screen.py for consumers of a restored pending_skill_install to confirm no branch goes half-dead now that the field is always None post-restore.
4. Update TASK-910-era round-trip fidelity tests (test_console_skill_install_confirm.py, test_skill_script_confirm_card.py) to pin the drop instead of the round-trip, mirroring TASK-1051's script-side test.
5. Run the specified pytest gates and reconcile failures against the documented pre-existing baseline.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stopped restoring pending_skill_install across screen navigation, mirroring TASK-1051's identical decision for pending_skill_script -- the two fields now go through byte-identical treatment in TaskResumeState.from_dict (both always None post-restore).

Why symmetric drop, not reconnection: re-verified TASK-1051's precedent chain still holds at this HEAD -- TldwCli._create_navigation_screen still builds a brand-new ChatScreen/ConsoleChatController on every navigation (never cached/re-mounted), so a skill-install confirm round (an entry in the OLD controller's _pending_skill_install_rounds dict, keyed by request_id and guarding a worker thread blocked on a threading.Event) cannot survive into the new controller. TASK-1143 (already merged, HEAD 84266ed88) makes this stronger, not weaker: its navigation guard now shows a confirm-before-leaving dialog when the fleet is busy and DENIES every in-flight/parked round on teardown regardless -- so a round captured in a save_state() snapshot is actively torn down before the snapshot could ever be restored, not merely orphaned. A real reconnection path would require rounds to survive controller teardown, which directly contradicts that deny-on-teardown architecture, so it was not attempted (matches the task's decision guidance).

Changes:
- tldw_chatbook/UI/Screens/chat_screen_state.py: from_dict now hardcodes pending_skill_install=None (previously restored via _payload("pending_skill_install")), with field-level comments on both pending_skill_install/pending_skill_script and an expanded from_dict docstring citing TASK-1051 + TASK-1130 and TASK-1143's teardown guard as the reason a reconnection path is architecturally excluded.
- Tests/UI/test_console_skill_install_confirm.py: replaced test_task_resume_state_pending_skill_install_roundtrip (asserted the round-trip survived) with test_task_resume_state_pending_skill_install_serializes_while_live (asserts to_dict() stays faithful while the field is live in-session, unchanged behavior) and test_restored_state_drops_the_pending_install_so_no_dead_card_appears (new; mirrors test_skill_script_confirm_card.py's script-side twin, asserts from_dict drops the payload).
- Tests/UI/test_skill_script_confirm_card.py: test_restored_state_drops_the_pending_script_so_no_dead_card_appears updated -- it previously asserted a sibling pending_skill_install payload SURVIVED restoration ("the asymmetry is scoped to the script card alone"); now asserts it is also dropped, since the asymmetry it was pinning no longer exists.

Consumer sweep (AC #3's "populated-but-never-read is fine, read-but-never-populated needs the branch removed" check): grepped for all `.pending_skill_install` attribute access outside chat_screen_state.py -- the only consumer is ChatTaskCards.sync_state (chat_task_cards.py:54), which is purely data-driven (`install_card.set_install(task_state.pending_skill_install)` / `task_state.has_pending_skill_install()`) with no branch that assumes a restored payload is present. `_set_console_pending_skill_install` (chat_screen.py) is the live-carrier mutation path, unaffected -- it doesn't go through from_dict. No half-dead branches found; nothing else to remove.

Verification: `.venv/bin/python -m pytest Tests/UI/test_skill_script_confirm_card.py Tests/UI/test_console_skill_install_confirm.py Tests/UI/test_skill_install_concurrent_confirms.py` -> 43 passed, 1 failed (test_set_console_pending_skill_script_preserves_other_resume_fields, AttributeError: 'ChatScreen' object has no attribute 'chat_state' -- documented pre-existing baseline failure per TASK-1051's own Implementation Notes, independently verified pre-existing on an earlier HEAD via git stash, unrelated to chat_screen_state.py). `.venv/bin/python -m pytest Tests/UI/test_console_parallel_runs.py Tests/UI/test_screen_navigation.py` -> 84 passed, 0 failed.
<!-- SECTION:NOTES:END -->
