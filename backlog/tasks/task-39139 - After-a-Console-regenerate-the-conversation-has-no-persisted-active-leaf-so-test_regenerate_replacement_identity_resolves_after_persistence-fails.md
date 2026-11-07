---
id: TASK-39139
title: "After a Console regenerate the conversation has no persisted active leaf, so test_regenerate_replacement_identity_resolves_after_persistence fails"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-07 06:33'
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_trajectory_capture.py
- tldw_chatbook/Chat/console_chat_store.py
- tldw_chatbook/DB/ChaChaNotes_DB.py
dependencies:
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-38111's notes (round-11 branch fu/r11-recovery) list it. Line numbers are dev's.

- `Tests/Chat/test_trajectory_capture.py::test_regenerate_replacement_identity_resolves_after_persistence` (`:523`) fails at `:555-556` with `assert None is not None`: after `regenerate_message`, `db.get_conversation_active_leaf` returns None.
- The trajectory projection then has no active-leaf anchor for the replacement's identity.
- The RecoveryRequired trap hid the node until TASK-38111. It is red on its own assertion alone, at base and on dev under a marker plugin.
- Whether regenerate fails to persist the cursor or the test is stale is not yet known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes record the cause as a product gap or as test drift, with the commit
- [x] #2 If it is a product gap: after a regenerate the persisted active leaf is the replacement, and the test passes; it is red before the fix
- [ ] #3 If it is test drift: the test is re-pinned without dropping its replacement-identity and ordering assertions
- [x] #4 The whole file passes 3 times at `-n 4`, with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Trace a regenerate's cursor writes on base: `create_sibling` writes the durable leaf while the new sibling has no persisted id yet, and find which persistence path gives the sibling its row and why it does not write the pointer through; find the commit.
2. Classify: product gap (the durable cursor does not follow the in-memory active leaf) or test drift.
3. Product gap: fix it at the shared persistence seam, so every path that gives the active leaf its persisted id writes the pointer; the existing node is the red-first test (red at base on its own assertion).
4. Run the whole file 3 times at `-n 4`, with the load recorded; run the covering store/controller tests.

## Implementation Notes

**Cause: product gap**, from 7970fbcada (2026-11-01, "feat: add literal chat prompt mix-ins on current dev").
- On a saved chat, recording a prompt mix-in attempt creates a not-yet-saved assistant row inside `ChatPersistenceService.save_conversation_mixin_attempt` (its `pending_assistant` branch). `ConsoleChatStore.publish_prompt_mixin_attempt` then adopts that row id, but it did not write the durable cursor through.
- A regenerate's new sibling is the only turn that reaches this branch with an unsaved assistant. `create_sibling` writes the cursor while the sibling has no row yet, so it writes `None`. After that, every persist of the sibling updates an existing row. `_persist_new_message`'s rule ("a newly saved message that is the active leaf writes the pointer") therefore never ran, and `conversations.active_leaf_message_id` stayed `None`.
- A probe on base: after two ordinary sends the durable leaf was the second assistant, because the durable turn commit creates both rows before the attempt. After a regenerate it was `None`, while the in-memory leaf was the saved sibling.

**Fix.** When `publish_prompt_mixin_attempt` adopts a row created by the attempt, it writes the cursor through if that message is the session's active leaf. This is the same rule `_persist_new_message` applies (`tldw_chatbook/Chat/console_chat_store.py`, +4 lines).

**Test.** `Tests/Chat/test_trajectory_capture.py::test_regenerate_replacement_identity_resolves_after_persistence` keeps all its replacement-identity and ordering assertions. It now also asserts that the durable leaf is the replacement itself (`replacement_event_id == f"message:{leaf}"`). Red at base and on this branch before the fix (`assert None is not None`), green after.

**Size waiver.** Owner decision (2026-11-06): a fix that pushes a module past its size ratchet raises the row to the measured size. `console_chat_store.py` 22,445 -> 22,449 in `Tests/Architecture/test_module_size_ratchet.py`, with a dated comment.

**Evidence** (run.sh, `-n 4`):
- Whole `test_trajectory_capture.py`: 10 runs of 32 passed (AC#4 asked for 3), load 46-71. These are TASK-39143's runs on the same tree.
- Covering set, run concurrently on head and base at load 33-42: 23 files (the prompt mix-in suites, regenerate branching, store tree, dispatch cursor recovery, edit/resend, turn resend, variant stream, fork and fork persistence, generation store, rewind summarize, trajectory, DB mix-ins, mix-in restore, and the three integration branching/edit-resend/fork flows). Head 75 failed / 753 passed; base 76 / 752. The only difference by node id is the fixed node. The 75 shared reds are pre-existing: mostly `RecoveryRequired` in files that carry no profile marker, plus own-assertion reds in `test_console_regenerate_branching.py` and `test_console_edit_resend.py` (listed under the report's proposed follow-ups).
- Module size ratchet: 201 passed.

AC#3 (re-pin as test drift) does not apply: the cause is a product gap, so it stays unticked.
