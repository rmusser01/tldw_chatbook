---
id: TASK-22060
title: Console durable dispatch forks the persisted conversation tree
status: In Progress
assignee:
  - '@claude'
labels:
  - console
  - data-integrity
  - regression
priority: high
---

## Description

Every accepted Console turn after the first was persisted as a NEW root in the
`messages` tree instead of extending the conversation's chain. The transcript
still rendered correctly, so the damage is invisible in the UI and lives only in
the durable tree that reload, rewind, branching, active-leaf resume and
trajectory export all walk.

Introduced by `a26cdafd8` ("fix(console): resume Library-gated sends"), which
added the durable dispatch checkpoint. Bisected against `10361e2ad`
(2026-08-15), which is green.

## Acceptance Criteria

- [x] An ordinary second send in one Console visit parents to the conversation's leaf
- [x] A send after navigating away from and back to Console parents to the leaf
- [x] Regression cover exists that fails on the pre-fix product code
- [x] No regressions across the surrounding Console suites

## Implementation Plan

1. Reproduce with the smallest possible case (second send, no navigation)
2. Bisect to the introducing commit
3. Fix at the parent resolution, not at the call site
4. A/B the surrounding Console suites against clean dev

## Implementation Notes

`console_dispatch_repository.insert_with_messages` writes its USER and assistant
rows with raw SQL and threads `acceptance.parent_message_id`. The controller
computed that from `echoed_user.parent_message_id` — but that field holds the
*persisted* parent id and `ConsoleChatStore._persist_new_message` is its only
writer. The optimistic echo is appended with `persist=False`, so it never went
through that method and the field was ALWAYS `None`; the guard
`if echoed_user.parent_message_id is not None:` therefore never ran and the row
was written as a root.

Fix: added `ConsoleChatStore.durable_parent_for_message`, which returns
`(has_native_parent, nearest_persisted_ancestor_id)` from the store's own tree
bookkeeping — the same `_nearest_persisted_ancestor_id` walk `_persist_new_message`
uses, so the two persistence paths now agree by construction rather than by
coincidence. The controller consumes that, and the existing "parent should be
persisted but isn't" pause is now keyed on `has_native_parent`.

Measured on dev `8ef5bf12e` with no product edits: the second send persisted
`parent_message_id=None` with the real leaf one row above it.

Modified: `tldw_chatbook/Chat/console_chat_store.py`,
`tldw_chatbook/Chat/console_chat_controller.py`,
`Tests/UI/test_console_persisted_chain.py` (new).
