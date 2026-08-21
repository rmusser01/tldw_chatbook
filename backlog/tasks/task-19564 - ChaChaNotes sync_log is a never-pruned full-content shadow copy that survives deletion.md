---
id: TASK-19564
title: >-
  ChaChaNotes sync_log is a never-pruned full-content shadow copy that survives
  deletion
status: To Do
assignee: []
created_date: '2026-08-21 20:14'
labels:
  - db
  - privacy
  - retention
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F5**, CONFIRMED LIVE by probe. Re-verified at this branch base.

The ChaChaNotes `sync_log` table is a **write-only, never-pruned, full-content
shadow copy** of the user's data:

- 35 triggers write the **complete row as JSON** into it.
- Both of its readers have **zero external callers** — nothing consumes it.
- There is **no `delete_sync_log_entries_before`** and **no `DELETE FROM
  sync_log`** anywhere for this database. (`Client_Media_DB_v2.py:2671,2690`
  and `Prompts_DB.py:4268` show the prune functions the Media and Prompts
  databases *do* have — ChaChaNotes simply never got one.)

The lane's probe: it soft-deleted a conversation, then found **the message body
still present in `sync_log`**.

Two consequences:

1. **Privacy.** "Delete" leaves the user's plaintext in the database
   indefinitely. A user who deletes a conversation has every reason to believe
   the content is gone; it is not. This is the same class of expectation-gap as
   the soft-delete search leak in TASK-19566.
2. **Size.** It roughly doubles on-disk size for message text, permanently.

## Acceptance Criteria

- [ ] Deleting a conversation, message, note or character removes the
      corresponding content from `sync_log`, or `sync_log` stops storing full
      content in the first place
- [ ] A pruning path exists for ChaChaNotes `sync_log`, matching what
      `Client_Media_DB_v2` already provides, and it actually runs rather than
      merely existing
- [ ] The retention decision is explicit and recorded: if the log is genuinely
      unused (both readers have no external callers), retiring the content
      columns is the durable answer — do not keep writing a shadow copy nobody
      reads
- [ ] A test reproduces the lane's probe: after a soft delete, the deleted
      content is not retrievable from `sync_log`
- [ ] Existing users' databases are addressed — a fix that only helps new
      installs leaves the plaintext in place for everyone who already has it
- [ ] The on-disk size effect is measured before and after on a realistic
      database
