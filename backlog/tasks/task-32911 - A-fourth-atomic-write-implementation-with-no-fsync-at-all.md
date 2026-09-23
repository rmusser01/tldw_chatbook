---
id: TASK-32911
title: A fourth atomic-write implementation with no fsync at all
status: To Do
assignee: []
created_date: '2026-09-22 03:00'
labels:
  - tier2-review
  - review-helpers
dependencies: []
parent_task_id: TASK-32896
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while hardening the shared helper (TASK-32896). The review catalogued **three** atomic-write
implementations and ranked them. There is a **fourth**: `tldw_chatbook/Skills_Interop/atomic_write.py`,
which has **zero `fsync` calls** — independently confirmed (`grep -c fsync` -> 0).

That makes it the weakest of the four on durability: it is `write` + `os.replace` with nothing forcing the
bytes or the rename to disk. It is, however, **stronger than the hardened helper in one respect** — it
precreates `owner_only`, which the shared helper does not express.

So this is not a straight "adopt the helper" conversion. Doing that naively would trade a real
confidentiality property for a durability one. Either the hardened helper grows an `owner_only` precreate
mode, or this site keeps its open path and gains the barriers.

## Two adjacent one-line upgrades found at the same time

`Utils/private_paths.py`, `Notes/sync_paths.py` and `MCP/permission_store.py` all already fsync **both** the
file and the parent directory, but via plain `os.fsync` — so on Darwin the write can still sit in the
drive's cache. Swapping to the new `Utils/file_durability.flush_file` gives them the `F_FULLFSYNC` barrier
for one line each. `private_paths` does careful platform-capability probing, so read it before touching it.

## Why this keeps happening

Worth recording, because it is the mechanism rather than the defect. `MCP/permission_store.py:1010` carried
a comment citing TASK-32808.5 asserting that it was the only writer fsyncing both the file and its parent
directory. That statement was true when written and is now false — and a comment like it is exactly how a
weakness survives a review: the next person greps, finds a confident note saying the question is settled, and
moves on. It has been de-fossilised as part of TASK-32896.

Prefer a test that asserts the property over a comment that claims it.

Source: tier-2 code review 2026-09-21, found while implementing TASK-32896.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Skills_Interop/atomic_write.py` fsyncs the file and the parent directory
- [ ] #2 Its `owner_only` precreate property is preserved, in the shared helper or in place
- [ ] #3 A test asserts the parent-directory fsync, not just the file fsync
- [ ] #4 The three plain-`os.fsync` sites are either upgraded to the Darwin barrier or recorded as deliberate
<!-- AC:END -->
