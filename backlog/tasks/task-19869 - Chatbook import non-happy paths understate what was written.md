---
id: TASK-19869
title: >-
  Chatbook import non-happy paths understate what was written
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - ux
  - honesty
  - chatbooks
priority: medium
dependencies:
  - TASK-279
  - TASK-19734
---

## Description

Source: surfaced by the reviewer of **TASK-19734**, which corrected the
chatbook import wizard's *successful* paths. These are the two remaining paths
where the wizard still tells the user less than it knows. Re-verified at
`3605bd52d`.

**1. A fatal error hides a completed write.**

When the import worker raises, the handler at
`UI/Wizards/ChatbookImportWizard.py:1116-1119` calls `_show_error` (`:1203`),
which updates only the finalize row and retitles the panel "❌ Import Failed".
The per-type rows that had already been set to `⟳ Preparing notes…` /
`⟳ Preparing characters…` / `⟳ Preparing media…` (`:1065-1073`) are never
updated again and freeze there, and **no stats panel renders at all**.

Meanwhile the items the run had already written are permanently in the
database. There is no rollback (TASK-279). So the screen reads as "nothing
happened, it failed while preparing", and the truth is "an unknown number of
items were committed and will still be there when you retry". A user who then
retries the import gets duplicates or conflicts they were given no reason to
expect.

**2. Server mode reports a 4-item chatbook as empty.**

At `:1017-1028` the server path builds a fresh `ImportStatus` from the job
payload:

- `successful_items` defaults to `0` when the key is absent
- `failed_items` defaults to `0`
- `total_items` is their sum
- `skipped_items` is **hard-set to 0**, regardless of what the server did

A payload that omits `successful_items` therefore yields `attempted_items == 0`
with `left_out_items == 0`, which `ImportStatus.outcome`
(`chatbook_importer.py:293-309`) resolves to `IMPORT_OUTCOME_EMPTY`, and the
panel says:

> ⊘ Nothing was imported — this chatbook contained no items.

…about a chatbook whose four-item manifest the wizard is holding in memory at
that moment. The hard-set `skipped_items = 0` is a second, independent false
claim: it is not something the server reported, it is a value the client
invented.

Both belong to the family TASK-19550, TASK-19734 and TASK-19861 are in — *the
app asserts an outcome it did not produce* — and both are on the paths where a
user most needs the truth, because both are paths where they are about to
decide whether to retry.

## Acceptance Criteria

- [ ] When the import fails partway through, the completion surface states that
      items may already have been written and cannot be rolled back
- [ ] Per-type rows do not remain frozen on a "preparing" state after a fatal
      error — each row says what is known about that type, including "unknown"
- [ ] The wizard never reports "this chatbook contained no items" for a
      chatbook whose manifest it holds and which lists items
- [ ] Server mode does not invent a `skipped_items` value the server did not
      report; an unreported count is presented as unknown, not as zero
- [ ] A server payload missing `successful_items` produces a message that
      distinguishes "the server told us nothing" from "there was nothing to
      import"
- [ ] Tests drive both paths — a fatal error raised after at least one item is
      committed, and a server payload with no counts — and assert the rendered
      text; both are mutation-checked against the current behaviour
- [ ] The one-vocabulary rule TASK-19734 established (empty = a claim about the
      file, excluded = a claim about the run) is extended to cover these two
      cases rather than bypassed

## Notes

Filed as one task because both are the same surface's remaining non-happy-path
claims and a fix for either will touch the same completion-panel code. The
rollback gap itself is TASK-279 and is not in scope here — what is in scope is
that the UI currently conceals it at exactly the moment it matters.
