---
id: TASK-32178
title: >-
  Library Notes: Obsidian mode follow-ups for fences, resolved-link counts,
  Windows detection, and aliases
status: To Do
assignee: []
created_date: '2026-09-09 09:16'
updated_date: '2026-09-09 09:16'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - obsidian
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32129. Four follow-ups from the Obsidian-mode
review: an unterminated ``` fence lets the fence scanner keep treating the
rest of the file as code, so a `[[link]]` that follows it gets rewritten
when it should not; the import receipt has no "N links resolved" line even
though the durable-ledger schema already has room for it; there is no vault
detection through the Windows discovery adapter; and aliases are stored as
ordinary keywords, indistinguishable from real tags.

Update 2026-09-09 (PR #2549 review, integration): AC #1 is done — an
unterminated ``` or ~~~ opener now runs to end of file, so nothing after it
is recorded or rewritten as a link (`_CODE_SPAN` in
`note_import_plan_models.py`, pinned by
`test_an_unclosed_fence_keeps_the_rest_of_the_note_as_code` and
`test_a_closed_fence_still_ends_at_its_closer`). AC #3's "not supported on
Windows" half is now stated in the guide's "Obsidian vaults" section, so the
rider carries only the implementation choice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fence scanner handles an unterminated ``` fence without
  rewriting wikilinks past it
- [ ] #2 The import receipt reports a "N links resolved" count
- [ ] #3 Obsidian vault detection works through the Windows discovery
  adapter, or the review records an explicit "not supported on Windows"
  decision
- [ ] #4 Aliases are distinguishable from tags at the storage or display
  layer, or the guide records the decision to keep them merged
<!-- AC:END -->
