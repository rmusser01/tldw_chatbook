---
id: TASK-32622
title: >-
  Library Notes: import receipt and picker nits — no way forward, an ambiguous
  link count, a wrong entry count, a front-truncated row
status: To Do
assignee: []
created_date: '2026-09-15 06:43'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D16, D13, D14, personas Jordan and Alex, Obsidian workflow. Four small things on the strongest screen in the product.

1. The receipt after a 54-note import offers only a collapsible Skipped disclosure and 'esc back to notes' -- no 'View imported notes' action after the biggest thing the user has done all session (B cap 25).
2. '54 links resolved' is ambiguous: the review itself counted about 9 wikilinks across 9 notes, so the number is either counting something else or counting it twice (B cap 25). A independently checked the vault and found 57 bracketed links, so the number may be honest and is certainly unexplained (A cap 22).
3. The Import-once picker reports 'Loaded · 17 entries' while displaying 14 entries plus the parent row: the three hidden dot-entries are counted but not shown (B cap 20).
4. A long file name is FRONT-truncated in the review table, so the row reads as a tail fragment ending in .md and its 'vault/Inbox/' prefix is gone -- the one thing that would let the reader find it in the tree, while every other row in the same table carries its folder (B cap 23).

Cause PROVEN by capture for all four. Wave 4's import copy work (task-32554, PR #2685) covered a different set of nits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The import receipt offers a way to the notes it created
- [ ] #2 The links figure says what it counted, or is removed
- [ ] #3 The picker's entry count matches what it displays, or says what it excludes
- [ ] #4 Long row labels keep their folder prefix and truncate where the information is least
<!-- AC:END -->
