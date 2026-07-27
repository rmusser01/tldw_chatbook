---
id: TASK-1091
title: >-
  Watchlist names keep leading whitespace, and the delete copy says "1 source are"
status: To Do
assignee: []
created_date: '2026-07-28 04:00'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two small copy/validation defects seen in the third Watchlists UAT (`origin/dev` `e82ac1b18`).

**Names are not trimmed.** Renaming a watchlist to `" Daily"` (leading space) stored it verbatim, and the tree renders the space as extra indentation, so the row no longer lines up with its siblings:

```
│ ▸     Daily  0           │
│ ▸  Security Watch  0     │
```

Create has the same gap. A name that is entirely whitespace is presumably also accepted, which would produce an unclickable, unnameable row.

**The delete confirmation mispluralises.** With one source attached it reads:

> Its **1 source are** not deleted. They stay in Watchlists and appear under Unassigned unless they also belong to another watchlist.

Should be "is" for one. The rest of that copy is genuinely good — it explains the consequence clearly, which is why the grammar stands out.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leading and trailing whitespace is stripped from a watchlist name on create and on rename
- [ ] #2 A name that is empty or whitespace-only is rejected with a visible reason, not silently accepted
- [ ] #3 The delete confirmation reads correctly for one source and for several
- [ ] #4 Tests cover the whitespace-only name and the single-source wording, proven to fail against current code
<!-- AC:END -->
