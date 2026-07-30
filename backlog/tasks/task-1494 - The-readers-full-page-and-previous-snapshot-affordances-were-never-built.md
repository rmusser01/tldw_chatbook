---
id: TASK-1494
title: The reader's full page and previous snapshot affordances were never built
status: To Do
assignee: []
created_date: '2026-07-30 13:30'
labels:
  - watchlists
  - spec-divergence
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Watchlists design spec's Content-pane mockup
(`Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`, "Content pane") promises
`[full page]` and `[previous snapshot]` affordances on a change item, reading from `url_snapshots`.
Phase D shipped both renderers without those buttons, and no reference to either exists anywhere in
the UI (verified by grep during TASK-1393). Since TASK-1343 made `content` the diff, the reader
shows *what changed* but offers no way to see the page it changed on — the data is stored and
unreachable.

The storage side is ready and deliberately protected: snapshots are per-(subscription, url) with
the newest 3 kept (TASK-1393, `_SNAPSHOTS_KEPT_PER_URL` — slot 2 exists specifically for this
affordance), and the `[previous snapshot]` needs the second-newest row for the item's URL.

Do not render `raw_html` as markup or hyperlinks — remote content; the reader's escaping decisions
are documented in `content_pane.py` and TASK-1348.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A change item in the reader offers [full page] and [previous snapshot], reading the newest and second-newest url_snapshots rows for the item's URL
- [ ] #2 When no previous snapshot exists (first check, or pruned by cap), the affordance degrades honestly rather than erroring or hiding silently
- [ ] #3 Remote snapshot content renders as text, never as markup or live hyperlinks
<!-- AC:END -->
