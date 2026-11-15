---
id: TASK-41107
title: "After a post-restore plugin authority reset, the old packages/ and packages-revisions/ bytes stay unregistered and their space is never reclaimed"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- storage
priority: low
references:
- tldw_chatbook/Plugins/coordinator.py
- Docs/Plugins/operations.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The set-aside plugins' package files keep counting against plugin storage capacity, and the only way out is deleting folders by hand. TASK-40308's review (M11) recorded this as the reviewer's follow-up.

**Evidence.** `coordinator.py:141-163` (`_set_restored_plugins_aside`) moves `staging/` and `acquisition-custody/` into `reset-<id>/`; its docstring says "Package bytes under ``packages/`` and ``packages-revisions/`` stay too, unregistered, until removed by hand." `Docs/Plugins/operations.md:149-151` tells users the same.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The app offers a reviewed way to remove package bytes no registry names (showing their size first), or reset moves them aside with the rest
- [ ] #2 Plugin storage capacity stops counting them once removed or set aside
- [ ] #3 A test covers the reclaim, and the Plugins operations page is updated
<!-- AC:END -->
