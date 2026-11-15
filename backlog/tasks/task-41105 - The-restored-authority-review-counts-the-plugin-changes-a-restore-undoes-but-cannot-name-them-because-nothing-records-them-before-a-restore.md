---
id: TASK-41105
title: "The restored-authority review counts the plugin changes a restore undoes but cannot name them, because nothing records them before a restore"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- backup
- ux
priority: low
references:
- tldw_chatbook/Plugins/service.py
- tldw_chatbook/Widgets/plugin_review_modal.py
- Docs/Plugins/operations.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A user adopting a restored plugin authority is told "This undoes N plugin changes made after this backup … A restore keeps no record of what each one was." To judge the restore they need to know which installs, updates, configuration and enable changes are lost. TASK-40308's owner ruling accepted a count for now; naming them needs new state.

**Evidence.** `Plugins/service.py:660-681` computes `undone` as `current.generation - marker.generation` (None when the marker is missing or foreign); `plugin_review_modal.py:423` says no record is kept; `Docs/Plugins/operations.md:119-124` documents the count. TASK-40308's notes: "Listing each change by name would need a record written before the restore, which the plugin owner has no hook for."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Before a restore replaces plugin state, the plugin owner keeps a list of the authority changes made since the backup (kind, plugin, generation), outside the restored files
- [ ] #2 The restored-authority review lists those changes by name when the list is available and falls back to the count otherwise
- [ ] #3 A test restores over two named changes and checks the review lists both
<!-- AC:END -->
