---
id: TASK-32587
title: >-
  Library Notes: the Obsidian toggle for a synced root is process-lived and
  needs device-schema v3 to persist
status: To Do
assignee: []
created_date: '2026-09-14 22:48'
labels:
  - library
  - notes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recorded by wave-4 group 2 with the evidence that blocked it. Turning the Obsidian vault toggle OFF for Keep a folder synced lasts only until you quit: the choice is not stored, so the vault is offered the toggle again, on, at the next start, and a later check of that root skips .obsidian/, .trash/ and Templates/ again. Storing it in the existing settings table is not available — setting_key is CHECK-constrained to a literal whitelist AND to NOT GLOB '*[^a-z0-9_]*', which the natural obsidian_mode:<root_id> key violates on the colon; one attempted path also hung for 300 s. Durable persistence needs a device-schema v3 field. The UI and the guide were made honest about this in the meantime and are pinned so the honesty cannot be tidied away.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The per-root Obsidian choice survives a restart
- [ ] #2 It is stored in the device sync store, not in the settings table whose key constraint rejects it
- [ ] #3 The schema change follows the migration rules in DB/migrations/README.md, including any index plan pin
- [ ] #4 The honest copy on the canvas blurb and in notes.md is replaced with what ships, and its pin updated rather than deleted
<!-- AC:END -->
