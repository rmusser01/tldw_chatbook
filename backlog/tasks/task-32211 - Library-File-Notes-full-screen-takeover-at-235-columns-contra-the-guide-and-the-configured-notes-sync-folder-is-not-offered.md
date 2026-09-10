---
id: TASK-32211
title: >-
  Library File Notes: full-screen takeover at 235 columns contra the guide, and
  the configured notes sync folder is not offered
status: To Do
assignee: []
created_date: '2026-09-10 14:52'
labels:
  - library
  - file-notes
  - layout
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 235 columns Notes ▸ Folder files replaced the whole Library frame with a full-bleed pane and a `‹ Library / Notes` breadcrumb (guide: 'At 120 columns and wider the rail stays beside it'); the pane said 'No folder selected' while `[notes] sync_directory` held three markdown files. MEASURED at dev 02374bf66a, before PR #2543 (fix/library-notes-file-notes) landed: re-verify on the current tip first; close if #2543 already fixed it. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Re-verified on the current dev tip; if still true, the rail stays beside Folder files at 120 columns and wider
- [ ] #2 When no File Notes root is set but `[notes] sync_directory` is, the pane offers it ('Use your notes sync folder (…, 3 files)') beside 'Choose folder…'
<!-- AC:END -->
