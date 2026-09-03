---
id: TASK-28022
title: Command palette - near-duplicate Library entries invite off-by-one selection
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 21:08'
labels:
  - ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Querying library yields four near-identical entries (Tab Navigation: Switch to Library; Tab Navigation: Library - Skills; Media and Content: Open Media Library; Library: Import). The 2026-09-01 live UX run's arrow-count selection landed on the wrong entry. Dedupe, reorder, or differentiate so the most common destination is the obvious first pick.

Also from the 2026-09-02 run: no palette command reaches the media Trash view at all (a "trash" query returns only tab-navigation fuzz).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A library query presents visually distinct, deduplicated choices
- [ ] #2 The most common destination ranks first
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RECON (not started — cross-provider palette curation): the near-duplicate entries come from 3 providers in app.py: TabNavigationProvider ('Tab Navigation: Switch to Library' + LIBRARY_SUBROUTE_COMMANDS 'Library — Skills'), MediaProvider ('Media & Content: Open Media Library'->TAB_MEDIA, 'Import New Media'->TAB_INGEST, 'Search Transcripts'->TAB_SEARCH), LibraryIngestProvider ('Library: Import…'->ingest). Redundancy: 'Import New Media' and 'Library: Import…' both open the import experience. AC#2 ('most common destination ranks first') needs cross-provider score influence (palette merges hits by score). The second concern ('no palette command reaches media Trash') is a NEW command to ADD, not dedup. Labels are PINNED by Tests/UI/test_command_palette_providers.py (503 'Switch to Library' fuzzy-match, 796-797 MediaProvider labels, 850/861 'Library: Import…'). Own PR.
<!-- SECTION:NOTES:END -->
