---
id: TASK-32516
title: "Library Folder files: the delete-receipt dismissal race (NoMatches '#file-notes-authority')"
status: To Do
assignee: []
created_date: '2026-09-13 00:13'
updated_date: '2026-09-13 03:40'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_notes_wave_editor_keys.py::test_delete_receipt_is_dismissed_leaving_the_list_for_folder_files`
is a standing race: on the wave-3 docs-sweep tree (dev 7159fc0b99 merged,
docs-only delta) it was red once and green on the immediate re-run with no
code change between; the T11 landing pass reported the same node failing
when run in isolation on dev. No run log was kept for either — the two
observations above are the whole record. The failure is a Textual
`NoMatches` for
`#file-notes-authority` raised from `library_file_notes_workspace.py`:
dismissing the Folder files delete receipt queries the authority line while
the workspace is being torn down for the return to the list, so the query
lands on a widget that has already been removed. The dismissal-then-leave
sequence is what the test exercises on purpose; the race is in the product,
not in the test's timing.

Evidence: the two runs above (wave-3 docs sweep, 2026-09-12, dev
7159fc0b99; T11 landing pass, PR #2654). AC#2 asks for the count a fix has
to beat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dismissing a Folder files delete receipt while the workspace is leaving for the list never raises NoMatches for the authority line, at any timing
- [ ] #2 The isolated test passes 10 consecutive runs on the fixed tree
- [ ] #3 The receipt still dismisses and the list is still shown afterwards, with a test pinning both
<!-- AC:END -->
