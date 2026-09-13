---
id: TASK-32264
title: >-
  Library Notes Folder files hides YAML frontmatter it is silently preserving,
  and neither the editor nor the guide says so
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 12:00'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified both ways by both assessors: the editor does not display a file's YAML frontmatter, and an in-app edit saved with the frontmatter intact on disk. So the app is hiding text it is faithfully preserving, and `Docs/User_Guide/library/file-notes.md` never mentions either half.

A user editing a vault file in Chatbook has no way to know whether their Obsidian properties survived the round trip -- and the answer is that they did.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The editor indicates that hidden frontmatter exists on a file that has it, and that it is preserved
- [x] #2 `Docs/User_Guide/library/file-notes.md` documents the behaviour
- [x] #3 Covered by a test asserting the indication for a file with frontmatter
<!-- AC:END -->


## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Confirm the split: `_parse_opened` moves the frontmatter into `preserved_prefix`, `_serialize_body` writes it back untouched.
2. RED tests: a service-level `frontmatter_lines`, and a workspace-level disclosure for a file that has frontmatter (and silence for one that does not).
3. Carry it on the editor's existing "what you are not seeing" line rather than a new widget.
4. Guide + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
`OpenedFileNote.frontmatter_lines` names the fact the editor needed: the
count of lines in the block `_parse_opened` split into `preserved_prefix`
and `_serialize_body` writes back byte-for-byte (BOM excluded; an
unterminated `---` block is not frontmatter and stays in the body).

The editor's existing "what you are not seeing" line --
`#file-notes-preview-status`, until now the large-file excerpt disclosure --
carries it, because that is the same question asked twice. Both disclosures
can show at once, joined by ` · `. No new widget, no new CSS.

Live GREEN: opening a vault daily note shows "4 lines of YAML frontmatter
above this body are hidden here and kept exactly as they are on disk."

Files: `Notes/file_notes_service.py`,
`Widgets/Library/library_file_notes_workspace.py`,
`Tests/Notes/test_file_notes_service.py`,
`Tests/UI/test_library_notes_wave_file_notes.py`,
`Docs/User_Guide/library/file-notes.md`.
<!-- SECTION:NOTES:END -->
