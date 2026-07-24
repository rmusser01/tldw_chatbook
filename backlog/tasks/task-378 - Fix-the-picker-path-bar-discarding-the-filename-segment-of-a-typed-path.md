---
id: TASK-378
title: Fix the picker path bar discarding the filename segment of a typed path
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ctrl+L opens a path input (good), but entering an absolute FILE path and pressing Enter/Go only navigates to the parent directory and resets the list highlight to '..' - the file is neither selected nor attached, with no message about the dropped filename. A natural second Enter then activates '..' and navigates UP a level. My scripted attaches repeatedly ended stranded in the wrong directory this way - a keyboard user pasting a known path cannot attach without re-finding the file by eye.

**Repro:** Composer Attach -> Ctrl+L -> type /full/path/to/test-image.png -> Enter. Listing shows the parent dir with '..' highlighted; nothing attached; Enter again navigates up.

**Verifier note:** Code-confirmed with a literal TODO: Third_Party/textual_fspicker/base_dialog.py:526-531 — a file path entered in the Ctrl+L bar navigates to path.parent with '# TODO: Ideally, we would also select the file in the list'. Known to the code author, never filed (task-219 was a different FileOpen bug). Downgraded P2→P3: the bottom filename input DOES accept full/~-expanded paths (_confirm_single, enhanced_file_picker.py:1270-1296), so a keyboard workaround exists; affects power users of one picker surface.

**Source:** Console UX expert review 2026-07-20 (finding j3-picker-path-entry-drops-filename; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-p1-after-enter1.png`, `j3-cap4-sixth-ctrl-l.png`, `j3-cap4-sixth-stuck.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A full file path entered in the path bar should select/attach that file (standard file-dialog behavior), or at minimum highlight it in the list and explain
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`DirectoryNavigation` gained `show_and_highlight(target)`: it navigates to the
file's parent directory and, once the async directory load repopulates the list,
highlights the file instead of leaving the cursor on `..`. Implemented via a
`_pending_highlight` target honored in `_repopulate_display` (before the default
top-of-list settle), with an immediate path when the directory is already shown.
`base_dialog.py`'s Ctrl+L path-bar handler now calls it for file paths, resolving
the long-standing `# TODO: Ideally, we would also select the file in the list`.
A keyboard user pasting a full path now lands on the file. RED->GREEN test
(verified: neutering the highlight leaves the cursor on `..`); 29 existing picker
tests green.
<!-- SECTION:NOTES:END -->
