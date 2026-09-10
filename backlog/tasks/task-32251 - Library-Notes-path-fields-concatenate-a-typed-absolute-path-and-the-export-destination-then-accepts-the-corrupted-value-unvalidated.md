---
id: TASK-32251
title: >-
  Library Notes path fields concatenate a typed absolute path and the export
  destination then accepts the corrupted value unvalidated
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - pickers
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cause PROVEN: `enhanced_file_picker.py:2547,2563-2565` `_sync_dir_path_input` writes `str(nav.location)` into `#dir-path-input` on mount and on every navigation, and nothing selects the value on focus. The identical lines are present at `c4a7b1911f` -- pre-existing, newly found; #2542 added the persistent "Folder path" label, which is what made the field visible enough to find.

Reproduced exactly. Clicking the Folder-files picker's field and typing an absolute path leaves `/Users/macbook-dev/private/tmp/claude-501/...` in the field (`R/caps/20`), and the resulting error is rendered **inside the dialog's bottom border** -- `+-- Path not found: ... -+` (`R/caps/21`) -- where nobody looks. On the export destination the same concatenation is accepted silently: the canvas showed `.../exports/Library export 2026-09-10.zip/private/tmp/.../notes-bundle.zip` as the chosen destination with no warning (`R/caps/44`).

Correction to the design assessor's account: running it does **not** create the bogus directory tree. The export fails at write time with a raw Python error surfaced verbatim -- `Error creating chatbook: [Errno 2] No such file or directory: '.../notes-bundle.zip.partial'` -- and nothing lands on disk (`R/caps/45`; `ls "$HOME/Library export 2026-09-10.zip"` -> No such file). So the defect is not data damage: it is that validation is deferred to an OSError repr with an internal noun in it. The vault path had to be typed three times because of this.

Scope against the peer critique, so nobody folds these together: peer task-32229 covers ONLY a focused path Input accepting a *pasted* absolute path and jumping the tree, plus `Ctrl+A` in the file-name box being move-to-start. It does **not** cover pre-fill concatenation, and it does **not** cover the export destination accepting a corrupted value. Those two are this task.

Fix: select-all on focus for any path/name Input that arrives pre-filled -- one change at the picker level, which is also the most on-brand fix available, since this is a terminal app whose path fields behave worse than a shell prompt; move the error to an inline row under the field; validate the destination when it is chosen, not when the zip is opened.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A path Input that arrives pre-filled selects its content on focus, so typing an absolute path replaces the pre-fill instead of concatenating onto it
- [ ] #2 An invalid path reports its reason on an inline row under the field, never painted into the dialog border
- [ ] #3 The export destination is validated when it is chosen: a destination whose parent is not an existing directory is refused there, with a reason
- [ ] #4 An export that still fails at write time reports a user-facing reason rather than an errno repr naming a `.partial` file
- [ ] #5 The configured `[notes] sync_directory` is offered as a start location by both pickers where it exists
- [ ] #6 Covered by a test for pre-fill select-on-focus and a test for destination validation at choose time
<!-- AC:END -->
