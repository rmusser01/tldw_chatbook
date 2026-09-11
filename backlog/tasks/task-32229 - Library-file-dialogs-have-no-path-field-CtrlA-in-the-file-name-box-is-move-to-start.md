---
id: TASK-32229
title: >-
  Library file dialogs have no path field; Ctrl+A in the file-name box is
  move-to-start
status: Done
assignee: []
created_date: '2026-09-10 14:56'
updated_date: '2026-09-10 19:02'
labels:
  - library
  - export
  - import
  - ux
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The export destination picker (and Import Browse…, Folder files) makes a terminal user click through a tree from $HOME; the only way to type a path is the File name box. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 28.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused path Input above the tree accepts a pasted absolute path (with ~) and jumps the tree to it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. The existing File name Input follows an absolute/~ directory path as it is typed (reusing resolve_typed_directory)
2. Ctrl+A selects the field instead of moving to start
3. Tests + docs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended the field that already exists rather than adding a second one:
`BaseFileDialog`'s "File name" Input (the one `task-32122` taught to
relabel itself "Folder path:") now navigates the listing as you type. On
`Input.Changed`, a value that is absolute or starts with `~` goes through
the same `resolve_typed_directory` "Select folder" already used, and the
`DirectoryNavigation` location follows it. Relative values are deliberately
left alone: they are file names, and one of them is the basename
`_select_file` pre-fills on a click for "Select folder" to read -- chasing
those would move the ground under the value the user just picked.

Ctrl+A needed a three-line `Input` subclass (`FileNameInput`): Textual maps
`ctrl+a` to "go to start" on `Input` itself, and the focused widget's own
bindings beat the screen's, so no screen-level binding can win. The
placeholder now reads "File name or path" -- the only in-app hint that the
field takes one.

This is the base class every FileOpen/FileSave in the app shares (Library
export destination, Import Browse…), which is why the fix belongs there.
`SelectDirectory` is NOT a `BaseFileDialog` and is untouched.

Tests: 4 in `Tests/UI/test_library_crit9_grammar.py` (absolute path, `~`
expansion, Ctrl+A, and the negative case that a bare file name does not
move the tree). Ran the 10 fspicker/dialog suites before and after: 138
passed with the change; the 2-3 `SelectDirectory` failures seen in some
runs reproduce on unpatched dev and shift names between runs (host under
load). Live-verified in the export destination picker: typed
`/Users/macbook-dev/Documents` and `~/Downloads`, breadcrumb followed both;
Ctrl+A replaced the pre-filled bundle name.
**Fix round 1.** The new handler's `except Exception` around `query_one`
is now `except NoMatches`; rooted-ness is tested on the stripped value
while the RAW value goes to `resolve_typed_directory`, whose docstring
makes leading/trailing spaces significant (a directory whose name ends in
a space was unreachable); and the vendored package's own ledger
(`Third_Party/textual_fspicker/ENHANCEMENTS.md`) gains section 7 covering
this change and task-32122's label flip, which was also missing.
<!-- SECTION:NOTES:END -->
