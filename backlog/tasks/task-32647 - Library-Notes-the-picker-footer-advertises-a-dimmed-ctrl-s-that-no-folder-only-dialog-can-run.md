---
id: TASK-32647
title: >-
  Library Notes: the picker footer advertises a dimmed ctrl+s that no folder-only dialog can run
status: To Do
assignee: []
created_date: '2026-09-15 17:05'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from task-32606. `FileSystemPickerScreen.check_action` vetoes
`select_current_folder` on every dialog that does not set
`_offer_select_folder`, and task-2222's comment says that removes the key
"from both the key map and the listing". Textual 8's `Footer` does not honour
that: it composes every `show=True` binding and renders a vetoed one DIM
rather than dropping it. Since task-32606 put a footer on every vendored
picker, `^s Select this folder` is now visible, dimmed and dead, on
`SelectDirectory`, `FileSave` and plain `FileOpen` -- and on the Folder-files
dialog it names the exact action the user came to perform.

task-32606 mitigated the crowding by ordering that binding LAST, so a narrow
terminal scrolls it off first. It did not suppress the chip.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The picker's footer does not advertise a key the focused dialog cannot run
- [ ] #2 Suppressing it does not remove the key from F1 help for dialogs that DO offer it
- [ ] #3 A test pins that no disabled FooterKey is rendered on SelectDirectory
<!-- AC:END -->
