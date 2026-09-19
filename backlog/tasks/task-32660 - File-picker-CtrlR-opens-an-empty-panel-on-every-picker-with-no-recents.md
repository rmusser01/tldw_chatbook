---
id: TASK-32660
title: 'File picker: Ctrl+R opens an empty panel on every picker with no recents'
status: To Do
assignee: []
created_date: '2026-09-15 23:24'
labels:
  - library
  - notes
  - picker
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ctrl+R ("Show recent locations") opens a panel that is empty on every picker that has no recents to show, and says nothing about why. task-32606 made it a labelled footer chip, so it is now far more reachable than when it was an undiscoverable keystroke, and reaching it is currently rewarded with a blank box. Found during task-32643 review round 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening the recents panel with no recent locations shows a visible explanation in the list rather than an empty box
- [ ] #2 The panel still OPENS when empty -- refusing to open breaks the Escape-peel order Tests/UI/test_fspicker_keyboard_save.py pins over three transients
- [ ] #3 Focus does not move into an empty list, so a keyboard user is not stranded on a row that cannot be chosen
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shape is already in the tree: `Widgets/enhanced_file_picker.py`'s `EnhancedFileDialog` appends a "No recent files yet" placeholder row for exactly this case. Reusing that keeps the panel openable AND the peel contract intact.

History worth having: task-32643 tried the other fix -- refuse to open an empty panel -- and it broke `test_fspicker_keyboard_save`, which opens all three transients on an empty picker to pin the order Escape peels them in. That was correctly reverted. `watch_show_recent` already skips focusing an empty list, so AC#3 holds today and only needs a pin.

Related, same surface, also low: `FileSystemPickerScreen._get_recent_paths` runs up to 20 `validate_existing_absolute_directory` stats on the event loop from `on_mount`. Local disk is microseconds; a stale network-mount entry would stall the picker opening. No cheap fix that does not reshape the panel's lifecycle (the list is populated at mount, and a pin reads it before ctrl+r is pressed), so it was left alone deliberately rather than missed.
<!-- SECTION:NOTES:END -->
