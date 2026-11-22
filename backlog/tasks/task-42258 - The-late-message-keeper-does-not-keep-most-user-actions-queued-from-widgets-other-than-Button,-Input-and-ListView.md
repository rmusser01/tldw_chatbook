---
id: TASK-42258
title: "The late-message keeper does not keep most user actions queued from widgets other than Button, Input and ListView"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- ui
- textual
priority: low
references:
- tldw_chatbook/__init__.py
- backlog/tasks/task-42150 - Queued-messages-other-than-a-button-press-are-still-dropped-when-a-container-removes-their-sender.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the recompose stream (TASK-42150, round 19; code read on `fu/r19-recompose`). The keeper in `tldw_chatbook/__init__.py` (`_kept_messages`, line 509) keeps only `Button.Pressed`, `Input.Submitted` and `ListView.Selected`, from `Button`, `Input` and `ListView`. A queued message of another type that the app handles above its sender is dropped when the sender's subtree is removed: `RadioSet.Changed`, `OptionList.OptionSelected`, DataTable selection, Tree node selection, `SelectionList` changes, `Collapsible` toggles, and app-defined messages a child posts. `Select.Changed`, `Checkbox.Changed` and `Switch.Changed` are left out on purpose, because a mount echo or a programmatic `value =` posts the same message; keeping them needs a signal that the user, not code, caused the change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each widget message type listed above is kept, or left out with its reason recorded in the keeper
- [ ] #2 `Select.Changed`, `Checkbox.Changed` and `Switch.Changed` are kept only for a user-origin change, with the origin signal named, and a programmatic or mount-echo change is still not kept
- [ ] #3 A test per kept type posts the message just before its sender's subtree is removed and shows its handler runs once; a test shows a programmatic change is not forwarded
<!-- AC:END -->
