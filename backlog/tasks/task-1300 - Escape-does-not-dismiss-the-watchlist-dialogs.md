---
id: TASK-1300
title: >-
  Escape does not dismiss the watchlist dialogs — only clicking Cancel does
status: To Do
assignee: []
created_date: '2026-07-28 04:00'
labels:
  - watchlists
  - bug
  - ui
  - a11y
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing `Escape` in the Rename-watchlist dialog leaves it open. Verified in the third Watchlists UAT on `origin/dev` `e82ac1b18`: `Escape` sent twice, the dialog still present both times; clicking `Cancel` dismissed it immediately.

```
dialog present after Escape: 1
dialog present after Cancel: 0
```

So a keyboard user cannot back out of a modal, and — because it is modal — cannot do anything else either. Every click goes to the dialog until the mouse finds `Cancel`. During the UAT a `Delete` click was silently swallowed by the still-open Rename dialog, which is exactly how this presents in normal use: the app appears to ignore you.

`Escape`-to-dismiss is the standard expectation for a modal, and the other dialogs on this screen (New watchlist, Add source, Delete confirmation) should be checked for the same gap rather than fixed one at a time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Escape` dismisses the Rename dialog without applying the change
- [ ] #2 The same holds for New watchlist, Add source, and the Delete confirmation
- [ ] #3 Dismissing by `Escape` leaves the watchlist untouched, exactly as `Cancel` does
- [ ] #4 A test presses `Escape` on each dialog and asserts it closed, proven to fail against current code
<!-- AC:END -->
