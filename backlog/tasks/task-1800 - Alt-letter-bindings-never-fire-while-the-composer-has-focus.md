---
id: TASK-1800
title: 'Alt+<letter> bindings never fire while the composer has focus'
status: To Do
assignee: []
created_date: '2026-08-01 13:20'
labels:
  - console
  - ux
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of the temporary-conversations work (2026-08-01) found that `Alt+T` did not open a new temporary tab when the composer had focus. It inserted a literal `t` into the draft instead.

The cause is app-wide, not specific to that binding: Textual 8.2.7 reports `Key("alt+t", "t").is_printable == True`, so a focused `Input` consumes the key before the screen-level binding runs. `Alt+M` (model popover) reproduces the failure identically, and `Alt+W` (workspace switcher) shares the same shape.

This affects every `alt+<letter>` binding on `ChatScreen`, and the composer is where a chat user's focus almost always is — so these chords are effectively unavailable in normal use while still being advertised in the footer and command palette. `Alt+T` was removed rather than shipped broken (PR for temporary conversations); `Alt+M` and `Alt+W` still ship and still fail.

Two chords that type a stray character into the user's draft is worse than two chords that do not exist, because the failure is silent and corrupts input.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Alt+M` opens the model popover while the composer has focus, or the binding is removed and every surface advertising it is updated
- [ ] #2 `Alt+W` opens the workspace switcher while the composer has focus, or the binding is removed and every surface advertising it is updated
- [ ] #3 No `alt+<letter>` chord inserts a literal character into the composer draft
- [ ] #4 Whichever route is taken, the behaviour is verified live in a real terminal, not only by unit test — the defect is invisible to the test suite
- [ ] #5 A regression test pins the outcome, so a future binding cannot silently re-introduce the same failure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence from the live pass: with focus outside the composer, `Alt+T` worked and produced a new tab. With the composer focused, the tab strip was byte-identical before and after, and the draft gained a `t`. `tmux` is not the cause — `send-keys M-t` reaches the app, and the same sequence works when focus is elsewhere.

If the fix is to make the bindings win, the seam is Textual's printable-key handling in the focused `Input`; consider a key handler on the composer's `Input` subclass that forwards unhandled `alt+` chords to the screen, mirroring how `SettingsCategorySearchInput` already overrides `_on_key` to re-arm `/`.
<!-- SECTION:NOTES:END -->
