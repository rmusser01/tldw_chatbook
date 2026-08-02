---
id: TASK-1800
title: 'Alt+<letter> bindings never fire while the composer has focus'
status: Done
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
- [x] #1 `Alt+M` opens the model popover while the composer has focus, or the binding is removed and every surface advertising it is updated
- [x] #2 `Alt+W` opens the workspace switcher while the composer has focus, or the binding is removed and every surface advertising it is updated
- [x] #3 No `alt+<letter>` chord inserts a literal character into the composer draft
- [x] #4 Whichever route is taken, the behaviour is verified live in a real terminal, not only by unit test — the defect is invisible to the test suite
- [x] #5 A regression test pins the outcome, so a future binding cannot silently re-introduce the same failure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence from the live pass: with focus outside the composer, `Alt+T` worked and produced a new tab. With the composer focused, the tab strip was byte-identical before and after, and the draft gained a `t`. `tmux` is not the cause — `send-keys M-t` reaches the app, and the same sequence works when focus is elsewhere.

If the fix is to make the bindings win, the seam is Textual's printable-key handling in the focused `Input`; consider a key handler on the composer's `Input` subclass that forwards unhandled `alt+` chords to the screen, mirroring how `SettingsCategorySearchInput` already overrides `_on_key` to re-arm `/`.
<!-- SECTION:NOTES:END -->

## Implementation Notes (completion)

<!-- SECTION:COMPLETION:BEGIN -->
Root cause was in this repo, not Textual. `ChatScreen.on_key` captured every
`is_printable` key as composer text; the parser gives `alt+m` a `character` of
`"m"`, and `on_key` runs before bindings and stops the event. Fixed with a
`_is_modified_chord` guard on the capture branch.

Both chords now fire (live-verified: Alt+M opened the model popover, Alt+W the
workspace modal) and plain typing still works.

**AC#3 note:** satisfied for `alt+<letter>`. `alt+<digit>` turned out to be a
different defect entirely and is tracked separately as TASK-1802 — those
bindings never carry the name `alt+N`, so no capture-level fix reaches them.

**AC#4/#5 note:** `pilot.press("alt+m")` cannot reproduce this — Textual's pilot
sets `character=None` for multi-character key names, so the chord is not
printable in tests. The regression test constructs the event the way
`_xterm_parser` builds it, and documents why.
<!-- SECTION:COMPLETION:END -->
