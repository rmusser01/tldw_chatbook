---
id: TASK-1802
title: 'Alt+digit tab-jump bindings can never fire'
status: To Do
assignee: []
created_date: '2026-08-01 17:05'
labels:
  - console
  - keyboard
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ChatScreen.BINDINGS` declares nine tab-jump chords, `Binding("alt+1", "jump_console_tab(1)")` through `alt+9` (`chat_screen.py:1585-1593`). None of them can ever match, in any focus state, on any screen.

Textual's terminal parser resolves an ESC-prefixed digit through `ANSI_SEQUENCES_KEYS` (`_xterm_parser.py:403`) **before** it reaches the alt-prefix branch that would produce `alt+1` (`:425-438`). That table maps `ESC 1` to `¡`, so the event arrives as `key="inverted_exclamation_mark"`, `character="¡"` — never `alt+1`.

Verified by feeding the exact bytes `0x1b 0x31` to a fresh `XTermParser`:

```
Alt+1 -> key=inverted_exclamation_mark character='¡' is_printable=True
Alt+M -> key=alt+m                     character='m' is_printable=True
```

This is a **different defect from TASK-1800**, which is why it was split out. TASK-1800's chords do reach the app correctly named and were merely being swallowed by the composer's printable capture; these nine never carry the right name at all, so no focus-level or capture-level fix can help them.

The correct fix is platform-sensitive and needs a decision rather than a patch: macOS Option+1 emits `¡`, while other terminals and layouts emit `ESC 1` or use the Kitty keyboard protocol. Re-pointing the bindings at `¡™£¢∞§¶•ª` would fix macOS and break everything else, so this likely wants a normalisation layer or a different chord family entirely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Jumping to Console tab N by keyboard works, or the nine dead bindings are removed and every surface advertising them is updated
- [ ] #2 Whatever is chosen works on macOS and does not regress Linux/Windows terminals, or the platform limits are stated explicitly in the code
- [ ] #3 Verified live in a real terminal — `pilot.press("alt+1")` cannot reproduce this class of defect (see the note below) and must not be the only evidence
- [ ] #4 A test pins the outcome so a binding that cannot match is not silently re-added
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Why the test suite cannot see this.** Textual's test pilot builds key events as `char = key if len(key) == 1 else None` (`app.py:2063`), and it does not run the terminal parser at all. `pilot.press("alt+1")` therefore synthesises a well-formed `Key("alt+1", None)` that matches the binding perfectly — the opposite of what a real terminal delivers. Any test written with the pilot will pass against completely dead bindings.

The same pilot gap is what hid TASK-1800 from ~10,000 passing tests. When testing key handling in this codebase, construct the event the way `_xterm_parser` builds it, or drive a real terminal.

Related: [[TASK-1800]] (alt+letter swallowed by the composer's printable capture — fixed), and the Console also binds `ctrl+t`, which works because its character is the C0 byte `\x14` and so is not printable.
<!-- SECTION:NOTES:END -->
