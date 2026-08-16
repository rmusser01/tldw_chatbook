---
id: TASK-16811
title: DEFAULT_CSS local variable fallbacks silently override bundle tokens
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - console
  - css
  - ui-polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual resolves `$variables` per CSS *source*, not globally-with-fallback: a
`$var: value;` declared inside a widget's `DEFAULT_CSS` unconditionally
governs every use of that name within that same source, even when the real
app bundle (loaded earlier in `CSS_PATH`) defines the token. The "local
fallback so DEFAULT_CSS parses standalone" pattern is therefore an
unconditional override, not a fallback.

Verified live during the turn-file-card final review (2026-08-16): a
selected `ConsoleTurnFileCard` renders background `$surface` (`#1e1e1e`)
instead of the bundle's `$ds-focus-bg` (`#51677e`), so a selected card is
visibly duller than every other selected transcript row despite the code's
own "parity with `.console-transcript-message-selected`" comment. The
pattern is inherited precedent, not unique to the card — `NavigationButton`
(`base_components.py`) and the `EmojiPickerScreen` rules carry the same
footgun.

Likely fix shape: move token-dependent rules for these widgets into the
scoped screen CSS sources (where bundle tokens resolve), keeping only
token-free structural rules in `DEFAULT_CSS` — then regenerate the bundle
(never hand-edit it). Assert the resolved background color in a
real-CSS-stack test, since a class-toggle assertion alone cannot catch
this (the class toggles correctly today; the color is what's wrong).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A selected ConsoleTurnFileCard renders the same focus background token as a selected plain transcript message under the real app CSS bundle, asserted on resolved color in a real-CSS-stack test
- [ ] #2 Widgets audited for the same local-`$var`-in-DEFAULT_CSS override pattern (at minimum NavigationButton and EmojiPickerScreen) are either fixed the same way or explicitly recorded as intentional with a comment
- [ ] #3 The CSS bundle is regenerated, never hand-edited, and existing geometry/CSS tests still pass
<!-- AC:END -->
