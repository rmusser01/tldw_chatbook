---
id: TASK-1560
title: 'Settings: Save/Revert commit affordance visible and honest'
status: Done
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, P1]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sr UX/HCI critique of the Settings screen (snapshot
.impeccable/critique/2026-07-31T01-48-54Z, 29/40): the commit affordance is
invisible exactly when it matters. In clean state the Scope Inspector renders
"Save / Revert" as one dim stacked block that reads as dead chrome; in a LIVE
dirty state on Console Behavior (visually verified) no Save/Revert buttons
exist anywhere on screen -- commit is keyboard-only, while the footer's
"s save category" hint is silently inert whenever an Input/TextArea has focus
(by design, unhinted). The power-user path "type a value, press s" fails
silently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every writable category shows Save and Revert as two separate buttons that visibly enable when the draft is dirty and show a disabled reason ("No changes to save") when clean.
- [x] #2 Each button is paired with its key ("Save (s)" / "Revert (r)").
- [x] #3 While an Input/TextArea has focus, saving still works (binding or focus-aware footer hint such as "Esc then s to save") -- no silent no-op.
- [x] #4 Live-verified in the dirty state with a screenshot.
<!-- AC:END -->

## Implementation Plan

1. Split the Scope Inspector into a fixed header (identity, draft status, guided-action state, Save/Revert) above a scrollable body.
2. Label buttons with their keys: "Save (s)" / "Revert (r)".
3. Make the advertised keys honest around text-entry focus.

## Implementation Notes

- The buttons already existed (#settings-save-category/#settings-revert-category) with correct enable-on-dirty plumbing; the critique's "invisible in dirty state" was them scrolling out of the rail. They are now pinned in `_render_impact_pane_header`, which cannot scroll.
- Labels carry their keys. Clean state keeps the honest disabled look + "Guided edits: change a field first."
- Footer hints are focus-aware: while an Input/TextArea owns focus the hints read "Esc, s save category ..." (`_register_footer_shortcuts` + descendant focus/blur hooks). Live testing then exposed that ESC ONLY BLURRED THE FILTER -- every other Input swallowed the follow-up keys; a screen-level `on_key` escape branch now releases text-entry focus everywhere, making the advertised chain true.
- Live-verified end to end with screenshots: dirty state shows enabled pinned "Save (s)/Revert (r)"; Esc then r reverts with toast; footer un-prefixes after blur.
