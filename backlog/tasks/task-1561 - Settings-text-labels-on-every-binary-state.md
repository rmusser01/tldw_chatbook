---
id: TASK-1561
title: 'Settings: text labels on every binary state (splash toggles, Image Gen backends)'
status: Done
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, accessibility, P1]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique finding (P1): Splash Screen's "Enabled", "Show", "Skip on" rows
render as blank boxes with no readable On/Off state; Image Gen backend rows
lead with a bare "X" glyph whose meaning is carried by color alone (green =
configured, red = not). This violates the product's own design law that
states must be text-labeled and color never the only carrier -- on the one
screen whose job is stating configuration. "Skip on" also reads as a
truncated label.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every toggle in Settings shows its state as text (On/Off or Enabled/Disabled), not an empty control.
- [x] #2 Image Gen backend status is text-labeled ("[x] Enabled" style); color becomes reinforcement only.
- [x] #3 "Skip on" label completed (e.g. "Skip on keypress").
- [x] #4 Live screenshot verification of Splash Screen and Image Gen categories.
<!-- AC:END -->

## Implementation Plan

1. On/Off Statics beside the three splash Switches, synced in their Changed handlers.
2. State-bearing checkbox labels on Image Gen ("On"/"Off" per backend row, "Context LLM: on/off"), synced on change.
3. Complete/unclip the "Skip on keypress" label.

## Implementation Notes

- Splash: `switch_state_label` + a `-state` Static per Switch, updated in each handler (widget test asserts the flip).
- Image Gen: the "Enabled" label WAS present but the row truncated under narrow widths when lengthened -- the geometry regression test caught my first "Enabled (on/off)" attempt pushing the Test button off a 120-col terminal; final form is the short state word ("On"/"Off") via `switch_word`, tooltip carrying the property meaning, synced in `handle_image_gen_checkbox_changed`. `toggle_label` covers the full-width "Context LLM" checkbox.
- "Skip on keypress" was complete in code; the shared label column truncated it -- the three splash labels now set an explicit 17-cell width.
- Process note: implementation preceded the new tests for this task (order deviation from strict RED-first); the pre-existing geometry suite provided the failing signal that shaped the final design, and new tests were added after.
- Live screenshot: "Enabled  Off / Show progress  On / Skip on keypress  On".
