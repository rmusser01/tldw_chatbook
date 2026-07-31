---
id: task-1582
title: 'Settings: interactive elements indistinguishable from prose'
status: To Do
assignee: []
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - rescore-p2
dependencies: []
priority: medium
---

## Description (the why)

Critique rescore P2: in Console Behavior (and elsewhere), toggle words
("Enabled"), editable inputs ("Threshold 50"), and explanation prose render
at near-identical weight and background, so finding "the thing I can
change" requires tabbing blindly or reading everything. Disabled controls
(Save/Revert, Clear saved key, Save Raw TOML) differ from enabled ones only
by luminance — low-vision users lose the affordance entirely. Theme's
"Dark theme" switch is an empty rectangle with no text state at all (the
only control-without-text-state found in the evidence pass).

## Acceptance Criteria (the what)

- [ ] One consistent visual convention distinguishes interactive controls
      from prose across Settings categories (e.g. bracketed toggles,
      bordered inputs)
- [ ] Disabled Save/Revert (and other disabled buttons) carry a text
      annotation or contrast treatment beyond dimming alone
- [ ] The Theme "Dark theme" switch shows a text state (On/Off) like the
      Splash Screen switches do
- [ ] A visible focus indicator exists on center-pane fields
