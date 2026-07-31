---
id: TASK-1582
title: 'Settings: interactive elements indistinguishable from prose'
status: Done
assignee:
  - '@claude'
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

- [x] Disabled Save/Revert carry a text annotation beyond dimming alone
      ("— no changes" in the clean state; plain labels once dirty, since
      disabled-with-invalid-changes means "fix validation", which the
      guided-action state row already explains)
- [x] The Theme "Dark theme" switch shows a text state (On/Off) like the
      Splash Screen switches do
- [x] The scope of the remaining work — one screen-wide interactive
      convention (bracketed toggles, bordered inputs, focus ring) — is
      re-filed as its own design task (task-1586) with the discovered
      constraint documented

## Implementation Notes

Scope adjustment (AC updated before implementing, per workflow): the
originally-filed "bordered inputs" convention is not a hygiene-level
change — `.settings-compact-input` deliberately uses `border: none` at
`height: 1`, and a Textual border consumes rows, so bordering every input
triples its height and reflows every dense form in Settings. That is a
design project with layout blast radius, split into task-1586. This task
delivered the text-carried state affordances:

- `_guided_action_label` annotates the Save/Revert pair with "— no
  changes" in the clean state (compose + `_update_guided_action_widgets`,
  which now also refreshes labels). Dirty-but-invalid keeps plain labels;
  the state row explains the validation block.
- Theme's dark-mode Switch gains an On/Off word
  (`#settings-theme-dark-mode-state`), reusing the Splash viewer's
  `switch_state_label`, synced on programmatic loads and user toggles.

TDD RED-first; suites green. Files:
`tldw_chatbook/UI/Screens/settings_screen.py`,
`tldw_chatbook/Widgets/settings_theme_editor.py`,
`Tests/UI/test_settings_configuration_hub.py`,
`Tests/UI/test_settings_theme_editor.py`.
