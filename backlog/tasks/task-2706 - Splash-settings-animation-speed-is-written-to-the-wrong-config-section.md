---
id: TASK-2706
title: 'Splash settings: "Animation speed" is written to the wrong config section'
status: To Do
assignee: []
created_date: '2026-08-01'
labels: [settings, splash, bug]
dependencies: []
---

## Description (the why)

Settings ▸ Splash Screen reads `animation_speed` (and `fade_in_duration` /
`fade_out_duration`) from **`[splash_screen.effects]`** but writes every
value to **`[splash_screen]`**, so the "Animation speed (x)" field saves
into a section nothing reads. The pane will not show the saved value again
on reopen, and the splash runtime keeps using the `effects` value.

Source (dev @ fb2df0c8a),
`tldw_chatbook/Widgets/settings_splash_screen_viewer.py`:

- loader: `_EFFECTS_KEYS = {"fade_in_duration", "fade_out_duration",
  "animation_speed"}` and it reads those from
  `"splash_screen.effects" if key in _EFFECTS_KEYS else "splash_screen"`
  (~:60-66).
- writer: `_save_config_value` calls
  `save_setting_to_cli_config("splash_screen", key, value)`
  unconditionally (~:225-232).

So the round-trip is broken for exactly the three effects keys. Only
`animation_speed` has a control today; `fade_in_duration` and
`fade_out_duration` have none, so they are unaffected in practice but
share the bug if controls are added.

Found while writing the G4 Settings user-guide page; the page currently
has to describe the field without promising it sticks.

## Acceptance Criteria (the what)

- [ ] Saving "Animation speed (x)" writes to the same section the loader
      reads (`[splash_screen.effects]`), so reopening the pane shows the
      saved value and the splash honours it.
- [ ] A test round-trips the value through save → reload → render.
- [ ] Confirm which section the splash runtime actually consumes and make
      loader, writer, and runtime agree; if any value already exists in
      the wrong section from this bug, decide and document whether it is
      migrated or ignored.
- [ ] Update the User Guide note in `Docs/User_Guide/settings.md` once the
      behaviour is fixed.
