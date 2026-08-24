---
id: TASK-21591
title: >-
  The splash skip on keypress setting is documented but cannot fire
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - ui
  - dead-config
priority: low
---
## Description

`[splash_screen] skip_on_keypress` is shipped, defaulted to `true`, and documented in the config
as "Allow users to skip with any keypress". It cannot work. `SplashScreen` is a `Container` that
is never focusable and is never focused, and Textual routes key events to the focused widget and
bubbles them *upward* — so the widget's `on_key` never runs and the setting is inert.

Either the skip works or the setting should not be advertised. A config knob that silently does
nothing is worse than no knob.

## Acceptance Criteria

- [ ] Pressing a key during the splash dismisses it when `skip_on_keypress = true`, or the setting and its documentation are removed
- [ ] Whichever way it goes, a test pins the behaviour so it cannot silently rot again
- [ ] If implemented, the skip is verified in a real terminal and not only under Pilot — the original observation was Pilot-only
- [ ] `skip_on_keypress = false` is verified to leave the splash running its full duration

## Evidence (verified first-hand on dev, 2026-08-23)

- `Widgets/splash_screen.py:26` — `class SplashScreen(Container)`, with no `can_focus`.
- `Widgets/splash_screen.py:454` — `async def on_key(...)`, gated on `self.skip_on_keypress`.
- Nothing in the tree focuses the splash.
- `config.py:3224` — `skip_on_keypress = true  # Allow users to skip with any keypress`;
  `app.py:8759` reads it and `:8774` passes it through, so the wiring is complete right up to a
  handler that never fires.

Observed by the TASK-21110 implementer: `pilot.press("space")` at +0.05 s and +0.5 s after splash
mount did not close the splash — it ran its full 1.5 s both times.
