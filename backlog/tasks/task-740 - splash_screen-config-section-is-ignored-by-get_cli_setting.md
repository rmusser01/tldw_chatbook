---
id: TASK-740
title: '[splash_screen] config section is ignored by get_cli_setting'
status: To Do
assignee: []
created_date: '2026-07-26 08:00'
labels:
  - config
  - bug
  - splash-screen
dependencies:
  - TASK-545
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Widgets/splash_screen.py:196` and `Widgets/settings_splash_screen_viewer.py:54` both call `get_cli_setting("splash_screen", <dict>)` — a bare section name with a non-string second positional argument. This hits the same bug class as TASK-547/TASK-658: `get_cli_setting(section, key=None, default=None)` treats a non-string second argument as `default`, not `key`, so `config.py`'s own fallback branch returns that argument unconditionally without ever reading the `[splash_screen]` TOML section. Confirmed at runtime against a real user config that has a populated `[splash_screen]` dict — it is discarded on every call.

`CLAUDE.md` documents `[splash_screen]` as one of the app's key configuration sections. Every user who customized splash screen settings (animation choice, timing, custom cards, etc.) has had those settings silently ignored.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] `Widgets/splash_screen.py`'s read of `[splash_screen]` uses a call shape that actually returns the configured section, not an unconditional default
- [ ] `Widgets/settings_splash_screen_viewer.py`'s equivalent read is fixed the same way
- [ ] Setting values under `[splash_screen]` in `config.toml` (e.g. a non-default animation or duration) is observably honored
- [ ] Unit test confirms a configured `[splash_screen]` value is used, and that the hardcoded default only applies when the section/key is genuinely absent
<!-- AC:END -->
