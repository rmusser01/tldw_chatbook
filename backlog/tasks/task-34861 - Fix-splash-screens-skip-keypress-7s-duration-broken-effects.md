---
id: TASK-34861
title: 'Fix splash screens: skip keypress, 7s duration, broken effects'
status: Done
assignee: []
created_date: '2026-09-12 00:00'
updated_date: '2026-09-12 00:16'
labels:
  - splash
  - bugfix
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Splash screen fixes: (1) keypress skip never fires because SplashScreen container is not focusable; (2) duration defaults to 1.5s instead of 7s; (3) world_map effect crashes (missing width/height attrs), ~11 effects render frozen frames (frame_count never advances), 3 dead card names in user config, animation_speed config loaded but unused.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keypress during splash skips it
- [x] #2 Splash displays for 7s by default and remains skippable
- [x] #3 world_map animates without crashing
- [x] #4 All active-card effects produce changing frames (no frozen rendering unless intentionally static)
- [x] #5 Targeted splash tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Make SplashScreen focusable so on_key skip works; set duration defaults to 7.0s; apply animation_speed config multiplier
2. Fix world_map (store width/height, call super().update())
3. Audit 11 frozen-looking effects, fix confirmed-frozen ones
4. Add targeted tests (registry/card cross-check + frame variance + skip + duration)
5. Update user config.toml (duration 7.0, remove dead card names)
6. Live-run app to eyeball previously-broken cards
ADR required: no - routine bug fixes, no architectural change
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed all three splash complaints with targeted, verified changes:

- **Skip**: `SplashScreen` was a non-focusable `Container`, so its `on_key` never fired. Added `can_focus = True` + `self.focus()` in `on_mount`. Verified end-to-end via Textual pilot (keypress closes splash) and headless real-app boot (closed 0.8s after space).
- **Duration**: defaults 1.5s/2.5s -> 7.0s in `app.py` and `splash_screen.py`; user `config.toml` `duration = 1.5 -> 7.0` (backup at `config.toml.bak-20260911`). Auto-close verified at 6.9s in a headless boot.
- **Broken effects** (audit of all 78 cards / 87 effects via smoke + frame-variance script):
  - `world_map`: crashed on frame 1 (used `self.width/height` never stored; `BaseEffect` swallows kwargs). Now stores dims, calls `super().update()`, bounds-guards drawing.
  - `typewriter_news`: `update()` returned `None` (rendering lived in an uncalled `render()`), and typing accumulated total-elapsed-time per frame (quadratic). Now delta-based and returns `self.render()`; also stores width/height/speed it used but never set.
  - `starfield`: radius scale kept all stars within ~8 cells of center (looked frozen). Scale `/10 -> /2` so stars traverse the screen.
  - `level_up`: static until the 100-frame level-up. Added a per-frame XP bar; progress step 1 -> 2 (level-up ~2.5s at default cadence).
  - Progress bar scale bug: callers pass 0.0-1.0 fractions but `ProgressBar(total=100)` received them raw (<=1%). `update_progress` now scales fraction -> percent.
  - `animation_speed` config was loaded but unused; now applied as a multiplier on the card's frame interval.
  - Audited and cleared as intentionally-slow/static, not broken: `arcade_high_score` (blink 0.5s), `loading_bar` (progress-driven), `scrolling_credits`, `raindrops_pond`, `bookshelf_browser`, `weather_system`, `emoji_face`, `custom_image`.
  - User `config.toml`: removed 3 dead card names (`classic`, `compact`, `minimal`) from `active_cards`.
- **Tests**: new `Tests/Widgets/test_splash_screen_fixes.py` (13 tests: focusability, 7s default, registry/card cross-check, all-effects smoke, per-effect regressions, pilot keypress-skip) + updated stale 2.5s default assertion in `test_splash_screen_config_read.py`. All pass; `Tests/UI/test_settings_splash_screen_viewer.py` still passes (9).
- **Docs**: two traps recorded in `backlog/docs/lessons-textual.md` (non-focusable Container key handlers; effect `update()` frame-return contract + BaseEffect kwargs swallowing).
- **Deviations/notes**: live verification uncovered a pre-existing, unrelated boot failure in the dirty working tree (`UI/Console_Modules/workspace.py` imports missing `console_appearance_picker_modal` WIP) — splash closes fine; the post-close screen push crashes. Not touched by this task.

Modified: `tldw_chatbook/Widgets/splash_screen.py`, `tldw_chatbook/app.py`, `Utils/Splash_Screens/{gaming/world_map,gaming/level_up,tech/typewriter_news,environmental/starfield}.py`, `Tests/Widgets/test_splash_screen_fixes.py` (new), `Tests/Widgets/test_splash_screen_config_read.py`, `backlog/docs/lessons-textual.md`, user `~/.config/tldw_cli/config.toml`.
<!-- SECTION:NOTES:END -->
