---
id: TASK-31000
title: Splash intro playback stability - frame-locked animation pacing
status: Done
assignee:
  - '@{self}'
created_date: '2026-09-02 06:22'
updated_date: '2026-09-02 15:44'
labels:
  - splash-screen
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Startup splash animations are unstable: sometimes smooth, sometimes jumpy, sometimes they skip from an early frame straight to the end, and sometimes they do not play at all. Root causes verified by measurement: (1) effects derive progress from wall-clock elapsed time while Textual set_interval permanently skips missed ticks under event-loop contention, so delayed ticks make reveals jump forward (probe: 1.2s loop block right after arming -> first visible frame already at 1.22s of a 2.5s reveal, 56/80 ticks rendered); (2) the first frame only renders at the first interval tick, so a blocked loop lets the auto-close timer win the race -> zero frames; (3) cards with 2.5-3s reveal timelines play under a default 1.5s splash duration and get cut mid-reveal; (4) identical consecutive frames still trigger a display re-render. Fix centrally in Widgets/splash_screen.py so both the startup splash and the Settings preview benefit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First animation frame renders synchronously at animation start (no blank window; auto-close can no longer beat zero rendered frames under loop contention)
- [x] #2 Animation progress is frame-locked: each rendered frame advances the effect clock by exactly one animation interval, so contention slows playback instead of jumping reveals forward
- [x] #3 Card reveal durations longer than the configured splash duration are clamped so a reveal completes before the splash closes
- [x] #4 Consecutive identical frames do not re-write the display widget
- [x] #5 Regression smoke: every animated card still renders 5 consecutive frames without falling back to static (five pre-existing broken cards pinned and excluded, follow-up TASK-31001)
- [x] #6 New playback tests plus existing splash tests pass
- [x] #7 (added mid-task) Default splash duration is 7.0s in the widget constructor default, the app-level get_cli_setting fallback, the Settings viewer defaults, and the config.toml template; skip_on_keypress stays true in the template
- [x] #8 (added mid-task) Any unbound keypress during the startup splash skips it (app-level forwarding; app bindings like ctrl+q unaffected), verified end to end in a full-app test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Reason: bug fix within the existing splash animation architecture (pacing change inside Widgets/splash_screen.py), no storage/sync/provider/security/UX-structure boundary changes.

1. Restructure SplashScreen._update_animation into a shared _render_animation_frame() used by both the interval tick and a synchronous first-frame render at animation start.
2. Frame-lock the effect clock: before each effect.update(), advance effect_handler.start_time to now - rendered_frames * animation_interval so wall-clock-elapsed effects advance exactly one interval per rendered frame.
3. Clamp card reveal duration to the splash duration when longer (effect kwargs override at construction).
4. Skip display.update() when frame content is identical to the last rendered frame.
5. Add Tests/Widgets/test_splash_animation_playback.py: first-frame-immediate, frame-locked pacing (typewriter reveal delta == 1 char per manual tick after a real-time sleep), duration clamp, identical-frame skip, all-animated-cards 5-frame smoke.
6. Run targeted tests: new file + Tests/Widgets/test_splash_screen_config_read.py + Tests/UI/test_settings_splash_screen_viewer.py.
7. (added) Default duration 7.0s across SplashScreen.__init__, _load_splash_config defaults, TldwCli compose fallback, SettingsSplashScreenViewer DEFAULT_SPLASH_CONFIG, and the CONFIG_TOML_CONTENT template.
8. (added) Make skip-on-keypress actually fire: SplashScreen.request_skip() public method; TldwCli.on_key forwards unbound keys to it while the splash is active (the splash is not focusable so its own on_key never fires). Full-app test pins the skip path with duration pinned to 60s so auto-close cannot mask a dead skip.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Splash playback stability (driver, Widgets/splash_screen.py):

- Root causes were measured, not guessed, with a headless probe (TldwCli via run_test + a controllable event-loop blocker): Textual's Timer (skip=True default) permanently skips interval callbacks that could not run while the loop was blocked, and effects derive progression from time.time() - start_time, so delayed ticks made reveals jump (probe: 1.2s block right after arming -> first visible frame already at 1.22s of a 2.5s reveal, 56/80 ticks). With the fix, the same 1.2s block resumes playback at virtual elapsed ~0.18s on a smooth 50ms cadence, and a 4.5s block (longer than the splash) still shows frame 0 immediately instead of zero frames.
- _render_animation_frame() is the single render path; it re-anchors effect_handler.start_time to now - (rendered_frames + 1) * interval before each update(), so every wall-clock-based effect advances exactly one interval per rendered frame. current_frame now counts rendered frames only (falsy frames do not advance the virtual clock). hacker_terminal's accumulator actually becomes more correct under this clock (it previously accumulated total elapsed per frame).
- Card reveal duration is clamped to the splash duration at effect construction (only when longer and duration > 0), so reveals complete before close instead of being cut mid-way.
- Identical consecutive frames skip display.update() (Static.update parses markup synchronously via visualize, so this also saves real CPU on steady-state effects).
- Construction-time first-render failure now leaves effect_handler None and falls back to static before the interval is armed (previously a broken first tick armed a dead timer).

Default duration and skip:

- Default is now 7.0s in SplashScreen.__init__, _load_splash_config, TldwCli's compose fallback, the Settings viewer DEFAULT_SPLASH_CONFIG, and the config.toml template. Existing config.toml files with an explicit [splash_screen] duration keep their own value (Settings > Splash > Duration changes it).
- Skip-on-keypress never fired at startup: the splash is not focusable and nothing else is mounted, so keys landed on the default screen and the widget's on_key was unreachable. TldwCli.on_key now forwards unbound keys to SplashScreen.request_skip() while the splash is active; App._check_bindings runs before it, so ctrl+q/ctrl+p etc. keep working during the splash.

Incident worth remembering: the first version of the TldwCli.on_key insertion landed between @on(SplashScreen.Closed) and its function, silently rebinding the decorator onto on_key. A decorated handler gets _textual_on metadata and is excluded from name-based key dispatch, so the skip appeared dead in every test while working when the method was monkeypatched at runtime. Found by printing the function's _textual_on attribute. (Lesson added to backlog/docs/lessons-testing-evidence.md.)

Out of scope, filed as TASK-31001: five animated cards are broken independently of the driver (cyberpunk_glitch/hypno_swirl/phonebooths emit markup Textual's Content.from_markup rejects, world_map raises AttributeError, typewriter_news returns no frames) -- roughly a 1-in-15 chance per launch of an intro that visibly does not play. They are pinned in KNOWN_BROKEN_EFFECT_CARDS in the new smoke test.

Modified/added files: tldw_chatbook/Widgets/splash_screen.py, tldw_chatbook/Widgets/settings_splash_screen_viewer.py, tldw_chatbook/app.py (on_key + duration fallback), tldw_chatbook/config.py (template), Tests/Widgets/test_splash_animation_playback.py (new), Tests/UI/test_splash_skip_on_keypress.py (new), Tests/Widgets/test_splash_screen_config_read.py (default pins). Verification: 22 splash tests green (playback x5 runs for flake check, skip x3), ruff clean, adjacent full-app startup test green.
<!-- SECTION:NOTES:END -->


## Renumbering provenance

Second collision: the renumbered ids 30016/30017 were themselves minted on dev (Server-capture backlog batch) while this PR was open. Final ids 31000/31001 sit far beyond dev's allocation frontier (concurrent `backlog task create` sessions mint at local max+1) so an open PR cannot keep racing the frontier.

Originally created 2026-09-02 06:22 as TASK-28026. Dev independently minted
another TASK-28026 (Library media viewer Analysis-tab search, created 06:46
the same day) and merged it first, with the id baked into thirty-plus
`task-28026:` code comments across the Library screens. The TASK-19601 rule
says the older arrival keeps the id, but the younger one here is already
merged and referenced throughout shipped code, so this task -- still an open
PR -- renumbers instead to TASK-31000 rather than rewriting dev's library
comments. Sibling follow-up task renumbered TASK-28028 -> TASK-31001 to keep
the pair adjacent. Verified by scripts/check_backlog_task_ids.py.
