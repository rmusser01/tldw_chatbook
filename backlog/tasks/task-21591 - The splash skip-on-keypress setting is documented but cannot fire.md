---
id: TASK-21591
title: >-
  The splash skip on keypress setting is documented but cannot fire
status: Done
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

- [x] Pressing a key during the splash dismisses it when `skip_on_keypress = true`, or the setting and its documentation are removed
- [x] Whichever way it goes, a test pins the behaviour so it cannot silently rot again
- [x] If implemented, the skip is verified in a real terminal and not only under Pilot — the original observation was Pilot-only
- [x] `skip_on_keypress = false` is verified to leave the splash running its full duration

## Evidence (verified first-hand on dev, 2026-08-23)

- `Widgets/splash_screen.py:26` — `class SplashScreen(Container)`, with no `can_focus`.
- `Widgets/splash_screen.py:454` — `async def on_key(...)`, gated on `self.skip_on_keypress`.
- Nothing in the tree focuses the splash.
- `config.py:3224` — `skip_on_keypress = true  # Allow users to skip with any keypress`;
  `app.py:8759` reads it and `:8774` passes it through, so the wiring is complete right up to a
  handler that never fires.

Observed by the TASK-21110 implementer: `pilot.press("space")` at +0.05 s and +0.5 s after splash
mount did not close the splash — it ran its full 1.5 s both times.

## Implementation Plan

1. Reproduce the diagnosis on the base rather than trusting the filing: read Textual's key
   routing, then drive a minimal host with Pilot and print `can_focus` / `App.focused`.
2. Decide implement-vs-remove and say why.
3. Make the splash focusable and focused *only when the skip is enabled*, so the Settings
   splash preview (which passes `skip_on_keypress=False`) cannot steal focus.
4. Pin it with tests, and mutation-test every one of them.
5. Verify both arms in a real terminal via tmux, A/B against the base, and check the
   TASK-21110 pre-import interaction with an immediate skip.

## Implementation Notes

**Implemented, not removed.** The knob is not only a config line: Settings ▸ Splash Screen
ships a **Skip on keypress** checkbox that writes it, so removing it would retire a control
users can already see and may already have toggled — for a fix that is three lines.

**Verified on the base first.** Textual's `App.on_event` does
`forward_target = self.focused or self.screen` and the event bubbles *upward* from there, so
a never-focused `Container` is unreachable. A Pilot probe on `7f38cb6ef` confirmed it:
`can_focus=False`, `App.focused=None`, and after `pilot.press("space")` the splash reported
`closed=[] skip_requested=False`. The filing's own observation was Pilot-only; this was then
reproduced **in a real terminal** (below), which the filing had not done.

**The fix.** `SplashScreen.__init__` sets `self.can_focus = bool(skip_on_keypress)` and
`on_mount` calls `self.focus()` under the same condition. Tying focus to the setting is what
keeps the Settings splash PREVIEW (`settings_splash_screen_viewer.py`, which constructs the
widget with `skip_on_keypress=False`) from taking focus away from the settings controls
around it.

**`on_key` consumes the key -- after trying the opposite and finding it broke
something already decided.** The first version deliberately did NOT stop the event, on the
reasoning that today a key during the splash bubbles to the app's bindings, so swallowing it
would break `ctrl+q`. That shipped green through every splash-adjacent suite and both live
arms -- and then a wider sweep turned
`Tests/UI/test_screen_navigation.py::test_navigation_keypress_during_splash_is_safely_ignored`
(task-1339's regression lock) red: **F9 mid-splash landed the user on Settings**.

The mechanism is worth recording because it is not visible at the call site.
`action_shell_destination` does not navigate; it *posts* a `NavigateToScreen`, and the guard
that swallows it lives in the handler (`_handle_screen_navigation_locked`, "initial screen
not yet mounted"). Message order on the App queue becomes `Key` -> `Closed` -> (Key's
binding posts) `NavigateToScreen`, so once the splash closes *immediately* the `Closed`
handler has already pushed the initial screen by the time the navigation is handled and the
guard no longer fires. Before this fix the splash never closed on the key, `Closed` only
arrived at the duration timer, and the request was correctly swallowed. Making the skip work
is precisely what re-opened that window.

So the key is stopped. The splash is a modal over an app that is not interactive yet, and
while it is up the key's only job is to dismiss it. Every already-decided contract stays
intact; the cost is that `ctrl+q` during the splash dismisses the splash and needs a second
press to quit -- which no task or test has ever asserted otherwise (checked: no test in the
suite locks app bindings firing during the splash). Verified live: `F9` 29 ms into a 25 s
splash dismissed it and left the app on Home, not Settings.

Modified: `tldw_chatbook/Widgets/splash_screen.py`, `Docs/User_Guide/settings.md` (Splash
Screen row + a new Verified-against stamp). Added: `Tests/UI/test_splash_skip_on_keypress.py`.

### Real-terminal verification (tmux, isolated `TLDW_CONFIG_PATH` + scratch HOME/XDG)

| arm | base `7f38cb6ef` | this branch |
|---|---|---|
| Space during a 25 s splash | still up at +6 s, closed only at its 25 s duration | gone within 3 s, app booted to first-run setup |
| Space 23 ms after the first painted splash frame | no effect | dismissed; full UI (nav bar, destinations) within 3 s |
| `F9` 29 ms after the first painted frame | no effect (swallowed) | splash dismissed, app settles on Home -- **not** Settings |
| `ctrl+q` during the splash | app quits | splash dismissed; app stays up, quit takes a second press |
| `skip_on_keypress = false` + Space | n/a | still up 5 s after the press; auto-closed after its full 20 s |

**TASK-21110 checked, not regressed.** Its splash-overlapped pre-import thread starts 0.2 s
into the splash, and an early skip can make the main thread block on the module lock. The
immediate-skip arm above presses *before* 0.2 s — earlier than any user could — and boot
completed to a working UI in under 3 s. That path was unreachable before this fix; it is now
reachable and it works.

### Mutation results (every test proven to discriminate)

| mutant | dismisses | key-consumed | no-double-close | full-duration | skip-off-routes-normally |
|---|---|---|---|---|---|
| M1 `can_focus` forced `False` | FAIL | FAIL | FAIL | pass | pass |
| M2 `can_focus` forced `True` unconditionally | pass | FAIL | pass | **FAIL** | pass |
| M3 `on_mount`'s `self.focus()` removed | FAIL | FAIL | FAIL | pass | pass |
| M4 `event.stop()`/`prevent_default()` removed | pass | **FAIL** | pass | pass | pass |
| M5 both `_skip_requested` guards dropped | pass | FAIL | **FAIL** | pass | pass |
| M6 focus + consume made unconditional | pass | FAIL | pass | FAIL | **FAIL** |

Every one of the five tests is killed by at least one mutant. M6 is compound on purpose:
"with the skip disabled, key routing is unchanged" is only reachable by making both the
focus and the consume unconditional.

One of these is a finding in its own right: removing `self.focus()` left **all four tests
green** at first. Textual's `App.AUTO_FOCUS = "*"` auto-focuses the first focusable widget on
screen mount, so `can_focus = True` alone was enough in the harness — a second writer
satisfying the assertion. The host app now sets `AUTO_FOCUS = None`, so the widget's own
`focus()` is the only path and the mutant reds. The feature must not silently depend on a
screen-level default it does not own.

### Quit / error walk

`_request_close()` and `on_key` both guard on `_skip_requested`, so a burst of keys yields
exactly one `Closed` (pinned by a test). `Closed` drives the real app's one-shot
`on_splash_screen_closed` — remove splash, mount chrome, push the initial screen — which is
why double-firing matters. A key that arrives after the splash is removed reaches the newly
mounted screen instead; the widget is out of the DOM and Textual's `_reset_focus` moves focus
on for us (verified live: after the skip the app is fully navigable). With the skip disabled
the widget is not focusable at all, so nothing changes for the Settings preview's unmount path.

### The sweep that caught it

The four splash-adjacent suites and both live arms were green on the rejected version. Only
`Tests/UI/test_screen_navigation.py` (506 tests, 5m19s) exposed it -- a file that never
mentions the widget under change. **A behavioural change to input routing has to be run
against the navigation suite, not only the feature's own.**

### Test counts

`Tests/Widgets/test_splash_screen_config_read.py`, `Tests/UI/test_settings_splash_screen_viewer.py`,
`test_splash_initial_screen_preimport.py`, `test_screen_preimport.py`,
`test_screen_preimport_pacing.py`, `test_splash_skip_on_keypress.py`,
`Tests/Utils/test_startup_polish_regressions.py`: **169 passed, 0 failed**.
`Tests/UI/test_screen_navigation.py` + `Tests/Wizards/test_first_run_setup_wizard.py`:
**507 passed, 0 failed** (506+1 after the correction; the rejected version failed
`test_navigation_keypress_during_splash_is_safely_ignored` here).
Focus/startup sweep (`test_product_maturity_phase1_keyboard_focus.py`,
`test_workbench_pane_focus.py`, `test_product_maturity_phase1_first_run.py`,
`test_product_maturity_phase6_focus_visual_sweep.py`,
`Tests/Performance/test_app_startup_performance.py`): **48 passed, 4 failed** — all four reds
are in `test_product_maturity_phase6_focus_visual_sweep.py` and fail identically on pristine
dev `7f38cb6ef` (**1 passed, 4 failed**, same nodeids), A/B'd in a worktree at that SHA.
