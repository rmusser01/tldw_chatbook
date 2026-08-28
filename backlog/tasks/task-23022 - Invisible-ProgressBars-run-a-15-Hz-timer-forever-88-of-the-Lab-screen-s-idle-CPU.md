---
id: TASK-23022
title: >-
  Invisible ProgressBars run a 15 Hz timer forever - 88% of the Lab screen's idle CPU
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
updated_date: '2026-08-27'
labels:
  - performance
  - idle
  - textual
priority: high
---

## Description

`ProgressBar(total=None)` makes Textual's `Bar` *indeterminate*, which arms `auto_refresh = 1/15` -
a **15 Hz `set_interval` that never stops**. Setting `display = False` does not stop it: Textual
gates only the *repaint* on `is_on_screen`, never the timer.

On the Lab screen this is **960 of 1018 timer fires in 15 s, changing zero pixels** - 88% of that
screen's idle CPU, at ~84 us per fire all-in.

This class is **structurally invisible** to the repo's timer census, which parses only
`tldw_chatbook/**.py`; no package file assigns `auto_refresh`, because the `set_interval` lives
inside `textual/dom.py`.

## Acceptance Criteria

- [x] A hidden or off-screen indeterminate `ProgressBar`/`LoadingIndicator` does not run a timer
- [x] The six live instances are fixed: `model_curated_view.py:471`, `model_installed_view.py:346` and `:352`, `model_remote_view.py:605`, `library_screen.py:12626`, `UI/CCP_Modules/ccp_loading_indicators.py:71`
- [x] The remaining 13 `ProgressBar(` and 6 `LoadingIndicator()` sites are audited and each is fixed or recorded as safe
- [x] A guard prevents a new hidden indeterminate progress widget from arming a permanent clock - the timer census cannot see this class today, so extending it is part of the work
- [x] Before/after idle CPU measured with interleaved arms

## Implementation Plan

1. Reproduce the mechanism against the pinned textual 8.2.8: `Bar.watch_percentage` arms
   `auto_refresh = 1/15` when percentage is None; `ProgressBar.on_mount` arms an unconditional
   `set_interval(1, self.update)`; `LoadingIndicator._on_mount` arms `auto_refresh = 1/16`;
   `display = False` never touches any of them.
2. Build one house module (`Widgets/pausable_progress.py`) with drop-in subclasses that pause
   every clock while the widget is outside the layout map (Show/Hide events + a `set_interval`
   interception that catches base-class arms a subclass cannot suppress).
3. Convert the live instances and every audited site to the house classes.
4. Guard: forbid constructing/subclassing the stock widgets anywhere else in the package
   (AST scan with alias resolution + positive-control fixtures). Census extension itself is
   TASK-23028's scope, per the wave plan.
5. Prove both halves with Pilot tests (hidden => zero fires; shown => fires AND the highlight
   moves), mutation-test every test, walk unmount/quit.
6. Interleaved live A/B on F7 and Personas: timer fires / 15 s (primary) + getrusage idle CPU.

## Implementation Notes

**Approach.** One new module, `tldw_chatbook/Widgets/pausable_progress.py`:
`HiddenClocksPausedMixin` intercepts `set_interval` (so even the clock armed by
`ProgressBar.on_mount` - which a subclass override cannot suppress, because Textual dispatches
naming-convention handlers for every class in the MRO - is captured), starts timers paused until
the widget's first `Show`, and pauses/resumes on `Hide`/`Show`. A paused `Timer`'s task blocks on
its `Event.wait()` - zero wakeups - and `Timer._skip` fast-forwards on resume, so no catch-up
burst. `PausableProgressBar` mirrors `ProgressBar.compose` with a `PausableBar` swap (no upstream
seam exists; a structural-parity test pins the copy against textual bumps).
`PausableLoadingIndicator` is the mixin applied to `LoadingIndicator`. Tracked timers are held
**strongly**: with weak tracking, the paused ETA timer (whose reference `ProgressBar.on_mount`
discards) exists only in the task<->timer reference cycle, and cycle GC destroys it mid-pause
("Task was destroyed but it is pending!") - observed during development.

**Sites.** The six live instances all funnel through two files: the five `ModelInstallProgress`
embeds (model_curated/installed x2/remote views, library_screen; paths moved to `UI/Screens/` on
this base, plus `FirstRunSetupWizard`) share one compose in
`Widgets/ModelArtifacts/install_progress.py`, now `PausableProgressBar`; the Personas CCP overlay
(`ccp_loading_indicators.py`) now composes `PausableLoadingIndicator`. The audit found a
**seventh live instance** the finding did not list: `console_conversation_inspector.py`'s
16 Hz `LoadingIndicator` mounted `display: none` on the persistent Console rail - fixed the same
way. All 13 remaining `ProgressBar(` and 9 remaining `LoadingIndicator(` sites (base drift from
the finding's 6) were converted to the house classes - several were latent instances of the same
bug (speech playground x2 hidden bars, status_dashboard hidden bar, enhanced_sidebar x2
`classes="hidden"` indicators, CodeRepoCopyPaste hidden overlay, detailed_progress x2
indeterminate bars) and the rest are visible-while-mounted (wizards, dialogs, splash, stats);
uniform conversion makes the guard exemption-free. `CCPLoadingWidget` is NOT permanently dead:
`LoadingManager.setup()` mounts it on Personas and `with_loading` handlers show it during
operations, so pause-while-hidden is the right shape there, not unmounting.

**Guard.** `Tests/Architecture/test_progress_widget_clock_guard.py`: no package file may
construct or subclass stock `ProgressBar`/`LoadingIndicator`/`Bar` (type references for
`query_one`/`isinstance` stay legal). AST scan resolves import aliases, attribute spellings and
the subscripted-generic call form; seven positive-control fixtures keep the scanner honest (one
caught a real scanner gap - `import textual.widgets` binds `textual` - during development).
Extending the timer census itself is TASK-23028 (wave B); notes for it are in the task report.

**Evidence.** Live interleaved A/B (tmux-driven real app, fresh isolated profile, 15 s pause-free
windows, fires counted by patching `Timer._tick`, CPU via `getrusage`):

| screen | arm | progress-widget fires / 15 s | total fires | idle % of a core |
|---|---|---|---|---|
| Lab (F7) | shipped | 960 / 962 / 960 | 1017 / 1019 / 1017 | 0.732 / 0.797 / 0.558 |
| Lab (F7) | fixed | **0 / 0 / 0** | 57 / 57 / 57 | **0.094 / 0.095 / 0.095** |
| Personas | shipped | 240 / 240 / 240 | 285 | 0.341 / 0.215 / 0.223 |
| Personas | fixed | **0 / 0 / 0** | 45 | **0.066 / 0.064 / 0.071** |

The shipped arm reproduces the review exactly (review: 1018 fires, 0.616%). 15 Pilot tests prove
both halves (hidden => 0 fires with a stock-widget control proving the harness measures; shown =>
fires AND `render_indeterminate().highlight_range` moves) plus lifecycle (remove-while-hidden
stops every timer task; app exit clean; live Ctrl+Q from Lab exits 0). Six mutations (M0 dead
harness, M1 no pause-at-create, M2 dead `_on_hide`, M3 dead `_on_show`, M4/M4b compose drift,
M5 site regressed to stock, M6 timer escaping pump teardown) each killed by named tests.

**Files.** New: `tldw_chatbook/Widgets/pausable_progress.py`,
`Tests/Widgets/test_pausable_progress.py`,
`Tests/Architecture/test_progress_widget_clock_guard.py`. Modified: 17 site files (see the
commit) - construction class swaps plus imports only.
