---
id: TASK-23021
title: >-
  Setup-modal snow costs 4.3% of a core on every new user's first screen
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
labels:
  - performance
  - idle
  - console
priority: high
---

## Description

The setup modal's snow backdrop burns **4.33% of a core at idle versus 0.096% with the tick
neutralised - 45x**. It is the screen every *new* user lands on, before they have typed anything.

TASK-21134's two fixes are intact and are **not** the problem. `_tick` itself costs 30 ms/15 s. The
cost is `Screen._on_timer_update` at **555-712 ms/15 s (15-19 ms per repaint)**, because the backdrop
is a full-viewport `Static` whose update dirties all 483 widgets on the screen - **20-25x what the
same screen's caret blink costs**.

Measured identical against the 08-24 pin, so this is pre-existing, not a delta.

## Acceptance Criteria

- [x] Idle CPU on the unconfigured first-run screen is brought near the 0.096% floor, or the decoration is retired
- [x] The fix addresses the whole-screen dirty, not the tick rate - lowering the rate again is explicitly not sufficient
- [x] Measured with an interleaved A/B, CPU sampled over a window containing no `pilot.pause()`
- [x] `reduce_motion` still freezes it; the setting's existing meaning is unchanged
- [x] What a user sees is stated plainly in the PR

## Evidence

Interleaved A/B, `on/off/off/on` x3, sole change = `ConsoleSetupBackdrop._tick` -> no-op:

| arm | median | all six runs |
|---|---|---|
| shipped | **4.33%** | 3.54 3.61 4.28 4.38 7.58 9.05 |
| tick neutralised | **0.096%** | 0.056 0.058 0.093 0.098 0.105 0.105 |

`console_setup_modal.py:79` `_SNOW_TICK_INTERVAL = 0.4` -> `:238` `_tick` -> `:259-280`
`_render_flakes` allocates a 170x48 list-of-lists (8,160 cells), joins ~8 KB, and updates a
full-viewport `Static`.

On constrained hardware: **13-30% of a core, permanently, for a decoration.** The burn is also
unusually load-sensitive (both arms drifted 5->9% together under load), which is itself the warning.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.

## Implementation Plan

1. Reproduce the finding on the real unconfigured first-run screen (mounted `ChatScreen` +
   blocking `ConsoleSetupModal` at 170x48, app-CSS tier loaded) and instrument the exact
   mechanism: per-repaint `Screen._on_timer_update` wall time, compositor renders-per-update
   (widgets actually re-rendered), and the dirty-region census (region count, union bbox,
   rows, full-screen?).
2. Measure the candidate mechanisms BEFORE choosing, interleaved in one process, 15 s
   getrusage windows containing no `pilot.pause()`: shipped Static tick, tick-neutralised
   floor, a per-cell-dirty "render_line" simulation (refresh only the changed 1x1 cells),
   and a minimal 3x3-cell repaint at the same 2.5 Hz.
3. Implement whichever mechanism the numbers support; prove where the dirty region ends
   (dirtied-widget count per tick, before and after).
4. Rewrite the animation-contract tests to pin the new mechanism structurally (no CPU
   thresholds as gates); mutation-check every test against a deliberately re-animated build
   and a deliberate teardown bug.
5. Walk unmount/quit mid-display; keep `reduce_motion`'s meaning intact; state plainly what
   a user sees; final interleaved A/B fixed vs legacy-emulated; preflight.

## Implementation Notes

**The animation is retired; the decoration stays.** The backdrop now renders its snow field
as a STILL frame -- same dim, same density, same ·/•/* glyphs behind the Get-started card --
drawn once per (re)size, with no timer and no repaints in between. What a user sees: the
snow no longer drifts; everything else about the first-run screen is unchanged.

**Why retirement and not a cheaper repaint -- measured, not assumed.** All probes ran on the
real unconfigured Console (mounted ChatScreen, blocking modal, field 170x44, 141 visible
widgets, app-CSS tier loaded), arms interleaved in ONE process, 15 s `getrusage` windows with
no `pilot.pause()` inside. Diagnosis A/B (ABBA x3, all runs):

| arm | cpu% (all runs) | per-repaint | renders/update | dirty rows |
|---|---|---|---|---|
| shipped Static tick | 4.280 / 3.618 / 3.743 | 13.1-15.8 ms | 124 | 44/44 |
| tick neutralised | 0.044 / 0.040 / 0.043 | - | 0 | 0 |
| per-cell dirty (render_line shape) | 3.359 / 2.704 / 3.632 | 9.9-13.6 ms | 124 | 43.9/44 |
| single 3x3 repaint @2.5 Hz | 0.583 / 0.537 / 0.551 | 1.7-1.8 ms | 31 | 3 |

This reproduces the review to the millisecond (15.81 ms per repaint in the first window) and
kills the two "keep the animation" candidates named in the filing:

- **Per-cell dirty regions do not end the whole-screen dirty.** Textual's
  `Compositor.render_partial_update` crops to the *bounding box* of the dirty regions, and
  `_get_renders(crop)` re-renders every widget whose clip overlaps that crop. Flakes span
  the field, so the bbox stays ~full-viewport and the compositor still walks 124 widget
  renders x 44 rows -- 2.7-3.6%, statistically the shipped burn. A `render_line` rewrite
  would inherit exactly this footprint.
- **Even a minimal animation cannot approach the floor.** A single 9-cell repaint at the
  same 2.5 Hz costs ~0.55% -- ~13x the floor and ~2x the app's worst normal idle destination
  (0.26%) -- because ~30 widgets stack under any cell of this overlay. On this screen ANY
  repeating repaint at any useful cadence stays an order of magnitude above the floor.

So the durable fix (owner ruling: stability over quick wins) is no repaint at all. The
reduced-motion presentation (TASK-2154.10's static frame) becomes the only presentation.

**Final interleaved A/B on the fixed build** (fixed vs the retired tick re-emulated
faithfully by a probe-side 0.4 s task issuing the same full-field
`Static.update(~8 KB, layout=False)`; ABBA x3, all runs):

- legacy: 7.367 / 5.831 / 4.936 / 3.231 / 2.036 / 2.836 % (load-sensitive, as the review
  warned -- the machine was busier in the early windows; interleaving puts that drift on
  both arms)
- fixed: **0.026 / 0.033 / 0.046 / 0.039 / 0.022 / 0.025 %** -- at/below the 0.096% floor
  (below the tick-neutralised arm, since no timer fires at all)
- dirtied-widget count per tick: **124 -> 0**; `Screen._on_timer_update` work:
  259-1008 ms per 15 s -> **0.0 ms**; repaints per 15 s: 37 -> 0.

**Changes.**

- `tldw_chatbook/Widgets/Console/console_setup_modal.py`: `ConsoleSetupBackdrop` loses the
  tick timer, physics (`speed`/`wobble`), `resume_snow`/`pause_snow`/`timer_paused`, and its
  `reduced_motion` branch; `_SnowFlake` is position+glyph only; `_render_flakes` runs only
  from the mount/resize path (where `layout=True` is correct). `ConsoleSetupModal` drops
  `_sync_snow_timer`/`on_mount`; its `reduced_motion` property remains as the recorded app
  preference the ChatScreen writes each guidance sync (meaning unchanged -- the field it
  used to freeze is frozen for everyone).
- `Tests/UI/test_console_setup_backdrop_repaint_cost.py`: rewritten as the TASK-23021
  contract -- structural assertions, no CPU gates: an AST guard that the module defines no
  `set_interval`/`set_timer`; zero timers and zero `refresh` calls over idle windows longer
  than three old tick intervals (slept, not pilot-paused); deterministic still frame;
  resize-then-still; the blocking-modal path; and the unmount/quit-mid-display walks.
- `Tests/UI/test_console_reduced_motion.py`, `Tests/UI/test_console_rail_sections.py`:
  animation-behaviour tests replaced by stillness/no-timer equivalents; the reduce_motion
  round-trip through the modal is still pinned in both settings.
- `Docs/User_Guide/console.md`: Get-started section now describes the still backdrop;
  Verified-against stamp updated (mounted-harness check on the real unconfigured screen).

**Mutation results.** Mutant 1 (re-animated build: 0.4 s `set_interval` re-added on mount
driving a field repaint): **10 of the 12 snow tests fail** (AST guard, all no-timer /
no-repaint / stillness assertions, both reduced-motion tests, both rail-sections tests);
the 2 survivors are the lifecycle walks, which discriminate teardown bugs, so Mutant 2
(`on_unmount` raising) was run separately: **both lifecycle tests fail**. Clean build:
all 55 tests across the three files pass. Restores were Edit-based.

**Quit walk.** `test_unmount_mid_display_is_clean` (modal.remove() while blocking) and
`test_quit_mid_display_is_clean` (run_test teardown from the blocking state) both pass;
with no timers left on the widget the historical timer-outlives-widget class is
structurally impossible, and the AST guard keeps it that way.

**Not adopted (pre-existing on the pristine tip, byte-identical failure sets verified via
`git archive` tree against the same venv):** the timer-census unclassified-sites failure
(`console_chat_store.create_kwargs` + `chat_screen._expanded_tool_output_ids` -- the known
red), 3 `test_console_modal_dismissal` inventory failures (`AttributeError: module
'tldw_chatbook.Image_Generation' has no attribute 'worker'` inside the launch-graph AST
walk), and `test_product_maturity_phase1_empty_setup_states[watchlists]` (Watchlists screen
compose `AttributeError`).
