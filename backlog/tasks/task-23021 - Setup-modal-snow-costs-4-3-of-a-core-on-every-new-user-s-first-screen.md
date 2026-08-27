---
id: TASK-23021
title: >-
  Setup-modal snow costs 4.3% of a core on every new user's first screen
status: To Do
assignee: []
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

- [ ] Idle CPU on the unconfigured first-run screen is brought near the 0.096% floor, or the decoration is retired
- [ ] The fix addresses the whole-screen dirty, not the tick rate - lowering the rate again is explicitly not sufficient
- [ ] Measured with an interleaved A/B, CPU sampled over a window containing no `pilot.pause()`
- [ ] `reduce_motion` still freezes it; the setting's existing meaning is unchanged
- [ ] What a user sees is stated plainly in the PR

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
