---
id: TASK-2725
title: Roleplay screen switch takes ~2s where every other screen takes under 1s
status: To Do
assignee: []
created_date: '2026-08-06 17:00'
labels:
  - roleplay
  - performance
  - uat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Timed tab-switch latencies during the full-app UAT walkthrough on `origin/dev` `b0185749c` (populated profile: 3 characters, 41 conversations; 235x52; latency = click → nav-highlight move, content follows):

| Screen | Latency |
|---|---|
| Workflows, ACP, Artifacts | 0.25s |
| Logs, Settings | 0.46s |
| Library | 0.48s |
| Lab | 0.68s |
| Watchlists | 0.89s |
| Schedules, MCP | 1.11s |
| **Roleplay** | **2.19s / 1.97s (repeat visit)** |

Roleplay is 2–4× slower than every other destination, and the repeat-visit number shows it is recurring screen-construction cost, not one-time module import. Since navigation runs through the FIFO-locked nav worker, those ~2s also delay any queued navigation. Screens are constructed fresh on every visit by design, so whatever Roleplay does in construct/compose (likely synchronous DB loads for characters/personas/dictionaries/lore up front) is paid on every single visit.

Profile before optimizing (per project rules) — the fix is likely deferring per-mode data loads to after first paint, or to the mode actually selected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] Roleplay tab switch (click → screen interactive) is within 2× the median of the other screens on the same profile, measured before/after with the same method.
- [x] The cause is identified by profiling and recorded in the task notes (not guessed).
- [ ] No functional regression in the four Roleplay modes (Characters, Personas, Dictionaries, Lore all render their data).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Investigation Notes (2026-08-06, profiling done — fix deferred)

<!-- SECTION:NOTES:BEGIN -->
Profiled via cProfile around `push_screen(PersonasScreen)` in the standard test-app harness (235x52): **3.2s**, and the cause is NOT DB I/O. The screen mounts **494 widgets, 358 of them inside `#personas-detail-stack`** — every mode's full detail surface is composed eagerly (character card + character editor + persona card + persona editor + dictionary/lore detail + transcript + two try-it panels: 108 Buttons, 21 Inputs, 17 TextAreas), almost all `display: none` on arrival. Cost concentrates in CSS application against the app-wide stylesheet: `stylesheet.apply` 2,644 calls / 2.30s cumulative, 7.77M selector `__hash__` calls, plus ~500 full `update_styles` subtree sweeps triggered by `add_class`/`set_class` mutations during the mount storm (331 + 1,011 calls). Widget count is the lever; per-widget CSS cost is the multiplier.

**Recommended fix (own PR):** defer composition of the non-active mode's detail widgets (mount on first mode activation). NOT landed in the 2720-2726 batch deliberately: 81 in-screen references + 4 external modules address these widgets, and `restore_state` runs before mount — deferral needs a centralized accessor plus an audit of every query site, which is its own reviewable change, not a rider. `textual.lazy.Lazy` alone is insufficient for the same reason (queries during `on_mount`/restore would hit unmounted children).

AC1/AC3 remain unchecked pending that change; AC2 (profiled cause) is satisfied by this note.
<!-- SECTION:NOTES:END -->
