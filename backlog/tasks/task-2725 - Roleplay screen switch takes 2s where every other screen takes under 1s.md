---
id: TASK-2725
title: Roleplay screen switch takes ~2s where every other screen takes under 1s
status: Done
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
- [x] Roleplay tab switch (click → screen interactive) is within 2× the median of the other screens on the same profile, measured before/after with the same method.
- [x] The cause is identified by profiling and recorded in the task notes (not guessed).
- [x] No functional regression in the four Roleplay modes (Characters, Personas, Dictionaries, Lore all render their data).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Investigation Notes (2026-08-06, profiling done — fix deferred)

<!-- SECTION:NOTES:BEGIN -->
Profiled via cProfile around `push_screen(PersonasScreen)` in the standard test-app harness (235x52): **3.2s**, and the cause is NOT DB I/O. The screen mounts **494 widgets, 358 of them inside `#personas-detail-stack`** — every mode's full detail surface is composed eagerly (character card + character editor + persona card + persona editor + dictionary/lore detail + transcript + two try-it panels: 108 Buttons, 21 Inputs, 17 TextAreas), almost all `display: none` on arrival. Cost concentrates in CSS application against the app-wide stylesheet: `stylesheet.apply` 2,644 calls / 2.30s cumulative, 7.77M selector `__hash__` calls, plus ~500 full `update_styles` subtree sweeps triggered by `add_class`/`set_class` mutations during the mount storm (331 + 1,011 calls). Widget count is the lever; per-widget CSS cost is the multiplier.

**Recommended fix (own PR):** defer composition of the non-active mode's detail widgets (mount on first mode activation). NOT landed in the 2720-2726 batch deliberately: 81 in-screen references + 4 external modules address these widgets, and `restore_state` runs before mount — deferral needs a centralized accessor plus an audit of every query site, which is its own reviewable change, not a rider. `textual.lazy.Lazy` alone is insufficient for the same reason (queries during `on_mount`/restore would hit unmounted children).

AC1/AC3 remain unchecked pending that change; AC2 (profiled cause) is satisfied by this note.
<!-- SECTION:NOTES:END -->

## Implementation Plan (fix round, 2026-08-07)

<!-- SECTION:PLAN:BEGIN -->
Scoping probe: the four heavy hidden center views carry 290 of the stack's 358 widgets — `PersonasCharacterEditorWidget` 132, `PersonasDictionaryDetailWidget` 67, `PersonasLoreDetailWidget` 60, `PersonaProfileEditorWidget` 31. Architecture check: `_show_center` already tolerates missing nodes (skips absent selectors) and the editor funnels through `_editor_or_none`, so deferred mounting is supported by existing tolerance.

Design: **defer-past-first-paint**, not mount-on-demand — compose the four heavy views nowhere; mount them (hidden, order-anchored) as the FIRST step of `_load_after_mount`, before `_apply_pending_restore`/auto-select. Every query site downstream of load sees the full DOM; only the compose→load window must tolerate absence, which `_show_center`/`_editor_or_none` already do. Total work unchanged; the click→paint critical path drops by ~59% of the screen's widgets.

1. RED A: with `_load_after_mount` stubbed, pushing the screen mounts none of the four and the widget total drops below 300 (was 494).
2. Guard B (passes before AND after): after a real load settles, all four are present, hidden, and the stack's child order is exactly the historical document order.
3. GREEN: compose drops the four yields; `_mount_deferred_center_views()` mounts them with `after=` anchors and `display=False`; `_load_after_mount` awaits it first.
4. Full personas suites + live tmux latency before/after with the walkthrough method.
<!-- SECTION:PLAN:END -->

## Implementation Notes (fix round)

<!-- SECTION:NOTES:BEGIN -->
Shipped the defer-past-first-paint design: `compose_content` no longer yields the four heavy hidden center views (character editor 132 widgets, dictionary detail 67, lore detail 60, persona profile editor 31 — 290 of the screen's 494); `_mount_deferred_center_views()` mounts them hidden and order-anchored (`after=` the widgets compose used to place them behind) as `_load_after_mount`'s first step, before `_apply_pending_restore`/auto-select, so every downstream query site sees the full DOM. The compose→load window is covered by existing tolerance (`_show_center` skips absent selectors; `_editor_or_none` returns None) — chosen over mount-on-demand precisely to avoid auditing the 85+ query sites. Idempotent for re-entered loads.

Results: tab-switch latency 2.19s/1.97s → **0.69–0.94s** live (same tmux click→highlight method, same seeded profile; other screens 0.26–0.47s, so within 2× median). Tests: mechanism pin (load stubbed → none of the four mount, widget total <300) watched RED first; integrity guard (all four present+hidden after load, stack document order exactly the historical sequence) green before AND after. Full personas suites 474 passed; live: all four modes cycled, character selected, deferred editor opened via Edit and cancelled back to the card, zero errors/warnings in both logs. Files: tldw_chatbook/UI/Screens/personas_screen.py, Tests/UI/test_personas_deferred_center_views.py, Docs/User_Guide/roleplay-chat-dictionaries.md (stamp).
<!-- SECTION:NOTES:END -->
