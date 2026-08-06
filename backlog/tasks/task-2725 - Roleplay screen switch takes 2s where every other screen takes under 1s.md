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
- [ ] The cause is identified by profiling and recorded in the task notes (not guessed).
- [ ] No functional regression in the four Roleplay modes (Characters, Personas, Dictionaries, Lore all render their data).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
