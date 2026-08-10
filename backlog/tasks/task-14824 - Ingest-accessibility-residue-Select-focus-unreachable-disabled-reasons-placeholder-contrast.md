---
id: TASK-14824
title: >-
  Ingest accessibility residue: Select focus, unreachable disabled reasons, placeholder contrast
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
labels:
  - library
  - ingest
  - accessibility
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 of the 2026-08-10 re-critique — three gaps the structural-focus and disabled-reason work missed, each measured live.

1. **`#opt-generic-encoding` focus is colour-only.** A per-focusable Tab walk found the Select's focused and unfocused plain-text captures byte-identical; only the background changes, at 1.12:1 between the two. Every other canvas focusable shows a glyph-level change (`┏━┓` on inputs, `┃label┃` on buttons, a `┃` marker on collapsible titles). The 13 global nav tabs are colour-only too, but those are out of this surface's scope.

2. **Disabled fields are keyboard-unreachable, hiding the reasons written for them.** The Audio & video group contributes exactly 2 tab stops (its collapsible title and Reset to defaults) because all 13 option fields and the Parakeet install button are disabled. A keyboard-only user can therefore never land on any of them to read the `— needs X installed` annotation that task-3304 added specifically so inert controls explain themselves. The explanation is currently mouse-and-eyes-only.

3. **Input placeholders measure ~3.5:1 in both states.** Enabled placeholder 3.52:1, disabled placeholder 3.49:1 — below AA for normal text, and a 0.03 delta means a placeholder-only field has effectively no colour cue for its disabled state. Related: the path Input has NO visible label at all — its identity is placeholder-only and vanishes once populated, the same defect task-2012 fixed for option fields, still present on this surface's primary control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Every focusable on the ingest canvas, including Selects, shows a glyph-level focus change asserted by a plain-text render diff
- [ ] #2 A keyboard-only user can reach the explanation for a disabled option without a mouse (either the control stays focusable-but-inert, or its reason is surfaced somewhere keyboard-reachable)
- [ ] #3 The path field carries a persistent visible label rather than a placeholder that disappears on input
- [ ] #4 Placeholder text meets the contrast floor, or placeholders are not the sole carrier of a field's identity or state
<!-- AC:END -->
