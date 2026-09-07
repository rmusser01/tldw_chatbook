---
id: TASK-1912
title: Duplicate a character-probe bench
status: To Do
assignee: []
created_date: '2026-08-02 04:15'
labels:
  - evals
  - character-probe
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Evals inspector offers Duplicate for a word bench but not for a character-probe
bench. `duplicate_bench`'s `BenchConfig`/`save_bench` round-trip rejects a
`CharacterProbeConfig`'s shape, so the affordance needs a `duplicate_character_bench`
counterpart rather than a widened gate.

Delete already works for both bench types, so there is no trap here — this is
missing convenience, not a dead end. Deferred from the character-probe Phase 2
authoring UI (`feat/character-probe-phase2-1691`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A character-probe bench can be duplicated from the same affordance a word bench uses
- [ ] #2 The copy carries the probe set, character selection, targets, and sampler of the original
- [ ] #3 The copy takes a unique name and does not disturb the original
<!-- AC:END -->
