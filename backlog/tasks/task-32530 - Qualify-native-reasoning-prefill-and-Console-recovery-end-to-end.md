---
id: TASK-32530
title: Qualify native reasoning prefill and Console recovery end to end
status: To Do
assignee: []
created_date: '2026-09-13 03:22'
labels: []
dependencies:
  - TASK-32521
  - TASK-32526
  - TASK-32527
  - TASK-32528
  - TASK-32529
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish release evidence for actual native continuation and the complete Console lifecycle across the supported target inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Console provider family and API mode has an evidence-backed support classification; at least one native target is live-qualified before enabled execution ships.
- [ ] #2 Advertised tool support has live multi-round evidence; wire and available rendered-boundary evidence, echo attribution, streaming and fallback outcomes identify exact tested constraints.
- [ ] #3 Targeted Console lifecycle, queue, recovery, persistence, exchange, privacy, existing-prefill, and UI checks pass or document unrelated baseline failures; no full sweep runs without user opt-in.
<!-- AC:END -->
