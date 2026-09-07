---
id: TASK-18922
title: 'Model popover: one-turn model override (for next send only)'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - models
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's `/model --once` (2026-08-19 hermes-release review). In the Alt+M quick Model popover, add a "just for the next send" mode: the chosen provider/model applies to exactly one accepted send, then the session reverts to its previous model automatically — no manual switch-back. Use case: try a stronger model on one hard prompt, or a cheaper one on a throwaway, without disturbing the session's model. Session-scoped only; never writes config. Complements the model-catalog selector work (ADR-020).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Alt+M popover offers a one-turn override affordance; the next accepted send uses the override and the following send reverts to the session's prior model with no user action
- [ ] #2 While armed the override is visible (Model section / status chip, e.g. "model (next send only)") and can be cancelled before the send
- [ ] #3 Consumption rule pinned and tested: consumed only on an accepted send; a failed or stopped turn does not silently consume it
- [ ] #4 Mid-run guard respected: the override cannot be armed or applied while a run is streaming, consistent with existing model-switch behavior
- [ ] #5 Tests cover arming, consumption-on-accept, revert, cancel, display, and that nothing is written to config
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: session-scoped presentation over the existing model-selection seam (ADR-020 catalog work governs the selector itself); no storage/boundary change.

1. Extend session model state with an optional one-shot override slot
2. Send path: apply override on accepted send, clear + revert after
3. Alt+M popover UI: affordance, armed indicator, cancel
4. Tests + User Guide console.md model-selection section update
<!-- SECTION:PLAN:END -->
