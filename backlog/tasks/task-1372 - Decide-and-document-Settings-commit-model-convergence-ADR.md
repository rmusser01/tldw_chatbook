---
id: TASK-1372
title: Decide and document Settings commit-model convergence (ADR)
status: Done
assignee: []
created_date: '2026-08-05 23:38'
updated_date: '2026-08-05 23:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique Question to Consider: three commit models (staged draft, labeled instant-apply, guarded advanced config) are each honestly labeled — is 'honest about complexity' the goal, or should Settings converge on one model with instant-apply as the justified exception? Resolve the question and record the decision as an ADR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR created in backlog/decisions recording the commit-model decision and rationale
- [x] #2 Alternatives (single converged model, status quo) considered in the ADR
- [x] #3 ADR linked from the task Implementation Notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract the three commit models from settings_screen.py (staged draft, labeled instant-apply, guarded TOML)
2. Resolve the re-critique convergence question
3. Record decision as ADR-033 in backlog/decisions/
ADR required: yes
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md
Reason: task is itself the architectural decision the ADR records
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved the re-critique's open question in favor of **honest pluralism**: keep three commit models (staged draft as default, labeled instant-apply as the restricted exception, guarded raw TOML as expert-only), and manage the complexity by labeling truthfulness per control rather than by eliminating models. Converging to a single model was rejected both directions — staged-everywhere adds friction and failure modes to harmless toggles; instant-everywhere is unsafe for validated/cross-field settings.

- Decision recorded: `backlog/decisions/033-settings-commit-models-three-honestly-labeled.md`
- Normative labeling mechanism already in code: `STAGED_SAVE_BEHAVIOR_COPY` / `INSTANT_APPLY_BEHAVIOR_COPY` in `tldw_chatbook/UI/Screens/settings_screen.py` (task-1341)
- Consequence for reviews: new Settings fields must state which model they follow; critique passes should score label truthfulness per control, not model count.

Modified/added files: `backlog/decisions/033-settings-commit-models-three-honestly-labeled.md` (new), this task file.
<!-- SECTION:NOTES:END -->
