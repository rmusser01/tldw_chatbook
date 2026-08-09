---
id: TASK-3314
title: >-
  Unify ingest consent inline — retire the guardrail modal
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-08 20:30'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved by the owner via task-3310 (ruling 2). The critique's "one consent grammar" question: guaranteed failures gate inline while missing tooling raises a blocking modal — the worse outcome gets the quieter treatment, and consent changes shape with failure type. The owner ruled to fold tooling-warning consent into the inline commit/gate grammar and retire `IngestGuardrailModal` entirely (its rendering was fixed in tasks 3300/3304, so this is a consolidation, not a bug fix). The inline preflight warnings already carry per-warning copy-install-command buttons (task-3304), so the modal's information is already on the canvas — only its consent step remains to move.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Starting an import with active tooling warnings requires an explicit second confirmation carried by the inline commit/gate grammar (the repo's incumbent two-press pattern), naming how many files may fail
- [ ] #2 `IngestGuardrailModal` and its tests are removed; no modal appears on any Start path
- [ ] #3 The copy-install-command affordance remains reachable at the inline warnings
- [ ] #4 Starts with no warnings are unchanged (single press); Esc/blur resets a pending confirm state
<!-- AC:END -->
