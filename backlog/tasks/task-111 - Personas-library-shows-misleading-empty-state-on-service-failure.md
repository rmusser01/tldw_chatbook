---
id: TASK-111
title: Personas library shows misleading empty state on service failure
status: To Do
assignee: []
created_date: '2026-06-11 03:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When refresh_persona_list fails or the backend lacks persona support, the workbench library renders the actionable 'No persona profiles yet' empty state instead of a recovery callout (DestinationRecoveryState). Surface service failures distinctly per the destination recovery pattern.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Service failure renders recovery copy distinct from the true empty state,True empty state copy unchanged
<!-- AC:END -->
