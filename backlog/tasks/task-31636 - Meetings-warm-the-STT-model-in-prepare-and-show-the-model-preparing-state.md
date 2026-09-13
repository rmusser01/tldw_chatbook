---
id: TASK-31636
title: 'Meetings: warm the STT model in prepare() and show the model-preparing state'
status: To Do
assignee: []
created_date: '2026-09-05 08:40'
labels:
  - meetings
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec 3.4 says prepare() loads the resolved model so Start is immediate. Phase 1 builds the TranscriptionService facade but never loads a model, so the first segment of every meeting pays the model load. Audio is not at risk (capture is independent of the transcriber), but the first transcript row can lag by the whole model-load time with nothing on screen explaining the wait.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start pressed after the screen has settled does not wait on a model load
- [ ] #2 A model-load failure shows an error state on the rail and records nothing
<!-- AC:END -->
