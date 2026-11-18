---
id: TASK-41524
title: "The Evals word bench falls back to http://localhost:8080 when the configured llama.cpp URL is refused"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- evals
- providers
priority: medium
references:
- tldw_chatbook/Evals
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in endpoint fix r1 (round 15), after the I-3 fix. When the configured llama.cpp URL is refused, the bench shows "not configured", but an existing unsteered target still sends to the hard-coded `http://localhost:8080` fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A refused or missing configured URL sends nothing, and the bench says why
- [ ] #2 The fallback address is removed or becomes an explicit setting
- [ ] #3 A test shows no request reaches the fallback
<!-- AC:END -->
