---
id: TASK-31214
title: Add guided vLLM environment and model preflight
status: To Do
assignee: []
created_date: '2026-09-03 22:32'
labels:
  - vllm
  - lab
  - onboarding
dependencies:
  - TASK-31213
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent predictable first-run vLLM launch failures by making environment, model-source, network, and managed-argument readiness visible before Start.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can select either a Hugging Face repository ID or a local model directory through source-appropriate controls.
- [ ] #2 Preflight reports interpreter resolution, vLLM module availability, model-source validity, port availability, and bind-address exposure without starting a server.
- [ ] #3 Start is disabled with a visible field-adjacent reason until required checks pass.
- [ ] #4 Managed host, port, and model flags cannot be duplicated or overridden through raw arguments.
- [ ] #5 Focused unit and mounted Textual tests cover success, failure, preservation, and recovery states.
<!-- AC:END -->
