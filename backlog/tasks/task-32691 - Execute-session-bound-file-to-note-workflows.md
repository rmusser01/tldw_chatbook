---
id: TASK-32691
title: Execute session-bound file-to-note workflows
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:59'
updated_date: '2026-09-16 05:18'
labels:
  - workflows
dependencies:
  - TASK-32690
documentation:
  - Docs/superpowers/specs/2026-09-16-workflows-first-run-design.md
  - Docs/superpowers/plans/2026-09-16-workflows-first-run.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local text-file, prompt, llama.cpp, editable review and Local Note workflow using session-owned execution that survives navigation but not restart, without new SQLite execution infrastructure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A saved immutable workflow revision completes the five-step example through the existing Workflows screen using the selected llama.cpp endpoint and actual model.
- [ ] #2 Run state and review edits survive screen navigation but are never resumed or replayed after app restart; saved definitions drafts and committed Notes remain durable.
- [ ] #3 File model and Note effects use captured destinations and fresh fail-closed authority; Off kill-switch and rejected or expired review prevent subsequent effects.
- [ ] #4 The selected keyless llama.cpp request has zero retries no redirects or proxies bounded response bytes an absolute deadline and physically settled cancellation; no Ollama fallback is used.
- [ ] #5 Note creation preserves existing policy and transactions with one attempt ID verified same-destination readback and commit-wins cancellation reporting; no blind retry or remote sync is dispatched.
- [ ] #6 Duplicate Run and Accept actions cannot duplicate effects; quit confirms loss fences new actions and drains owned work before dependent services close.
- [ ] #7 No new SQLite schema ownership locks PID records helpers or persistent execution writes are introduced and historical workflow rows remain untouched.
- [ ] #8 Targeted automated tests real temporary databases actual-app UI checks and an isolated live localhost:9099 file-to-reviewed-Note run pass with truthful lint and performance evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; direct implementation of the user-approved session-bound amendment to ADR-138. ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md. Reason: storage ownership and service-composition boundaries remain governed by ADR-125 and ADR-036; the approved design already specifies the opt-in model and permission integration. 1. Map exact current source seams and write Docs/superpowers/plans/2026-09-16-workflows-first-run.md from the approved spec. 2. Qualify bounded llama.cpp and strict permission integration with failing/passing targeted tests. 3. Implement bounded local effects and the in-memory sequential session with retained physical operations. 4. Wire existing app lifecycle and Workflows Run/setup/review/Open Note surfaces without redesign. 5. Verify targeted regressions and isolated actual-app/live-model UAT, self-review and independent review, and update docs/task evidence. No implementation code is written during the planning checkpoint.
<!-- SECTION:PLAN:END -->
