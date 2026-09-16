---
id: TASK-32691
title: Execute session-bound file-to-note workflows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:59'
updated_date: '2026-09-16 16:40'
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
- [x] #1 A saved immutable workflow revision completes the five-step example through the existing Workflows screen using the selected llama.cpp endpoint and actual model.
- [x] #2 Run state and review edits survive screen navigation but are never resumed or replayed after app restart; saved definitions drafts and committed Notes remain durable.
- [x] #3 File model and Note effects use captured destinations and fresh fail-closed authority; Off kill-switch and rejected or expired review prevent subsequent effects.
- [x] #4 The selected keyless llama.cpp request has zero retries no redirects or proxies bounded response bytes an absolute deadline and physically settled cancellation; no Ollama fallback is used.
- [x] #5 Note creation preserves existing policy and transactions with one attempt ID verified same-destination readback and commit-wins cancellation reporting; no blind retry or remote sync is dispatched.
- [x] #6 Duplicate Run and Accept actions cannot duplicate effects; quit confirms loss fences new actions and drains owned work before dependent services close.
- [x] #7 No new SQLite schema ownership locks PID records helpers or persistent execution writes are introduced and historical workflow rows remain untouched.
- [x] #8 Targeted automated tests real temporary databases actual-app UI checks and an isolated live localhost:9099 file-to-reviewed-Note run pass with truthful lint and performance evidence.
- [x] #9 Source prompt response and review payloads remain absent from ordinary logs and error diagnostics including resolved Note titles on failed writes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; direct implementation of the user-approved session-bound amendment to ADR-138. ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md. Reason: storage ownership and service-composition boundaries remain governed by ADR-125 and ADR-036; the approved design already specifies the opt-in model and permission integration. 1. Map exact current source seams and write Docs/superpowers/plans/2026-09-16-workflows-first-run.md from the approved spec. 2. Qualify bounded llama.cpp and strict permission integration with failing/passing targeted tests. 3. Implement bounded local effects and the in-memory sequential session with retained physical operations. 4. Wire existing app lifecycle and Workflows Run/setup/review/Open Note surfaces without redesign. 5. Verify targeted regressions and isolated actual-app/live-model UAT, self-review and independent review, and update docs/task evidence. No implementation code is written during the planning checkpoint.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-138's saved-revision, session-bound file -> prompt -> keyless llama.cpp -> editable human review -> Local Note flow in the existing Workflows screen. Reused existing permissions, Notes and SQLite owners; no new schema, persistent execution, locks/PID/helper service, retries or Ollama fallback. Captured exact-effect approvals, navigation-safe review, Open Note and retained physical quit/cancel settlement are covered.

Qualification fixed nested Library mounting and stale editor focus restoration plus privacy-test logging isolation. Eight passing actual localhost:9099/Gemma processes include four same-profile fresh-process restarts without replay. Final branch review found an Open Note/Note-worker circular wait; the existing route guard now supports nonblocking UI checks, manual busy/retry feedback and stale-message cleanup while retaining default blocking worker semantics and destination identity checks. Both fixes have deterministic RED/GREEN. One scoped re-review closed the P1 with no new findings.

Verification: initial final affected run159 passed/1liveSkip; final fix covering229 and separate ProductionApp12 passed; exact final-source controller run11 passed/1liveSkip/1existingwarning in38.16s. No new static debt. Counts overlap and are not additive. Historical merged failures remain recorded, not relabeled green. An unrelated unchanged Skills test-double failure, legacy lint/dependency warning debt and native PTY/font limits are disclosed. No full repository suite or new live requests were run for the final guard fix.

ADR required: no new ADR; existing backlog/decisions/138-portable-workflow-definitions-and-local-execution.md and ADR-125 govern these boundaries. All six task reviews, one whole-branch review and its single fix-wave scoped review are reconciled. Reviewed source head: bcc8290dd2d883893c781a83a3e5cd0f13f27eb5. Evidence: Docs/Developer/Workflows/2026-09-16-first-run-uat.md. Acceptance, rulings and review record: Docs/Developer/Workflows/2026-09-16-first-run-review-record.md. User guide and testing lessons updated. No push, PR or merge performed.
<!-- SECTION:NOTES:END -->
