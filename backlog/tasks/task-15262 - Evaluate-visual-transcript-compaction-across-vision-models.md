---
id: TASK-15262
title: Evaluate visual transcript compaction across vision models
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 15:33'
updated_date: '2026-08-11 16:12'
labels:
  - console
  - context
  - evals
dependencies: []
references:
  - backlog/tasks/task-14914 - Add-deterministic-visual-transcript-compaction.md
  - backlog/decisions/054-deterministic-visual-transcript-compaction.md
priority: medium
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Produce reproducible, model-specific evidence for visual transcript compaction so users and maintainers can see its real token cost, latency, recovery quality, and safety limits before any model is recommended or the default policy is reconsidered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A versioned representative corpus covers prose, structured constraints, code, math, tool results, Unicode, and adversarial transcript text
- [x] #2 A reusable evaluator runs text and visual representations through an explicitly selected vision-capable provider/model without storing transcript or credentials
- [x] #3 Reports distinguish measured provider usage from estimates and record render plus end-to-end latency
- [x] #4 Quality scoring reports OCR fidelity, code/math recovery, instruction recall, and adversarial-text behavior with unknown values never treated as passing
- [x] #5 A generated support-matrix artifact records model, renderer, corpus, and evaluator versions and makes no default recommendation without all gates passing
- [x] #6 Automated tests cover corpus determinism, result validation, redaction, unknown metrics, and recommendation gating
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define a versioned synthetic benchmark corpus and immutable evaluation/report contracts that contain no credentials or private transcript data.
2. Run paired text and visual requests through the production Console provider preparation and dispatch seam, recording provider usage, rendering latency, end-to-end latency, and strict quality scores.
3. Add a confirmation-gated CLI that writes only an atomic, content-free support matrix and keeps help/refusal paths outside application config initialization.
4. Cover corpus determinism, strict validation, redaction, measured-versus-estimated usage, unknown metrics, adversarial behavior, and recommendation gating with automated tests.
5. Perform the smallest authorized live evaluation against an explicitly selected vision model, then document the result and preserve text-summary default behavior.

ADR required: no

ADR path: `backlog/decisions/054-deterministic-visual-transcript-compaction.md`

Reason: This task implements ADR-054's existing model-evaluation gate without changing storage, provider ownership, or default policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented a versioned synthetic corpus, strict content-free evaluation and support-matrix contracts, paired text/visual execution through the production Console prepared-request gateway, and a two-call confirmation-gated CLI. Reports include measured-versus-estimated usage, render/request latency, renderer/corpus/evaluator identity, strict recovery and adversarial scores, and evidence-derived recommendation invariants; raw transcript, image bytes, credentials, endpoints, and model output are not persisted. Help and refusal paths use lazy application imports so they cannot initialize a user profile.

The isolated live `openai/gpt-4o` run left the real config hash unchanged. It measured 914 text input tokens versus 1,835 visual input tokens across two pages (-100.8% reduction), with a 56 ms render. The visual response was invalid under the strict JSON contract, so quality metrics remain unknown and the model is not recommended. Text summary remains the default under ADR-054.

Verification: Ruff format/check passed; 19 focused evaluator/renderer tests passed; an injected-provider integration probe through the real Console gateway completed two prepared requests with measured usage. The wider gateway suite is blocked on Windows before test execution because the dev-branch network guard intercepts pytest-asyncio's loopback `socketpair`; the same representative test fails identically on detached `dev` at `45587bc9d`.

ADR required: no. Existing ADR: `backlog/decisions/054-deterministic-visual-transcript-compaction.md`.
