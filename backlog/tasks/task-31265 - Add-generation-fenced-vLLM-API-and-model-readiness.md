---
id: TASK-31265
title: Add generation-fenced vLLM API and model readiness
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:32'
updated_date: '2026-09-04 02:15'
labels:
  - vllm
  - lab
  - readiness
dependencies:
  - TASK-31263
  - TASK-31264
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace process-liveness completion with an explicit, privacy-bounded vLLM lifecycle that proves the OpenAI-compatible API and served model are ready.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Lab distinguishes not configured, checking, launching, loading model, ready, stopping, and failed states.
- [x] #2 Ready requires a current-generation bounded models-endpoint probe and an admissible exact served-model identity.
- [x] #3 Cancellation, target edits, process death, recomposition, and newer checks prevent stale results from enabling actions.
- [x] #4 Activity and recovery expose bounded categories without retaining credentials, raw commands, paths, or unrestricted child output outside the Lab-owned boundary.
- [x] #5 Unit, loopback HTTP, lifecycle, privacy, and mounted UI tests cover the state machine.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow Docs/superpowers/plans/2026-09-03-vllm-lab-console-complete-redesign.md Task 2 and ADR-117.
2. Add RED owner, loopback probe, cancellation, process-exit, privacy, and mounted recomposition tests.
3. Implement the app-scoped VllmConnectionOwner, bounded activity, credential-aware health/models probing, and LLMScreen lifecycle orchestration.
4. Run the focused Task 2 and incumbent lifecycle suites, self-review, and record exact evidence.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already fixes the connection owner, generation fencing, privacy boundary, lifecycle ownership, and rollback behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an app-scoped immutable vLLM connection owner with exact generation/fingerprint/runtime fencing, bounded allowlisted Activity, credential-aware bounded health and model-list probes, exact local alias verification, admissible existing-server model IDs, and sanitized failure categories. Moved Check/Start/Retry/Stop orchestration to LLMScreen while retaining shared server-lifecycle claims and reducing the legacy event module to picker/compatibility glue. Added readiness/Activity UI projection plus source-side suppression for programmatic Textual field updates, and covered draft edits, raw arguments, cancellation, process death, screen detach, recomposition, response bounds, privacy canaries, and stale settlement. Focused evidence: readiness/UI 33 passed; prescribed filtered readiness 7 passed/26 deselected; incumbent lifecycle 31 passed; incumbent vLLM setup/action 34 passed/17 deselected; Ruff and focused mypy passed; git diff --check passed. ADR required: no. ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md. ADR-117 already fixes ownership, fencing, privacy, lifecycle, and rollback.

Fix Round 1 binds the immutable launch snapshot to the exact shared lifecycle claim, retries live processes only from that binding, rejects cancelled claims, and restores exact runtime ownership across draft invalidation and screen replacement. READY results now require a canonical credential-free target, exact `chatbook-vllm` identity for owned launches, and fail-closed owner revalidation. Stop-before-publication settles as cancellation, preflight failures settle into the authoritative owner, and Stop enablement is derived independently from exact live process ownership. Final focused evidence: readiness/workflow 45 passed; lifecycle/status 31 passed; Task 1 setup compatibility 34 passed; deferred-view compatibility 2 passed/7 deselected; Ruff, focused mypy, and `git diff --check` passed.

Fix Round 2 closes the remaining owned-target identity gap: READY settlement for a Chatbook-owned token now derives the canonical completion endpoint from the exact claim-bound launch snapshot, requires the exact bound operation token and fingerprint, and refuses any other canonical endpoint. External-server targets remain claim-independent. The canonical port-8001 mutation against a port-8000 claim failed before the fix and passes after it; final focused evidence is 48 readiness/workflow and 31 lifecycle/status tests passing, with Ruff, focused mypy, and `git diff --check` green.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-31215. During the branch integration sweep,
current `origin/dev` already shipped `task-31215 -
Personas-mount-heavy-center-views-on-first-use.md` at add commit
`2516735cfd27df249ab45e96c96f15b8aee35d15`. The unmerged vLLM task therefore
moved to collision-free TASK-31265, carrying every dependency and documentation
reference with it. The vLLM record was originally added by
`ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0`.
