---
id: TASK-31200
title: Define the llama.cpp Lab-to-Console connection and readiness contract
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 12:46'
updated_date: '2026-09-03 13:23'
labels:
  - llamacpp
  - lab
  - console
  - ux
  - architecture
dependencies: []
documentation:
  - backlog/decisions/114-llamacpp-lab-console-connection-authority.md
  - Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md
  - >-
    Docs/superpowers/plans/2026-09-03-task-31200-llamacpp-lab-console-contract.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the cross-surface contract that turns a Lab-owned llama.cpp process or an existing llama.cpp endpoint into an explicitly adopted Console destination. Resolve the current 8001, 8080, and 9099 divergence and define endpoint, model, lifecycle, persistence, privacy, and ownership truth before implementation.

ADR required: yes. This task must author the governing ADR because it changes the provider/runtime boundary and long-lived application structure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ADR chooses one canonical credential-free connection descriptor and one default/fallback policy shared by Lab launch, local discovery, Console readiness, and Console execution.
- [ ] #2 The ADR distinguishes process reserved, process alive, API healthy, model available, Console session adopted, and saved default states, with one named authority for each transition.
- [ ] #3 The contract defines Start on this computer, Connect to an existing server, Use in this Console session, and Make default as explicit actions; no action silently overwrites a different saved endpoint.
- [ ] #4 The contract keeps credentials, executable paths, GGUF paths, and raw logs within their owning surface while permitting the sanitized endpoint, provider identity, model identity, and readiness evidence needed by Console.
- [ ] #5 The ADR reconciles ADR-002, ADR-025, ADR-095, TASK-16473, TASK-16476, and TASK-26837 and records backward-compatible handling for existing llama.cpp configuration.
- [ ] #6 The ADR includes the end-to-end state sequence, failure settlement rules, observability boundary, and verification strategy for real loopback HTTP and mounted Textual flows.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/114-llamacpp-lab-console-connection-authority.md
Reason: TASK-31200 changes the provider/runtime boundary, cross-screen state ownership, persistence scope, privacy boundary, and long-lived setup flow.

1. Author ADR-114 with the sanitized LlamaCppConnectionTarget, separate runtime/connection/product states, explicit action ownership, canonical absent-value port 8080 policy, compatibility rules, and verification obligations.
2. Reconcile ADR-002 exact-endpoint discovery, ADR-025 GGUF/process-lease authority, ADR-095 Console/default ownership, and TASK-16473/TASK-16476/TASK-26837 behavior.
3. Index ADR-114 and link it from the approved handoff wireframe, this implementation plan, and TASK-31200.
4. Self-review the contract against all six acceptance criteria, reject placeholders and ambiguous ownership, and verify documentation integrity without changing or testing production code.
5. Accept ADR-114, complete TASK-31200 metadata, and commit only the ADR, index, wireframe, plan, and task record.
<!-- SECTION:PLAN:END -->
