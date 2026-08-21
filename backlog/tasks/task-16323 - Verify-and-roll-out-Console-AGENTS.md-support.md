---
id: TASK-16323
title: Verify and roll out Console AGENTS.md support
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-20 15:33'
updated_date: '2026-08-21 04:17'
labels:
  - console
  - agents
  - verification
  - docs
dependencies:
  - TASK-16322
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the Console project-instruction UX, provider interoperability, performance evidence, live verification, and user documentation required for a safe release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Console rail and Context surface cover Off, Choose folder, None, loaded-count, warning, binding-recovery, source-precedence, scope, omission, and nested-activation states without displaying automatically loaded bodies outside explicit payload inspection.
- [ ] #2 Cold startup resolution remains O(1), first nested activation remains O(depth), and deterministic concurrency/performance evidence is recorded against a deep synthetic tree.
- [ ] #3 Optional isolated live verification succeeds with at least one native cloud provider and one fenced/local-model path, including nested activation, retry, and multimodal input when supported.
- [ ] #4 User and developer documentation explains discovery, precedence, scope, trust, persistence, consent, configuration, read-only behavior, warnings, and the deliberate differences from Codex and Claude Code.
- [ ] #5 Full focused and affected regression suites, static analysis, formatting checks, security checks, and license checks pass with no automatic instruction-body leakage.
- [ ] #6 ADR-069, all three Backlog tasks, implementation notes, verification evidence, and any genuinely reusable lesson learned are complete and internally consistent.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the established Console project-instruction UI tests for every final rail/Context state, recovery transition, activation summary, literal label, key/focus contract, and required viewport.
2. Implement only the tested state mapping, warning aggregation, recovery actions, and explicit nonpersistent Next Send projection within the existing Neon Workbench Console surfaces.
3. Update the three Console user guides and root AGENTS.md with discovery, authority, trust, consent, persistence, configuration, read-only, warning, and ecosystem-difference guidance.
4. Run the approved focused UI, Ruff, formatter-policy, responsive pilot, detector, and diff gates; inspect the scoped changes and record Task 13 evidence.
5. Complete the separate Task 14 performance, provider UAT, broad verification, sentinel audit, and final task closeout without treating those later gates as Task 13 work.

ADR required: yes
ADR path: backlog/decisions/069-console-project-instruction-local-state-and-preflight.md
Reason: ADR-069 already governs the provider/runtime trust boundary, local-only state, binding authority, and UX disclosure model; Task 13 implements that accepted decision and requires no new ADR.
<!-- SECTION:PLAN:END -->
