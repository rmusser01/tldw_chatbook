---
id: TASK-16323
title: Verify and roll out Console AGENTS.md support
status: To Do
assignee: []
created_date: '2026-08-20 15:33'
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
- [ ] The Console rail and Context surface cover Off, Choose folder, None, loaded-count, warning, binding-recovery, source-precedence, scope, omission, and nested-activation states without displaying automatically loaded bodies outside explicit payload inspection.
- [ ] Cold startup resolution remains O(1), first nested activation remains O(depth), and deterministic concurrency/performance evidence is recorded against a deep synthetic tree.
- [ ] Optional isolated live verification succeeds with at least one native cloud provider and one fenced/local-model path, including nested activation, retry, and multimodal input when supported.
- [ ] User and developer documentation explains discovery, precedence, scope, trust, persistence, consent, configuration, read-only behavior, warnings, and the deliberate differences from Codex and Claude Code.
- [ ] Full focused and affected regression suites, static analysis, formatting checks, security checks, and license checks pass with no automatic instruction-body leakage.
- [ ] ADR-069, all three Backlog tasks, implementation notes, verification evidence, and any genuinely reusable lesson learned are complete and internally consistent.
<!-- AC:END -->
