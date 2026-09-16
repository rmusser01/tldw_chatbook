---
id: TASK-32668
title: Inspect immutable native plugin packages
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:16'
updated_date: '2026-09-16 05:51'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32645
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users a bounded, inspectable native package inventory with stable identity and explicit compatibility blockers before any code can execute.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Portable Agent Plugins 1.0.0 and the closed Chatbook extension normalize identity, component inventory and dependencies without executing or fetching package content.
- [ ] #2 Manifest limits, contained paths, links, executable mode and platform collisions are checked while materializing a snapshot; provenance and effective digests are reproducible.
- [ ] #3 Malformed recognized extensions retain inspectable portable components while blocking activation when required constraints are unknown; successful guarded packages remain usable.
- [ ] #4 Inspection reports distinct support, selection, readiness and evidence axes, and ambiguous dialect candidates require an explicit choice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: Direct implementation of accepted plugin inspection and constraint-preservation boundaries.
1. Establish RED tests against native inspection/materialization and independent portable fixtures.
2. Implement frozen inventory, native extension and dependency validation with locally pinned portable rules.
3. Implement bounded filesystem capture, secure containment/link adaptation, identity and reproducible digests.
4. Exercise every scoped failure with valid controls; run exact Plugins tests plus changed-file static checks.
5. Self-review, commit, independent task review, then complete ACs and implementation notes. Vendor qualification stays in I2; F1 records candidate identity/ambiguity without claiming vendor readiness.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented non-executing native inspection and bounded POSIX snapshot materialization under ADR-162/163. Frozen records separate format support, selection, disabled availability and parsed evidence; native skills/MCP/commands/rules/agents/hooks retain dependency constraints and unknown-scope blockers. Snapshot digests bind copied bytes and executable mode, while a separate source digest preserves internal-link provenance. Vendor candidates retain identity/ambiguity but remain unqualified. Added original licensed fixtures, authoring/operation documentation and 64 feature tests; 103 tests pass including the unchanged 39 path-validation neighbors, with Black, Ruff and syntax checks passing. macOS filesystem evidence only; Windows descriptor capture fails closed pending qualification. Independent review and AC/status closeout remain controller-owned.
<!-- SECTION:NOTES:END -->
