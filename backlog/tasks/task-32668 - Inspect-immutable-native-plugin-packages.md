---
id: TASK-32668
title: Inspect immutable native plugin packages
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:16'
updated_date: '2026-09-16 06:14'
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
- [x] #1 Portable Agent Plugins 1.0.0 and the closed Chatbook extension normalize identity, component inventory and dependencies without executing or fetching package content.
- [x] #2 Manifest limits, contained paths, links, executable mode and platform collisions are checked while materializing a snapshot; provenance and effective digests are reproducible.
- [x] #3 Malformed recognized extensions retain inspectable portable components while blocking activation when required constraints are unknown; successful guarded packages remain usable.
- [x] #4 Inspection reports distinct support, selection, readiness and evidence axes, and ambiguous dialect candidates require an explicit choice.
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
Implemented non-executing native package inspection and bounded POSIX snapshot materialization under ADR-162/163. Frozen records retain component/dependency constraints and separate support, selection, availability and evidence; reproducible materialized identities bind bytes and executable mode, with source-link provenance recorded separately. Added native fixtures, authoring/operation documentation and malformed-input/resource/default-immutability regression coverage. Independent task review and scoped re-review are clean after fixing three Important findings. Final affected tests: 79 passed; the earlier unchanged path-validation neighbor run passed 39 tests. Changed-file Black, Ruff, syntax and whitespace checks pass. macOS filesystem execution is qualified; Linux remains unverified and Windows descriptor acquisition fails closed. APFS collision-key behavior is tested without claiming distinct case/NFC-NFD entries can be created. Vendor candidates remain explicitly unqualified pending the adapter task. No full-suite run or additional runtime permissions were introduced. ADRs: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. Commits f61db231dc and 0c7e16bd9e.
<!-- SECTION:NOTES:END -->
