---
id: TASK-32668
title: Inspect immutable native plugin packages
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:16'
updated_date: '2026-10-02 00:50'
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
ADR required: yes (existing accepted contracts). ADR paths: backlog/decisions/162-managed-agent-plugins.md and backlog/decisions/163-expanded-console-hook-runtime.md. 1. Read the native inspection and resource contracts and reviewed F1 checkpoint. 2. Reuse reviewed F1 fixtures and demonstrate missing package inspection through the intended entry. 3. Port bounded immutable capture, closed normalization and materialization without activating runtime code; include the reviewed portable MCP syntax repair. 4. Run native inspection, file-boundary, provenance and neighboring validation tests in the isolated worktree. 5. Self-review, record current evidence and platform limits, complete ACs and notes, mark Done via CLI and commit exact task files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused reviewed F1 package inspection with its reviewed portable MCP syntax repair. Native component inventory, closed constraints, deterministic content/effective identity and descriptor-anchored bounded materialization publish no runtime authority. Targeted native inspection/file tests: 92 passed, zero skips/warnings; all 14 new Python files pass Ruff and formatting. Real macOS filesystem boundaries exercised; Windows fails closed and Linux/vendor execution remain unqualified. Existing ADR162/163 and authoring/provenance docs apply. Evidence: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. Implementation files: Plugins models, schemas, inspection, package_files and portable adapter; original test fixtures.

PR #2946 Qodo review: the existing installed Pydantic dependency now strictly validates known manifest/author fields while retaining unknown top-level inspection fields and the dictionary API. Fixed-code errors omit raw external values; public Google-style contract documented. Existing ADR-162 applies. Current RED/GREEN and native manifest evidence: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.
<!-- SECTION:NOTES:END -->
