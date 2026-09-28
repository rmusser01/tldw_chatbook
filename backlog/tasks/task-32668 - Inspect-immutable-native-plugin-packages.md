---
id: TASK-32668
title: Inspect immutable native plugin packages
status: To Do
assignee: []
created_date: '2026-09-16 04:16'
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
