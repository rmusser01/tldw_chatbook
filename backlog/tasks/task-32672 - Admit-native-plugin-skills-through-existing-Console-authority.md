---
id: TASK-32672
title: Admit native plugin skills through existing Console authority
status: To Do
assignee: []
created_date: '2026-09-16 04:19'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32671
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the first usable local plugin path through existing skill, tool and context services with workspace-specific eligibility.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reviewed native skill can install disabled, enable in a named workspace and run through the existing Console skill/tool path under ordinary permissions.
- [ ] #2 Stable workspace and installation generations are rechecked before injection, launch, approval acceptance and invocation; one installed revision applies across scopes.
- [ ] #3 Plugin context stays attributed and untrusted with whole-block limits; manual-only and inline/fork metadata, empty tool allowlists and dependencies preserve their meaning.
- [ ] #4 Owned skill listings expose package provenance while service-level standalone edit/delete/overwrite paths refuse package mutations; unrelated standalone skills remain usable.
<!-- AC:END -->
