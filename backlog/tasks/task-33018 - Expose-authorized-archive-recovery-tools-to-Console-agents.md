---
id: TASK-33018
title: Expose authorized archive recovery tools to Console agents
status: Done
assignee:
  - '@codex'
created_date: '2026-09-27 15:26'
updated_date: '2026-09-27 18:19'
labels: []
dependencies:
  - TASK-33015
  - TASK-33017
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make recovery available under either Library retrieval mode with the approved authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reserved recovery tools coexist with Direct or RAG through authenticated runtime registration.
- [x] #2 Only eligible interactive primary runs with Library access execute; blocked, unattended, temporary and subagent paths refuse.
- [x] #3 Production dispatch preserves execution gates and does not expose recovery through shared MCP descriptors.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md
Reason: direct implementation of approved authority-bound runtime/agent provider seam.
1. Read Task5 brief and prior registry/coordinator contracts.
2. Add failing exact-class/capability/reservation/admission and actual Console dispatch tests.
3. Add issuer-owned user-origin proof at real UI producers, propagate immutable turn/queue provenance, fail closed for missing proof.
4. Integrate narrow recovery provider with frozen Library Allowed, native-call ID, existing persona/execution gates, retained runtime/confirmation hooks and teardown.
5. Verify Direct/RAG-only paths and targeted regressions, final UTF-8 serialization bound, memory-store thread ownership, scoped static checks and independent review.
No full suite; native restore callback takes app-only call_id per recorded ruling.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented authority-bound archived-chat recovery tools through native Console dispatch, preserving frozen Library access, user-producer provenance, native run/call identity, persona approval and execution capacity. Runtime retains typed results and joins storage through run/session close and shutdown; included a narrowly verified baseline terminal-close ordering repair. ADR-166 and approved plan apply. Independent task review and scoped fix review are clean. 44 focused recovery/inventory checks, 2 close/dispose checks and 6 fix regression checks passed; new-file static checks pass and no legacy diagnostics were introduced. Required broader regressions produced 27 failures individually reproduced at the pre-task base; destination disclosure qualification remains Task33020. Task33019 supplies mounted controls. Source commits640d27e258 and417e1713b1.
<!-- SECTION:NOTES:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
