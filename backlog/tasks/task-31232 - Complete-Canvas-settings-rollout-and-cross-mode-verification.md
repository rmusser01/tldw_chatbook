---
id: TASK-31232
title: Complete Canvas settings rollout and cross-mode verification
status: To Do
assignee: []
created_date: '2026-09-03'
updated_date: '2026-09-03'
labels: [canvas, settings, documentation, verification]
dependencies: [TASK-31228, TASK-31229, TASK-31230, TASK-31231]
priority: medium
---

## Description

Finish the Canvas V1 product boundary with canonical settings, conservative measured quotas, user and model guidance, a kill switch, and evidence that the complete native and served workflows satisfy the approved architecture.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The canonical Settings screen exposes the Canvas enable/auto-open controls and measured hard quotas with clear security and compatibility copy
- [ ] #2 One kill switch removes Canvas tools and HTML-block actions and makes browser routes/control connections fail closed in both modes
- [ ] #3 Conservative quota defaults are fixed from recorded provider-output, compiler, virtual-engine, and browser-memory probes and enforced consistently across all boundaries
- [ ] #4 User documentation explains preview-first workflow, temporary/durable history, revisions/branches, source safety, runtime compatibility, remote authentication, and recovery
- [ ] #5 Model/tool guidance uses Canvas only when the visual medium materially helps and generates against the supported V1 runtime profile
- [ ] #6 Native live verification covers create, update, submit draft, download, historical selection, undo, and temporary save/destruction
- [ ] #7 Served live verification covers split pane, hot reload, branch switch, exact transcript reopen, authenticated remote/proxy flow, and two-browser isolation
- [ ] #8 Archive 3.0 round-trip and zero-egress evidence are captured through the outermost user-visible paths
- [ ] #9 Targeted Canvas, Console, database-migration, Chatbooks, web-server, packaging, and browser suites pass; a full repository sweep is run only with explicit user approval
<!-- AC:END -->

## Related Design

- `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- `Docs/superpowers/plans/2026-09-03-chatbook-canvas-implementation.md`
- `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`
