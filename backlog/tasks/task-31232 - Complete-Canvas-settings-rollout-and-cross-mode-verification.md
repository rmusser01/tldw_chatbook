---
id: TASK-31232
title: Complete Canvas settings rollout and cross-mode verification
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03'
updated_date: '2026-09-05 09:53'
labels:
  - canvas
  - settings
  - documentation
  - verification
dependencies:
  - TASK-31228
  - TASK-31229
  - TASK-31230
  - TASK-31231
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the Canvas V1 product boundary with canonical settings, conservative measured quotas, user and model guidance, a kill switch, and evidence that the complete native and served workflows satisfy the approved architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The canonical Settings screen exposes the Canvas enable/auto-open controls and measured hard quotas with clear security and compatibility copy
- [x] #2 One kill switch removes Canvas tools and HTML-block actions and makes browser routes/control connections fail closed in both modes
- [x] #3 Conservative quota defaults are fixed from explicitly labeled synthetic assistant-authored page, compiler, virtual-engine, and browser-memory probes and enforced consistently across all boundaries; measured compilation work above 100 ms stays off interactive event loops without bypassing lifecycle or scope checks
- [x] #4 User documentation explains preview-first workflow, temporary/durable history, revisions/branches, source safety, runtime compatibility, remote authentication, and recovery
- [x] #5 Model/tool guidance uses Canvas only when the visual medium materially helps and generates against the supported V1 runtime profile
- [ ] #6 Native live verification covers create, update, submit draft, download, historical selection, undo, and temporary save/destruction
- [ ] #7 Served live verification covers split pane, hot reload, branch switch, exact transcript reopen, authenticated remote/proxy flow, and two-browser isolation
- [ ] #8 Archive 3.0 round-trip and zero-egress evidence are captured through the outermost user-visible paths
- [ ] #9 Targeted Canvas, Console, database-migration, Chatbooks, web-server, packaging, and browser suites pass; a full repository sweep is run only with explicit user approval
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: this delivery completes the long-lived Canvas product/security boundary already accepted by ADR-115: global enablement, effective ceilings, recovery/operations guidance, and cross-mode release evidence. The ADR will be amended with measured final values and verified operational behavior rather than duplicated.

1. Add canonical F9 Settings controls/status for Canvas enablement and auto-open plus read-only effective quotas; implement one global kill switch across tool advertisement, message actions, native/served gateways, and control channels while preserving stored/exportable data.
2. Build reproducible content-free compiler/runtime/browser probes, measure the current ceilings on the supported baseline, and only retain or lower limits based on responsiveness and interruption evidence.
   Resolve the measured native/served preview and HTML-block import compiler scheduling gap before rollout: keep expensive pure compilation outside interactive loops and shared authority locks, bound admitted work, and revalidate captured scope/lifecycle before publication or mutation.
3. Write user, model, security, operations, and recovery documentation for the exact V1 workflow and runtime subset, explicitly deferring bundled libraries, multi-file projects, elevated capabilities, and server sync.
4. Extend and run outermost native, served, archive, and zero-egress flows through user-visible paths, including branch/history, temporary lifecycle, confirmed actions, remote authentication, two-browser isolation, and archive restore.
5. Run targeted Canvas/Agents/Console/database/Chatbooks/Web Server/packaging/browser checks and changed-file static analysis, then ask the user whether they want the optional full repository sweep.
6. Request independent code/security/UX review, check every Canvas task/design invariant, update ADR-115 and this task with evidence, and mark Done only after its Definition of Done is satisfied. TASK-31003 remains To Do as the explicit sync-contract follow-up.
7. Repair the live-reproduced exact-card selection delivery gap: child authority selects the historical root but the already-open served renderer stays on the branch. Keep existing ADR-115 capability revocation, pin/follow semantics, shell ownership and stale-response fences. Add targeted gateway and real native/served browser regressions before claiming exact reopen complete; no authentication relaxation or unrelated production repair.
8. Implement ADR-115 selection-intent amendment: fence original browser epoch and expected child conversation/Canvas/revision/generation before mutation; preserve generation through capability/bootstrap round trips and reject missing served expectations. Verify delayed old commands after different-revision and same-revision pin cannot mutate, while passive polling and explicit current navigation retain their distinct behavior.
<!-- SECTION:PLAN:END -->
