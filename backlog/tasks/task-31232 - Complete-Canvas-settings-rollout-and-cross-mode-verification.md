---
id: TASK-31232
title: Complete Canvas settings rollout and cross-mode verification
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03'
updated_date: '2026-09-05 19:11'
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
- [x] #6 Native live verification covers create, update, submit draft, download, historical selection, undo, and temporary save/destruction
- [x] #7 Served live verification covers split pane, hot reload, branch switch, exact transcript reopen, authenticated remote/proxy flow, and two-browser isolation
- [x] #8 Archive 3.0 round-trip and zero-egress evidence are captured through the outermost user-visible paths
- [ ] #9 Targeted Canvas, Console, database-migration, Chatbooks, web-server, packaging, and browser suites pass; a full repository sweep is run only with explicit user approval
- [x] #10 Final review regressions demonstrate production staging count/byte admission, supported DOM move/reinsert behavior, byte-preserving ordinary disabled continuations, nonblocking transcript Canvas actions, and pin-preserving settlement publication; helper limit validation and close/hide copy match their actual contracts
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
9. Apply one consolidated final-review correction wave under existing ADR-115: enforce production staging admission, preserve virtual DOM move identity, keep ordinary non-opt-in continuations byte-compatible, remove synchronous transcript compatibility compilation, and retain historical pins on ordinary publication. Correct helper limit validation and explicit Close versus Hide wording. Add exact failing regressions before each fix, run affected targeted checks/statics, and obtain one scoped rereview; do not fix unrelated baseline failures or weaken runtime/security contracts.
10. User-authorized additional focused pass: enforce the distinct default 8 MiB temporary-session ceiling across committed and concurrently staged history; support detached edits/restructuring and exact empty/false form-state reconstruction; fence late compile-refusal repair against disable, owner/session and exact source-block changes. Write failing production-path/browser regressions, implement under existing ADR-115, run affected targeted checks and one scoped rereview. Prior I3/I5/M1/M2 closures and unrelated baseline scope remain unchanged.
11. User-approved DOM-only correction: restore explicit select values after rebuilt options and handle mixed-presence descendants without duplicate native IDs. Add actual-renderer RED/GREEN cases for non-first/empty selections and new/live children under detached parents; preserve default controls, identity, cycles and limits. Verify targeted DOM/assets/statics and owned browser cleanup, then one scoped rereview from 648530ac6 under existing ADR-115. No unrelated baseline work or full sweep.
12. User-authorized six-baseline repair: reproduce the exact six IDs in Docs/Canvas/V1_VERIFICATION.md, distinguish fixture drift from product failures, and repair only their causes and directly affected regression coverage. Existing ADR-097 (retained soft-delete semantics and guarded mutations), ADR-079 (Library authority), ADR-094 (raw CLI consent) and ADR-115 apply; no new ADR required. Preserve guards/privacy and exercise real image-only send, MCP filtering, promotion, export/deletion and Settings readiness contracts. Run targeted RED/GREEN, affected checks/statics and independent scoped review; reconcile AC9 honestly without a full repository sweep or integration.
13. User-authorized retry correction: reproduce the two failed-assistant retry cases, trace exact run/assistant settlement and durable projection through retry, and repair their shared cause or stale expectation without weakening atomic commit, discarded failed history, cleanup, rollback or restart hydration. Existing ADR-115 applies; no new ADR required. Coordinator runs isolated targeted tests/statics, worker performs static edits only; obtain independent scoped review and reconcile AC9 without full sweep or integration.
<!-- SECTION:PLAN:END -->
