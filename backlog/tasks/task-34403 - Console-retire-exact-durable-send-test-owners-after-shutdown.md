---
id: TASK-34403
title: 'Console: retire exact durable-send test owners after shutdown'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-05 19:29'
updated_date: '2026-10-05 20:40'
labels:
  - console
  - testing
  - resource-ownership
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3024'
documentation:
  - Docs/QA/task-33620.9/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Durable-send and durable-acceptance qualification accumulates real SQLite handles from local test helpers and direct owners that never retire their own controllers or database registries. Make the affected test modules own their exact teardown so physical resource evidence is meaningful without changing production lifetime policies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every database created by this module’s controller and ready-store helper calls is retired only after its associated live controller shutdown succeeds.
- [x] #2 Direct module-owned evidence controllers use the same exact-owner teardown; foreign databases and shared configuration owners are untouched.
- [x] #3 Original send, recovery, token-fencing, queue and retention assertions remain unchanged, with raw warnings and strict descriptor results recorded.
- [x] #4 The three real-SQLite queued-recovery controls register their exact database/controller pairs and retire them after supported shutdown without changing recovery assertions or closing shared-profile owners.
- [x] #5 All acceptance-module helper-created and directly constructed database/controller owners retire through the existing opt-in fixture without changing rollback, accepted-cancellation or publication assertions; every original module teardown leaves no SQLite descriptors.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve exact durable-send strict RED, then implement opt-in local helper capture and direct registration using supported shutdown/quiescence. 2. Verify the unchanged twelve controls and complete durable-send module with raw diagnostics/strict resources, retaining prior failed receipts and unchanged 1,000-send count/timeout. 3. Queue-recovery extension: reproduce the three real-SQLite queued recovery cases in isolation; reuse the same exact-owner retirement body as an explicitly requested shared Tests/Chat fixture, retain the durable module's local helper wrappers, and register only each queued test's own controller/database pair. Keep failed drains loud and files retained; no global constructor/shared-profile/cache changes. 4. Verify affected non-retention controls and shared-fixture ownership/failure contracts, static and artifact guards, independent review and evidence; do not duplicate unchanged 1,000-send bodies or claim native/scale qualification. 5. Acceptance extension: retain the broader 349-pass strict failure and isolated 19-pass strict RED with 55 descriptors; wrap only the acceptance module's own ready-store helper binding using the existing exact-owner fixture and register the four direct accepted-cancellation controllers/databases before starting work. Retain every rollback, policy, attachment, cancellation and publication assertion. Verify original 19 plus affected preparation/durable non-retention and existing failed-drain/foreign-owner controls with strict resource observations, static/guards and independent review before publication. No production/helper implementation/global constructor, shared-profile or foreign-owner change. ADR required: no. ADR path: N/A; existing ADR120/198 apply. Reason: test-only reuse of incumbent exact-owner shutdown/quiescence; no storage/runtime/service/GC or lifetime-policy change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented opt-in exact ownership in the affected durable-send test module only: local helper wrappers capture returned DB/controller owners, the direct evidence case explicitly registers, and existing awaited shutdown precedes same-file quiescence. Failed drains retain their own files and rethrow while independent owners still retire; no production/shared-config/global constructor/GC or warning-filter changes. Strict RED: same12 bodies pass8.34s but process exits1 with48SQLite descriptors. GREEN: same12 pass8.16s no warnings strict0 zeroDBfiles. Complete module25pass336.58s no warnings strict0/zeroDBfiles at every teardown; original1000-send count/timeout unchanged, case323.71s. Combined25 shutdown/postcommit/maintenance controls pass9.31s strict0/zeroDBfiles but retain2 raw intentional closed-loop ContextVar warnings. Independent review has no actionable scoped finding. All11 artifact guards pass; test format clean,2 inherited Ruff findings unchanged/no additions. Existing ADR120/198 apply; no new ADR. QA receipt Docs/QA/task-33620.9/README.md. Keep In Progress while broader PR review/qualification and inherited static debt remain unmet; do not claim native/scale or warning-free emergency qualification.

Queued-recovery extension reuses the existing finalizer in an explicitly requested Tests/Chat fixture; durable wrappers remain module-local and only the three real-SQLite queue cases opt in. Published-head RED3 pass4.09s/no warnings but strict1 retains12SQLite descriptors. GREEN28 including non-retention durable controls and new real-DB failure/foreign-owner check pass17.50s/no warnings strict0/zeroDBfiles at every teardown. Original1000-send body/count/timeout unchanged and explicitly deselected here; previous complete-module receipt remains historical. Independent scoped review has no actionable finding. All11 artifact guards pass, all3 paths format clean,6 inherited Ruff diagnostics unchanged/no additions. Logs /tmp/switcher-queued-retirement-Kk49kF/{red,green,preflight}.log. No production/shared-profile/cache/GC/diagnostic change; existing ADR120/198. Broader seven-file/emergency/native/Windows/participant/scale and Qodo-credit gates remain open; task stays In Progress.
Acceptance extension: original19 bodies pass12.85s/no warnings but strict1 retains55SQLite descriptors, matching the broader349-pass failed resource receipt. Only this modules ready-store binding now captures returned databases; four directly built accepted-cancellation pairs register before submit creation. The real helper and sibling imported bindings are unchanged. Existing shared awaited shutdown/quiescence and failed-drain/healthy/foreign-owner control reused, all behavioral assertions retained. GREEN133 acceptance/complete-preparation/durable non-retention controls pass63.31s/no warnings, strict0, all133 DB-file censuses empty; original1000-send body not duplicated. Changed test format clean, three inherited Ruff findings unchanged. Independent scoped review finds no actionable issue. QA retains all prior failures; existing ADR120/198, no new production/global/foreign/cache/GC or warning policy. Keep In Progress for broader corrected batch and native/Windows/participant/scale/current-head external review, not waived.

Acceptance publication gate: all eleven artifact guards pass; raw receipt /tmp/switcher-emergency-context-ciQIs5/acceptance-preflight.log. No qualification or external-review waiver.
<!-- SECTION:NOTES:END -->
