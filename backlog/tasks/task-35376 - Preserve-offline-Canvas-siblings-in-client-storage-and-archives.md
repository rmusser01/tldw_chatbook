---
id: TASK-35376
title: Preserve offline Canvas siblings in client storage and archives
status: Done
created_date: 2026-10-18 05:54
dependencies:
- TASK-31003
labels:
- canvas
- sync
- client
priority: medium
assignee:
- '@codex'
updated_date: 2026-10-18 16:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the local artifact contract to preserve concurrent offline revisions without changing immutable source, ownership or branch projection. Implement the approved ADR-239 revision-order and archive compatibility foundation while Canvas synchronization remains disabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A genuine current-schema database upgrades atomically and matches a fresh database; original Canvas values, owner/source protections and exactly-one-root enforcement are preserved, with rollback and retry covered.
- [x] #2 Complete valid graphs permit equal-sequence siblings while retaining positive contiguous distinct sequence values, strictly lower same-Canvas parents, quotas and deterministic sequence/UUID ordering. Ordinary local settlement still allocates max(sequence)+1 and rejects stale authored sequence claims.
- [x] #3 A new explicit Canvas-bearing Chatbook archive contract round-trips tied graphs and exact source/profile/origin metadata; existing valid 3.0 archives remain readable, non-Canvas exports retain their existing format, and legacy readers reject new archives before partial import.
- [x] #4 Targeted repository, branch-resolution, migration, archive, backup-validation and packaging qualification plus documentation cover the changed contract; no transport, runtime capability or host-profile migration is activated.
- [x] #5 Response-rule migration and recovery regressions qualify the installed current endpoint after the Canvas schema upgrade, preserving all accepted historical lineages, user history and byte-stable rejection of unsupported candidates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes; existing accepted ADR path: backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md. Reason: direct implementation of its approved sibling-order/schema/archive contract; no duplicate ADR. 1. Follow Docs/superpowers/plans/2026-10-17-canvas-sync-client-foundation.md under isolated synthetic state. 2. Add RED/GREEN schema/repository/recovery ownership and sequence controls, then implement migration and typed graph admission without changing authored settlement safety. 3. Add RED/GREEN major-version archive and canonical-order/legacy-refusal controls, implement all creator/importer callers and exact compatibility. 4. Independently review each task and the full foundation, run only affected tests/static/package checks, update living documentation and record evidence before closing.
PR #3153 CI follow-up: ADR required: no new ADR; existing ADR-239 applies. Hosted Console P0 job 117726289469 exposed five old schema-81 endpoint pins after the legitimate schema-82 migration. Reproduce both affected files, make their installed endpoint assertions current-version-relative while retaining exact frozen lineage and unchanged-source checks, update the renamed test in both CI selectors, run both files plus Canvas migration/recovery and CI selector contracts, and obtain scoped independent review before pushing. Keep Canvas synchronization paused.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the approved local sibling-graph foundation under accepted ADR-239 (backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md), following Docs/superpowers/plans/2026-10-17-canvas-sync-client-foundation.md. Schema 82 transactionally replaces sequence uniqueness with canonical sequence/UUID ordering and a native single-root constraint, preserving immutable values and serial authored max+1 allocation. Repository graph validation is bounded by revision count. Chatbook 4.0 / Canvas extension 2.0 round-trips tied graphs; valid legacy 3.0/1.0 and no-Canvas exports remain supported. Exact frozen core/hybrid recovery catalogs retain all four v81 predecessors and qualify installed v82 migration. Changed DB/migration/recovery, Canvas repository/archive, Chatbook model/creator/importer, focused tests, index census classification and living architecture/archive documentation; packaging discovery needed no new registry. Review found and fixed legacy V1/V2 Canvas-member dropping with refusal before extraction/database access, plus date-labeled an earlier pre-approval closeout. Independent task, scoped correction, whole-branch and final prose re-reviews are complete, with no remaining findings. Evidence: 103 final model/archive tests, seven fresh real-package/installed checks, 30 independent final-parser confirmations, ten migration controls, eleven recovery controls and derived preflight checks passed. Broader affected run was 264 passed / one migration-source census failure reproduced identically on exact base for seven unrelated SQL files. Scoped Ruff has 762 identical inherited per-file diagnostics, zero introduced; twelve baseline-clean files format successfully and four retain pre-existing formatting debt. No global-green claim. Root checked package/source bytes and all 5504 Backlog records. Qualification and limitations: Docs/Canvas/SIBLING_GRAPH_FOUNDATION_QUALIFICATION_2026-10-17.md; raw local receipts /private/tmp/chatbook-canvas-sync-foundation.KPZ9cL. Scope clarifications were exact installed 81/82 hybrid recovery, count-bounded contiguous validation and correctness-only index census classification; no old catalog or guard relaxation. Sync remains disabled: no transport, runtime authority, server write, normal-profile migration, dependency install, full-suite, hosted/platform/server qualification, push or merge. TASK-35377–35383 remain To Do; schema/archive allocations must be rechecked before integration.
2026-10-18 PR #3153 CI repair: the legitimate schema-82 upgrade exposed five schema-81 endpoint pins in the hosted Console P0 step. Full response-rule migration/recovery modules reproduced 16 stale-pin failures (14 passed). Test-only correction uses the installed current endpoint, exact ordered frozen catalogs for all four historical lineages, and a genuinely unsupported future version; retained history, FK, source-byte stability and cold constructor-free recovery checks remain intact. Updated all three renamed-selector consumers in the existing workflow and CI registries; independent review caught and root reproduced the third registry miss (one failed/53 passed) before correction. Qualification: 223 focused migration/recovery/lane cases passed; 995 CI contracts passed with one intentional nested-child skip; exact failed 120-case workflow step passed at unchanged timeout=180. Selections overlap. Edited regression modules lint/format clean; two existing Console P0 registry lint findings and critical-registry formatter debt are unchanged against base. Independent scoped review has no remaining findings. Updated Docs/Canvas/UAT_2026-10-18.md and the incident-backed migration-pin lesson. Existing ADR-239 applies; no production logic, migration, recovery policy, dependency, job, quota or sync activation changed. Raw receipts: /private/tmp/chatbook-canvas-pr-20261018.2eyGc8. Hosted results and protected merge remain pending; no full-suite/normal-profile/live-provider/server-interop qualification is claimed.
Final CI-repair preflight passed every derived-artifact, task and census check (ci-repair-preflight.log); git diff --check passed. The scoped requesting-code-review substitute review has no Critical, Important or Minor findings after the third-selector correction. Latest fetched dev remains b07fc0425288fb51d17cde5f745051720a7c9db6 with no branch divergence from its reviewed base. Local implementation qualification is complete; publishing and protected hosted-CI/merge follow-up remain owned by PR #3153. Sync tasks35377–35383 and spike35490 remain To Do.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the local Canvas sibling-graph foundation and the test-only PR #3153 CI compatibility repair under ADR-239. All acceptance criteria and targeted local checks pass; independent review has no remaining findings. Hosted CI and protected merge are tracked by PR #3153, not claimed complete here. Server/client Canvas synchronization remains paused and unimplemented.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
