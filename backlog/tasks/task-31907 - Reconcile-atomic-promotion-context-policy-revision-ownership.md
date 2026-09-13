---
id: TASK-31907
title: Reconcile atomic promotion context-policy revision ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:55'
updated_date: '2026-09-05 20:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair the context-policy promotion regression without suppressing genuine staged-policy failures or weakening atomic persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A policy staged through the public Store path promotes with matching live and durable revision ownership and no false settings failure.
- [x] #2 Policy transaction failures still roll back without live publication and genuine staged-policy failures remain observable.
- [x] #3 The complete affected promotion and settings-policy tests and scoped static checks pass.
- [x] #4 Inherited revision-zero sparse policy retains atomic bundle persistence and publishes revision ownership that permits subsequent Apply.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the original stale helper sentinel and public real-SQLite staged/fork revision failures.
2. Add RED public-flow regressions for staged success, postcommit failure/retry, and inherited fork policy followed by Apply; retain bundle rollback coverage.
3. Under ADR-095 send revision-positive staged policy only through the existing postcommit writer. Keep revision-zero inherited policy in the bundle and publish its known fresh-row revision None/1 following the existing durable-turn receipt precedent.
4. Replace the old helper-invocation sentinel with the actual write seam; run complete related files and scoped static checks, then parent review before commit.
ADR required: no
ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md
Reason: Routine repair implementing existing staged-settings post-promotion ownership, preserving ADR-079 bundle publication; no new interface, schema, reload, or ownership policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected promotion ownership under ADR-095: publicly staged revision-positive context policy uses only the existing postcommit CAS writer and failure ledger; revision-zero inherited policy stays in the atomic bundle and publishes its fresh-row None/1 revision following the existing durable-turn receipt. No interface or schema change and no postcommit reload. The old regression now intercepts the actual write seam rather than a harmless helper call.
Public-flow real-SQLite regressions cover empty/nonempty staged success, durable conversation retention and exact-policy retry after a real postcommit failure, and inherited temporary fork policy followed by successful Apply. New tests quiesce all same-file connections, including retry worker handles, and assert zero registered connections afterward. Existing atomic rollback coverage remains unchanged.
Evidence: RED 4 expected failures/3 controls (/private/tmp/tldw-31744-red.xml). Final complete promotion/settings Store files: 63 passed, 1 existing Requests warning, 30.78s (/private/tmp/tldw-31744-promotion-settings-final.xml). Related complete files: 386 behavior tests passed plus 24 census tests passed; 3 census failures reproduced identically with exact HEAD Store source (/private/tmp/tldw-31744-census-head.xml) and remain separately tracked, not waived. Related aggregate run also emitted a 216-descriptor growth warning; no resource-wide cleanup or threshold change was applied.
Whole Store and test Ruff lint passed, test whole-file format and changed complete Store method format passed, and git diff whitespace passed. Whole Store format check was already failing at HEAD and is not claimed green. Parent independently reviewed production and regressions, approved scope and baseline accounting before commit. ADR required: no new ADR; implements existing ADR-095 and preserves ADR-079 publication.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31744 was renumbered to TASK-31907 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
