---
id: TASK-31242
title: Build character conversation projection and Keyword index
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 02:05'
updated_date: '2026-09-05 21:49'
labels:
  - database
  - search
  - characters
dependencies:
  - TASK-31241
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide one authority-safe local read model and selected-branch Keyword corpus for every later Character navigation surface, including deterministic unresolved-link recovery services without exposing new UI.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-31234 on 2026-09-04. The final pre-commit worktree sweep
found the older `Auto resume always lands on the cursor item` task created at
01:50; it keeps TASK-31234 under the older-arrival rule. This unshipped task
moves with all plan and dependency references.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Resolved and unresolved identities are tagged, bounded, data-authority-scoped, and never inferred from names, display text, filesystem paths, or current selection.
- [x] #2 The projection returns bounded recent groups, global Keyword results, exact totals, and stable keyset pages from local conversations only.
- [x] #3 A canonical eligibility projector includes only the selected visible user/assistant branch and fails closed for invalid branch graphs.
- [x] #4 A separate versioned FTS generation is built and maintained without reusing messages_fts or exposing excluded message content.
- [x] #5 Unique legacy authority links backfill deterministically; ambiguous, missing, and deleted links remain typed Unavailable results.
- [x] #6 Library repair candidates are same-authority only and compare-and-set repair invalidates affected derived search state.
- [x] #7 ensure_keyword_index is dormant until an owning UI calls it and performs no startup background work.
- [x] #8 Migration, isolation, paging, eligibility, concurrency, performance, and no-server-canary tests pass on real SQLite fixtures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reviewed Task2 replay and qualify its isolated PR tree on PR1 head 5539118b; preserve delivery 74993748e and all source/safety refs.
2. Carry audited Task2-owned later fixes and opt-in SQLite owner cleanup; exclude UI, activation, Meaning, settings, and unrelated shared-file changes.
3. Reproduce complete descending (last_modified, created_at, conversation_id) paging on real SQLite before correcting cursor, predicates, browse ordering, and next cursors; cover ties, limit-one traversal, mutation refresh, and unchanged-row continuity.
4. Verify exact identity/migration/projection/selected-branch/ownership tests and affected migration/derived-index guards; run scoped Ruff/format, inherited diagnostic comparison, whitespace and reference checks. Rebase onto merged PR1 dev only after tests stop.
ADR required: no
ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md
Reason: faithful existing authority/storage/pagination contract.
Final schema version: 66; current origin/dev 7aa048790 is schema 65, preserving shipped migrations.
5. Fix round1: TDD prior-ready snapshot metadata/availability, atomic SQLite build claims and fenced finalization, indexed-text-only FTS trigger; keep v66 fresh/upgrade parity, capture compact raw logs and rerun exact scope without rebase.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prepared the independently reviewable Task2 projection/Keyword slice from PR1 5539118b. Replayed the four reviewed Task2 commits and carried later owning fixes for strict identity decoding, exact-character Keyword filtering, unavailable paging, fenced repair/preview reads and opt-in SQLite resource evidence. Corrected cursor, row metadata, SQL keyset predicates and resolved/unavailable recent ordering to descending (last_modified, created_at, conversation_id), preserving Keyword relevance ordering and all authority/revision fences. Five real-SQLite ordering cases failed before the correction and pass afterward. Existing source timestamps required no additional migration: schema remains unshipped v66 after shipped v65.
Packaging qualification also registered the dedicated DDL module with the schema guard, allowlisted its five actual tables, updated the current-version migration pin, and recorded no-statistics plans for the five existing indexes. A focused RED/GREEN guard fix recognizes standalone index names beyond idx_/uq_ while still excluding negative assertions. No threshold, inventory exclusion or pre-convention exemption was added.
Final exact identity/migration/projection/selected-branch/ownership plus affected schema/SQL-validation/index-guard gate: 124 passed, 1 environment RequestsDependencyWarning in 26.14s. Opt-in cleanup: 65 owners, 40 handles drained, 0 registered remaining; process FD delta +13 is subset evidence, not whole-process resource clearance. New-file Ruff/format passed; legacy Ruff diagnostic counts and message multisets unchanged (ChaChaNotes 588, sql_validation 6, schema checker 1, index checker 1). Schema/index/task-ID, CSS bundle, profile-owned-path and diagnostic inventories passed. Complete PR-range whitespace passed.
ADR required: no; backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md governs. Lesson added for the observed guard source/name assumptions. Task remains In Progress pending independent review, controller-owned PR publication and final integration onto merged PR1/current dev; no full suite, native app or Meaning/UI work was performed.

Fix round1 against ca573bfc5 addresses I1/I2/I3: ready Keyword pages now expose a frozen generation/policy/source-revision/completion-time snapshot separately from live data_revision. Prior ready snapshots remain queryable during source/policy replacement and failure, with dirty candidates removed before SQL totals/bounds and existing current-source eligibility/final revision checks retained. Claim check+insert is one SQLite immediate transaction; batch writes and promotion require the unexpired owned generation; promotion must affect one row and stale owners cannot delete another ready generation. Incremental/reconcile writes verify their ready owner before cleanup. The unshipped v66 update trigger runs FTS maintenance only when indexed label/title/body actually changes, preserving fresh/v65-upgrade parity without another migration.
Regression evidence: 9 failed/2 passed before fixes, 11 passed after; the first ineligible fixture was corrected to use canonical message deletion after a role-only update proved unsuitable. Four older source-advance ABSENT expectations were changed to retained READY snapshot expectations while keeping stale-content suppression. Final six-file covering gate: 82 passed, 1 inherited RequestsDependencyWarning in24.80s; 70 owners,51 handles drained,0 remaining, process FD delta+13 is still subset-only evidence. Scoped Ruff and format, schema/index/task-ID/CSS/profile/diagnostic guards and complete Task2 whitespace pass. No legacy Python file changed this round, so the previous legacy-diagnostic comparison is not relabeled as fresh evidence. Raw logs: /private/tmp/character-keyword-round1.MECi6U/ (red-valid.log,green-final.log,final-gate.log,static.log and named guard logs).
Compatibility review: CharacterConversationPage appends keyword_snapshot with a None default; existing positional construction and live data_revision fences are preserved. Later Context character_context.py:676 and Roleplay personas_conversations_controller.py:295 discard snapshot metadata, so their owning PRs must propagate it for snapshot-time labeling. No UI edits were included. ADR120 remains authoritative, no new ADR/schema allocation. Status remains In Progress pending independent review; no rebase/push this round.

Final merged-base closeout (2026-09-05): PR1 merged at 4e904f54db74497950eb31594fb37c8cd48568f3. Preserved approved 7dead4898 in codex/task-31242-pre-final-base-7dead4898, then rebased only the six Task2 commits from documentation ancestor304d7cc45 onto that latest dev. All six range-diff entries are equivalent; no conflicts or Task2 runtime overlap. Both Buddy Stop and Task2 index-guard lessons remain. Verified runtime head29a80b9a8861fe7da419b0bb325d4f1056c77f91 retains complete-date-key paging and independently approved I1/I2/I3 corrections (pr-31242-rereview-1.md:3 addressed,0 open Important/Critical,0 new findings).
Actual merged-base nine-file gate:135 passed,1 inherited RequestsDependencyWarning in31.92s. Raw log:/private/tmp/character-keyword-final-base.gZ5RTQ/final-gate.log. Resource subset:76 owners,51 handles drained,0 remaining; process FD delta+13 remains disclosed. All scoped Ruff/format and fatal legacy checks passed; fresh diagnostic code/message multisets equal merged base (588/6/1/1/0/0). Schema110 tables,index275/275/62 pins,Backlog3330 files,CSS,profile and diagnostic inventories and full PR-range whitespace passed; compact logs in the same directory. No source edit during tests, no full suite/native app/dependency/cap changes. Schema66 remains unshipped after upstream65, ADR120 applies with no new ADR.
All Task2 ACs and scoped review/qualification gates are complete, so status is Done under the final-base brief. Earlier In Progress/pending-review paragraphs are historical. PR2 has not been published or merged by this worker. Snapshot-label propagation, cancellation/Data Profile/activation/live-navigation integration remain assigned to owning Tasks3–5; no such UI gate is claimed complete. Controller owns remote publication and subsequent review/merge.
<!-- SECTION:NOTES:END -->
