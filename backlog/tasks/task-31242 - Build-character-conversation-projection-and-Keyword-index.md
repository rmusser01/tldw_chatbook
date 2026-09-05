---
id: TASK-31242
title: Build character conversation projection and Keyword index
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 02:05'
updated_date: '2026-09-05 22:31'
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
6. External Qodo round1: reproduce bounded repair paging, strict unavailable query validation and card-insert invalidation with real SQLite RED tests; relocate unshipped v66 DDL to a packaged versioned artifact, document public contracts and independently verify retained snapshot fencing. Run focused GREEN then the nine-file covering gate, migration packaging and scoped static/guards. ADR120 applies; no new ADR. Keep In Progress pending independent review; no rebase/push.
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

PR2434 external Qodo round1 (FIX_BASE cb68308cf): reopened In Progress pending independent review. Moved unshipped v66 DDL to chachanotes_v65_to_v66_character_conversation_search.sql and existing file-backed dispatcher/package/guard discovery. Added capped repair_candidates(key, *, offset=0, limit=20) returning CharacterRepairPage(candidates,total,next_offset), limit1..50, deterministic SQLite NOCASE name then ID; callers must follow continuation and restart after mutations. Unavailable metadata query rejects nontext or >200 characters before stripping; empty date browsing preserved. Documented public navigation/repository contracts. Card INSERT now advances revision and enqueues exact-ID affected chats under activated ownership, preventing an incomplete in-flight READY promotion. Wrong-authority and name-only cases remain excluded. Retained-snapshot revision fence independently verified and unchanged.
TDD raw logs: /private/tmp/character-keyword-qodo1.3RcKio/. Initial RED7 query cases and missing artifact; corrected fixtures reproduce unbounded tuple and both card-insert defects (three failures). Focused GREEN artifact8,paging/query12,insert2. Covering nine-file gate plus4 real wheel/sdist migration checks:151 passed,1 inherited warning in39.67s, final-gate.log. Owners88,handles drained63,remaining0; process FD delta+13 remains subset-only. Scoped Ruff/format and fatal legacy rules pass; legacy code/message Counters equal FIX_BASE588/6/1/1/0/0. Schema110,index275/275/62pins,task3330 and whitespace pass. ADR120 applies, no new ADR/schema version, no UI/native/Meaning/dependency changes or full suite. Task3 Library/Context callers must adopt explicit pages; no silent first-page slicing. Prior Done paragraphs are historical; controller gates independent review before completion and owns remote writes/rebase.

PR2434 final approved refresh: scoped review pr-2434-qodo-rereview-1.md resolved all6 findings with0new/open. Preserved927b26df82cfbc9a8965bd072ce2c0504ddd2ca4 as codex/task-31242-pre-pr2434-final-927b26df8, fetched actualdev53194eee674865bd8b4aa6daac4b1e7d97160594 (schema65), then rebased8Task2 commits from4e904f54. All8range-diff entries equal; no conflicts. Upstream Console trace changes and lesson retained. Tested runtime tip a663ba73cec17337c8737de95bc1295122b248f1.
Fresh exact151-test nine-file plus wheel/sdist migration gate:151passed,1inherited warning in50.17s, raw /private/tmp/character-keyword-pr2434-final.5d7h9u/final-gate.log. Resource88owners/63handles drained/0remaining, FDdelta+13 remains subset-only. Scoped Ruff/format/fatallegacy and fresh531-base diagnostic Counters588/6/1/1/0/0 equal. Schema110,index275/275/62pins,task3332,CSS,profile and persistentdiagnostic guards all pass; raw static.log,legacy-diagnostics.log,guards.log and rebase/range-diff logs in same directory. Full PR whitespace pass. No runtime edits after tests. All ACs and independent review/fresh local qualification complete; markingDone under final rebase brief. Earlier InProgress paragraphs historical. ADR120/unshipped66 remain; Task3 repair-page continuation and snapshot-label obligations, later UI/cancellation/DataProfile/live gates remain separately owned. No push/merge; controller owns remote review/CI and strictbase integration.
<!-- SECTION:NOTES:END -->
