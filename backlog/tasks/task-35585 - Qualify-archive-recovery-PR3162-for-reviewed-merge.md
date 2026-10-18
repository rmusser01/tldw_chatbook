---
id: TASK-35585
title: Qualify archive recovery PR3162 for reviewed merge
status: In Progress
created_date: 2026-10-18 22:46
dependencies:
- TASK-35584
assignee:
- '@codex'
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3162
documentation:
- backlog/decisions/166-agent-assisted-archive-recovery.md
- backlog/decisions/228-sqlite-store-kernel.md
updated_date: 2026-10-18 23:21
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase archive recovery onto current dev, repair attributable CI and review blockers, and qualify the existing pull request for the repository's guarded merge process.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The branch includes the fetched dev tip without losing archive authority, storage, confirmation, cancellation or Open behavior.
- [x] #2 Fresh independent review and reported PR/CI findings are addressed with targeted behavioral/static evidence and no weakened guards.
- [ ] #3 The reviewed branch and qualification evidence are published with exact leases; review-thread and required-check gates are explicitly satisfied before authorized queue submission.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Existing ADR166/097/228 and design-token/component ADR150/161 apply; this is compatibility, lifecycle and ratchet repair within existing boundaries.
1. Rebase clean tracked state 1f9c62c1d7 onto fetched dev5eb9a37b34a1b0af3e91ad360c0b997497ee7894, preserving original feature branch and exact remote lease; inspect Library_Collections kernel changes and verify ancestry/duplicate definitions.
2. Reproduce and repair the actual CI failures: resolve task33164 ownership by creation/add provenance and renumber its younger owner; preserve Settings concurrency arguments while meeting the existing module budget; re-key archive card bare-type CSS selectors using intended widget classes without raising performance/governance budgets. Regenerate CSS normally.
3. Dispatch one fresh read-only reviewer with exact rebased base/head and the approved archive spec. Review the complete production/test feature diff and connected callers, excluding immutable historical QA payloads. Address every actionable finding and PR comment using meaningful focused RED/GREEN, current source/ref guards and source hashes. No new live requests (30/30 consumed), full suite or guard weakening.
4. Run current Library_Collections/kernel, archive search/restore/capture/mounted controls and actual CI regression checks; record scoped static/generated and independent review evidence. Publish explicit files with an exact lease, update PR qualification and mark ready for review so hosted reviewers can review the final head.
5. Confirm Qodo review on the current head and resolve all actual review threads with evidence. Check MERGE_QUEUE immediately before arming. With queue on, arm auto-merge only after current-head review and all threads resolved, let the queue rebase/run required pull_request CI, never manually merge/update an armed branch, and verify mergedAt/mergeCommit. Address evictions or failures under the same gates; never workflow_dispatch the required check as a workaround.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased cleanly onto fetched dev5eb9a37b34a1b0af3e91ad360c0b997497ee7894 and retained original verified feature history. Fixed attributable CI failures without raising guards: younger task-ID collision with provenance, AST-identical Settings Undo extraction with lowered budget, dedicated native CSS subjects and tightened selector budget, reviewed diagnostic/index inventories, worker-safe Library/Preview navigation and literal notifications. Fresh independent review drove bounded acquisition-close cleanup, consistent workspace timestamp comparison, undated keyset ordering, queued native-source mount/remount fences, and corrected architecture wording. Current dev Collections schema expectation and actual private-profile/UI publication witnesses are aligned; all existing privacy, origin, revision/name, cancellation and exact-confirmation checks remain. Local RED/GREEN and superseded diagnostics, scope limits and source hashes are recorded in Docs/superpowers/qa/console/2026-10-18-archive-recovery-reviewed-merge.md and its companion directory. ADR required: no; existing ADR166/097/228/150/161 govern. Hosted publication/review/required-CI/merge gates are still pending; no merge is claimed and task remains In Progress. No full suite or new live requests; request cap30/30 remains consumed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
