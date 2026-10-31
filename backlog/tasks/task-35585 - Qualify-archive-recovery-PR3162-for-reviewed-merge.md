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
updated_date: 2026-10-31 20:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase archive recovery onto current dev, repair attributable CI and review blockers, and qualify the existing pull request for the repository's guarded merge process.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The branch includes the fetched dev tip without losing archive authority, storage, confirmation, cancellation or Open behavior.
- [ ] #2 Fresh independent review and reported PR/CI findings are addressed with targeted behavioral/static evidence and no weakened guards.
- [ ] #3 The reviewed branch and qualification evidence are published with exact leases, and the authorized queue merge satisfies current-base, resolved-thread and required-CI gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Existing ADR166/097/228 and design-token/component ADR150/161 apply; this is compatibility, lifecycle and ratchet repair within existing boundaries.
1. Rebase clean tracked state 1f9c62c1d7 onto fetched dev5eb9a37b34a1b0af3e91ad360c0b997497ee7894, preserving original feature branch and exact remote lease; inspect Library_Collections kernel changes and verify ancestry/duplicate definitions.
2. Reproduce and repair the actual CI failures: resolve task33164 ownership by creation/add provenance and renumber its younger owner; preserve Settings concurrency arguments while meeting the existing module budget; re-key archive card bare-type CSS selectors using intended widget classes without raising performance/governance budgets. Regenerate CSS normally.
3. Dispatch one fresh read-only reviewer with exact rebased base/head and the approved archive spec. Review the complete production/test feature diff and connected callers, excluding immutable historical QA payloads. Address every actionable finding and PR comment using meaningful focused RED/GREEN, current source/ref guards and source hashes. No new live requests (30/30 consumed), full suite or guard weakening.
4. Run current Library_Collections/kernel, archive search/restore/capture/mounted controls and actual CI regression checks; record scoped static/generated and independent review evidence. Publish explicit files with an exact lease, update PR qualification and mark ready for review so hosted reviewers can review the final head.
5. Use the human-authorized requesting-code-review replacement for Qodo (workspace out of credits) on the current head and resolve all actual review threads with evidence. Check MERGE_QUEUE immediately before arming. With queue on, arm auto-merge only after current-head review and all threads resolved, let the queue rebase/run required pull_request CI, never manually merge/update an armed branch, and verify mergedAt/mergeCommit. Address evictions or failures under the same gates; never workflow_dispatch the required check as a workaround.
6. Repair the hosted startup MountError with a deterministic pre-mount/after-mount regression; verify cold/warm boot census without widening ADR097 budgets and re-review the bounded lifecycle delta before republishing. ADR required: no, existing ADR166/097 and Textual mount ownership apply.
2026-10-31 local integration preparation after replaying all six PR3162 commits onto dev57e330e735 (head52269c3dd4).
ADR required: no
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md; backlog/decisions/228-sqlite-store-kernel.md
Reason: preserve approved archive, SQLite custody and predecessor recovery boundaries during compatibility repair.
1. Inspect final archive chronology/CAS and exact retained recovery catalogs at ChaChaNotes85 and Workspace10, and correct only demonstrated regressions.
2. Run original focused archive tools/search/coordinator/dispatch/card ownership tests and static checks; retain exact logs and limitations in Docs/superpowers/qa/2026-10-31-pr3162-backlog-rebase.md.
3. Review diagnostic statement differences before regenerating the inventory, rebuild CSS from source, and append only the two new archive UI files to the unchanged current census prefix after they pass.
4. Under the explicit 2026-10-31 user instruction to ignore size limits, pin affected size rows to exact final measurements and remove the duplicate controller row; do not contort source for budgets.
5. Commit the local review-ready correction/evidence with task In Progress. Root owns final current-dev rebase, independent requesting-code-review in place of Qodo, publication and merge. No full-suite run or new live requests; original30/30 endpoint cap remains consumed.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased cleanly onto fetched dev5eb9a37b34a1b0af3e91ad360c0b997497ee7894 and retained original verified feature history. Fixed attributable CI failures without raising guards: younger task-ID collision with provenance, AST-identical Settings Undo extraction with lowered budget, dedicated native CSS subjects and tightened selector budget, reviewed diagnostic/index inventories, worker-safe Library/Preview navigation and literal notifications. Fresh independent review drove bounded acquisition-close cleanup, consistent workspace timestamp comparison, undated keyset ordering, queued native-source mount/remount fences, and corrected architecture wording. Current dev Collections schema expectation and actual private-profile/UI publication witnesses are aligned; all existing privacy, origin, revision/name, cancellation and exact-confirmation checks remain. Local RED/GREEN and superseded diagnostics, scope limits and source hashes are recorded in Docs/superpowers/qa/console/2026-10-18-archive-recovery-reviewed-merge.md and its companion directory. ADR required: no; existing ADR166/097/228/150/161 govern. After publication, repaired the real Textual exit/mount/answerability race, serialized first-use projection after captured manual/queued custody, and removed all12 archive first-paint residents. Final223 contract and19 mounted tests pass; final isolated census1030/1031, tightened size guards and derived/static checks pass. Related execution bodies retain verified fidelity; six unchanged size failures and a shrinking inherited transcript failure are explicitly attributed. Three immutable review records and final104-file hashes/validation are in the QA companion directory. User explicitly substituted requesting-code-review for Qodo on this PR. Exact final publication/current-head verification and hosted required-CI/current-base/thread/queue merge gates remain pending; no merge is claimed and task stays In Progress. No full suite or new live requests; request cap30/30 remains consumed.
2026-10-31 integration handoff on dev57e330 / replay head52269c3dd4: preserved current portrait83/region84 and binding-authority9; archive successors are ChaChaNotes85 and Workspace10 with exact predecessor recovery catalogs and the existing Workspace8 automatic-recovery refusal. Native admission/authorizer guards remain intact. Final archive-version/sync retention checks pass.
Local evidence: original feature batch173 passed before4 dispatch wait failures; cards14 passed; final schema/sync/migration/CSS/affected-size guard batch32 passed; final selected ownership/thinking/chronology/manual+received dispatch/structured-captured followup batch21 passed, with the mounted-agent case still failing. A real mounted button journey passed; stock received-send proof failed before archive assertions in its shared10s setup probe. Exact native admission/UI skill-discovery and later lifecycle-writer stack/source controls are retained, but the remaining mounted-agent and received-send failures are not dismissed as baseline-only. Current-head AC2 is unchecked pending these blockers and fresh independent review.
Corrected only test fixture runtime retirement, waiter consumption, bounded state/turn waits and the fixture's explicitly empty skill inventory; no production timeout or permission check was relaxed. Parsed90 changed Python files/49 product files, no duplicate tests and no newly introduced local Ruff findings. Existing parent lint debt is explicitly scoped. Diagnostic inventory verifies unchanged after statement review; CSS rebuild reproduces. Under the explicit2026-10-31 size waiver,12 affected rows are exact and the duplicate controller row is removed. Census appends only the verified cards file (1173-entry prefix unchanged; floor1174); the planned flow addition is deferred. QA/commands/JUnit/source hashes: Docs/superpowers/qa/2026-10-31-pr3162-backlog-rebase.md and companion directory.
ADR required: no; existing ADR166/228 apply. Task remains In Progress. Root owns final current-dev rebase, fresh requesting-code-review replacing Qodo, publication and merge under the user's CI waiver. No full suite, live endpoint request or remote action; original30/30 request cap remains consumed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
