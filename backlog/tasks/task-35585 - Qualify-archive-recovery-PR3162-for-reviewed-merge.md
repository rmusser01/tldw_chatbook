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
updated_date: 2026-10-31 21:02
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
2026-10-31 final current-dev integration onto 0d05d27fa1d569f1edebbc2346d43f8c03da9911.
ADR required: no
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md; backlog/decisions/228-sqlite-store-kernel.md
Reason: compatibility rebase preserving the existing archive and native custody contracts.
1. Replay the clean local head3f9676738c onto the fixed current dev tip; preserve landed Plugins, OS notifications, approval-profile authority, Kanban F9 and schema85/10.
2. Resolve conflicts, inspect range-diff, reconcile canonical CSS and reviewed diagnostic entries, retain the current1174 census prefix and append only the verified archive cards file (1175).
3. Measure only affected size rows under the dated user waiver; run narrowly scoped conflict/static checks without repeating known mounted failures or broader tests.
4. Record remaining mounted-agent and stock received-send qualification blockers, commit a clean local handoff, and leave task In Progress. Root owns fresh independent review and remote actions; no full suite or live calls.
2026-10-31 independent-review follow-up on608ad5418a.
ADR required: no
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md
Reason: restore the approved bounded native-call deduplication and pinned confirmation contracts.
1. Add focused RED controls for pending/completed retries after result-page eviction, changed arguments and bounded receipt eviction through ArchiveRecoveryRuntime.
2. Remove only the redundant outer result resolution; retain the coordinator new-agent evidence check and bounded history.
3. Add a minimal mounted busy/expired receipt control proving the displayed request, manual scope, visibility and focus survive unrelated settlement; clear controls only for their owning operation.
4. Run the smallest affected GREEN controls, record exact source hashes and receipts alongside all prior failures, and commit a clean local handoff for fresh independent review. No remote, full-suite or live actions.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased cleanly onto fetched dev5eb9a37b34a1b0af3e91ad360c0b997497ee7894 and retained original verified feature history. Fixed attributable CI failures without raising guards: younger task-ID collision with provenance, AST-identical Settings Undo extraction with lowered budget, dedicated native CSS subjects and tightened selector budget, reviewed diagnostic/index inventories, worker-safe Library/Preview navigation and literal notifications. Fresh independent review drove bounded acquisition-close cleanup, consistent workspace timestamp comparison, undated keyset ordering, queued native-source mount/remount fences, and corrected architecture wording. Current dev Collections schema expectation and actual private-profile/UI publication witnesses are aligned; all existing privacy, origin, revision/name, cancellation and exact-confirmation checks remain. Local RED/GREEN and superseded diagnostics, scope limits and source hashes are recorded in Docs/superpowers/qa/console/2026-10-18-archive-recovery-reviewed-merge.md and its companion directory. ADR required: no; existing ADR166/097/228/150/161 govern. After publication, repaired the real Textual exit/mount/answerability race, serialized first-use projection after captured manual/queued custody, and removed all12 archive first-paint residents. Final223 contract and19 mounted tests pass; final isolated census1030/1031, tightened size guards and derived/static checks pass. Related execution bodies retain verified fidelity; six unchanged size failures and a shrinking inherited transcript failure are explicitly attributed. Three immutable review records and final104-file hashes/validation are in the QA companion directory. User explicitly substituted requesting-code-review for Qodo on this PR. Exact final publication/current-head verification and hosted required-CI/current-base/thread/queue merge gates remain pending; no merge is claimed and task stays In Progress. No full suite or new live requests; request cap30/30 remains consumed.
2026-10-31 integration handoff on dev57e330 / replay head52269c3dd4: preserved current portrait83/region84 and binding-authority9; archive successors are ChaChaNotes85 and Workspace10 with exact predecessor recovery catalogs and the existing Workspace8 automatic-recovery refusal. Native admission/authorizer guards remain intact. Final archive-version/sync retention checks pass.
Local evidence: original feature batch173 passed before4 dispatch wait failures; cards14 passed; final schema/sync/migration/CSS/affected-size guard batch32 passed; final selected ownership/thinking/chronology/manual+received dispatch/structured-captured followup batch21 passed, with the mounted-agent case still failing. A real mounted button journey passed; stock received-send proof failed before archive assertions in its shared10s setup probe. Exact native admission/UI skill-discovery and later lifecycle-writer stack/source controls are retained, but the remaining mounted-agent and received-send failures are not dismissed as baseline-only. Current-head AC2 is unchecked pending these blockers and fresh independent review.
Corrected only test fixture runtime retirement, waiter consumption, bounded state/turn waits and the fixture's explicitly empty skill inventory; no production timeout or permission check was relaxed. Parsed90 changed Python files/49 product files, no duplicate tests and no newly introduced local Ruff findings. Existing parent lint debt is explicitly scoped. Diagnostic inventory verifies unchanged after statement review; CSS rebuild reproduces. Under the explicit2026-10-31 size waiver,12 affected rows are exact and the duplicate controller row is removed. Census appends only the verified cards file (1173-entry prefix unchanged; floor1174); the planned flow addition is deferred. QA/commands/JUnit/source hashes: Docs/superpowers/qa/2026-10-31-pr3162-backlog-rebase.md and companion directory.
ADR required: no; existing ADR166/228 apply. Task remains In Progress. Root owns final current-dev rebase, fresh requesting-code-review replacing Qodo, publication and merge under the user's CI waiver. No full suite, live endpoint request or remote action; original30/30 request cap remains consumed.
2026-10-31 fixed-current-dev replay: rebased all8 local commits from3f9676738c plus the integration plan onto0d05d27fa1d569f1edebbc2346d43f8c03da9911. Preserved schema85/10/native admission, captured Pressed inputs, Plugins/MCP owner checks, all landed notification and inherited approval-profile authority, and Kanban F9. Added archive proof after the current async configuration/owner fences and drained WAL setup statements under the existing deadline. Existing selection helpers remain in their current module. Canonical CSS rebuilt and current1174 census prefix preserved with only verified cards appended (1175);12 affected size rows are exact under the dated user waiver.
Fresh focused evidence: both dedicated native normal/queued archive-origin regressions pass and compare every non-proof configuration field against the actual producer;3 sync-capture/WAL controls pass; both size guards pass for12 affected rows;90 changed Python files parse, no duplicate test names, conflict F821/F811 clean. Targeted scanner of48 changed product files exactly matches the committed current-base diagnostic inventory after reviewing its sole statement delta.
The single current-base mounted archive representative still fails before provider entry:60s stack shows activation/config reads waiting on storage initialization, with qualification evidence and AgentRuns private-artifact workers active. Cleanup completes. Original mounted/received failures remain open. Two pre-existing eager-MCP wired assertions also fail; setup-only probe proves unchanged valid source/profile/alias ownership, and current stock capture defers persisted definitions while temporary sessions can stay empty. Original assertions remain untouched; the dedicated proof tests avoid that superseded eager-read assumption. A single authorized plain-turn control using the same fixture/cleanup is running read-only to distinguish shared initialization cost, with no timeout padding. Receipts and source provenance: Docs/superpowers/qa/2026-10-31-pr3162-backlog-rebase/current-dev/.
Task remains In Progress with AC2 unchecked. Root owns full independent review and remote steps; no broad sweep, live request or remote action.
The one authorized non-archive native-turn control has now finished:1 failed87.93s total/78.10s call under the same60s state bound, with full cleanup and zero provider/tool calls. Its activation worker was inside bootstrap._registry→native Windows stat/open_handle while another database worker waited at storage_admission.initializing. It reproduces the pre-provider failure without archive tool execution, distinguishing shared startup/admission work from an archive-result operation; it does not qualify the archive journey or establish platform-wide cause. Exact source, command, JUnit and trace are in current-dev/nonarchive-control*; no retry or received-send rerun followed. Dedicated normal/queued exact archive-proof tests pass2/2, retaining every other field observed from the actual configuration producer. Clean handoff for independent review; no further tests running.
2026-10-31 independent-review repairs on608ad5418a (fixed current-dev base0d05).
Removed the runtime pre-resolution that bypassed the coordinator native-call deduplication order. Identical pending/completed calls retain their original outcome after FIFO page eviction or TTL expiry; changed arguments still refuse, new agent calls still cannot select manual-only evidence, and receipt eviction leaves old calls/consumed decisions expired. No cache or retention limit changed.
Receipt projection now clears controls and redirects expired focus only when the displayed confirmation belongs to the settled operation. Minimal mounted controls prove foreign busy/expired receipts preserve exact request identity, manually chosen scope, visibility and focus across sync; the own-expired positive control still clears/moves focus.
RED: both runtime eviction variants failed at the redundant lookup; after correcting only the test scope-event setup, both foreign receipt variants failed and own-expired passed. GREEN:10 targeted cases passed in25.67s, including existing deduplication/receipt-eviction/one-shot/stale-event controls. Four changed code files parse with no duplicate tests; no introduced F821/F811 findings versus608ad (15 existing imported-fixture warnings remain). A final fixture-specific noqa comment is AST-identical to the passing test source; both tested and final source hashes are retained.
Evidence: Docs/superpowers/qa/2026-10-31-pr3162-backlog-rebase/review-followup/. Prior failures remain recorded. Parent reports fresh independent review accepted the shared pre-provider and stale eager-MCP classification as bounded baseline evidence; these repairs do not turn those failed journeys into passes. ADR required:no; existingADR166 applies. Task remains In Progress pending review/publication. No broad retry, full-suite, live or remote action.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
