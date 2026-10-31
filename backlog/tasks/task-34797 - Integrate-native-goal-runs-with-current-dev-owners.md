---
id: TASK-34797
title: Integrate native goal runs with current dev owners
status: Done
assignee:
  - '@codex'
created_date: '2026-10-11 22:38'
updated_date: '2026-10-12 02:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the accepted native-goal first release safe on current storage, runtime, accounting, filesystem and UI owners before protected merge of PR2563.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Populated current AgentRuns data upgrades additively and backup/recovery admits exact supported catalogs.
- [x] #2 Progress wakes, chat starts, allowance roots and authority remain valid with goal iterations.
- [x] #3 An unstarted paused goal resumes safely; accepted uncertain work cannot replay.
- [x] #4 Mounted controls, isolated CLI failure/correction/success and restart evidence demonstrate the integrated behavior.
- [x] #5 Independent review and targeted static/governance checks have no unresolved blocking findings.
- [x] #6 Notes status publications from tasks lacking Textual app context debounce safely, and shutdown publications cannot restart a timer.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: current-dev integration amendment to the accepted native-goal decision.
Follow Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md; preserve historical source evidence and qualify actual current owners.

Combined review fix wave from 8bf7c02e51: add real-SQLite regressions for unresolved root descendants and late uncertainty before quality review, then share canonical settlement guarding across checkpoint/removal/review. Add fresh-versus-borrowed finite/native connection retirement and maintenance-drain controls, then reuse existing operation_owned_connection/run_owned_db_call at Goal callback boundaries without changing foreign or in-memory owners. Compare inherited hook-preflight before scope expansion. Correct historical plan whitespace, rerun targeted native/mounted/restart and static qualification at exact fixed head, and append the task-1 fix report. ADR227 remains governing; TASK34797 AC5 stays open for root scoped re-review.

Publication-contract closeout: inspect intended metadata diagnostic moves before regenerating the existing inventory; fix actual post-await Goal setup/status widget lookups with held-await dismissal regressions. Run the required preflight and affected widget lane without changing authority or caps; keep AC5 open until scoped review and guards pass. ADR required: no new ADR; existing227/097/150 apply.

Hosted Perf Guard closeout: reproduce282/274 bare-type CSS and559/558 preimport costs; class-key only Goal subjects and defer settings_goal_policy at actual call sites without duplicating inventories or raising ratchets. Follow Task3 in the existing integration plan. ADR required:no new ADR; existing097/150/227. AC5 reopened until scoped review and fresh guards.

Task4 required-UI prerequisite: reproduce hosted active_app LookupError deterministically with the existing Notes status listener on native Textual in an empty Context, with cross-thread positive control; compare untouched dev. If confirmed, establish the existing app context only while creating its timer and fence shutdown publication. Preserve native debounce/refresh ownership and add targeted watcher/recovery/delete/teardown checks. ADR required:no new ADR; routine callback-context correction under existing lifecycle.

Task4 review fix: gate the green nine-case native/pure Notes listener module by appending the required UI census and tightening its entry floor179->180; preserve all prior entries/shards. Task5 fast-lane migration fixture: reproduce2 failures caused by v24 Goal trigger/table residue in synthesized v15; remove only absent-at-v15 Goal shape before later dependencies are stripped, assert clean predecessor shape, and test app/standalone upgrade twice preserving upstream records/authority. Production migration and triggers remain unchanged. ADR required:no new ADR;227 applies. AC1 reopened before fixture changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current-dev integration (2026-10-11): implemented on a3c47c6c72 through complete code head 8bf7c02e51, with live ADR backlog/decisions/227-native-console-goal-runs.md. Schema 24 and exact recovery preserve current v23 owners; native Goal submission uses runtime custody, shared finite allowance roots and current admitted filesystem execution. Unstarted Pause/Resume now safely dispatches; uncertain accepted work remains fenced. Source history above remains historical.

Current qualification: 36 mounted Goal tests; 81 startup/hydration/review/wrap-up checks; 58 corrected native recovery/scheduling/source cases; 46 authority/storage/evidence checks; 75 existing budget/hooks checks; 16 capacity/real executor checks; current custody/profile rerun 6 passed. Final complete-head receipt and source hashes: pr2563-complete-head-receipt.json and pr2563-complete-head-verification.txt in /Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-wave2. The 59-test storage phase receipt was integrated working-tree evidence, not standalone phase-commit evidence.

New Python files pass Ruff and format; modified shared files introduce no Ruff findings over the pinned base. Screen ratchet passes; pre-existing controller/store excess shrinks and unrelated inherited governance failures are recorded without raising caps. Actual production CSS mounted setup/review was inspected at 80x24 and 160x44. External model trial remains explicitly unqualified/skipped; no full test sweep or cloud spend. Original source F9 location is historical; current canonical Settings is F4. Full integration independent review/latest-base publication remains tracked separately by TASK34797 AC5. See Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md and its task-1 report for exact evidence and limitations.

Resource qualification reopened: the complete-head 62-case receipt passed but reported FD growth 785; earlier mounted/native lanes also warned. Exact database/worker-owner diagnosis is pending while source stays frozen at 8bf7c02e51 for independent review. Do not interpret the preceding closeout as a completed resource-lifetime qualification. Task1 report records this DONE_WITH_CONCERNS status.

Combined independent-review fix 790d456f21: canonical allowance-root descendants and unknown accounting now block checkpoint settlement and late result approval; native Goal service/coordinator/UI/provider/tool callbacks retire only their newly acquired handles through current operation-owned boundaries. Fresh/borrowed real CLI and mounted paths, cancellation until physical completion, in-memory/foreign controls and maintenance drains pass. Supporting targeted lanes: 80 Goal runtime, 96 current budget/hooks/SQLite/maintenance, 36 mounted UI; no resource warnings. The exact post-finalizer fixed probe has zero AgentRuns/Chat/Workspace connections for both actual native paths. Original warning receipts and the explicitly labeled repeated baseline remain historical; fixed receipts use distinct names. ADR227 governs; source IDs/notes retained. New exact committed-head qualification is recorded in pr2563-fixed-head-receipt.json and pr2563-fixed-head-verification.txt, with full commands/limitations in the task-1 report. Status remains In Progress pending root scoped independent re-review; TASK34797 AC5 remains unchecked. No full sweep or live external-model trial.

Current integration closeout: independent complete-head and P1 fix review passed; publication-contract Task2 and its one scoped fix re-review passed with zero blockers. Latest dev7b7193d647 rebase preserved canonical native continuation ownership;25 affected continuation/startup tests passed, with unchanged1033/1033 startup-headroom warning. Actual current preflight passed all derived-artifact guards. Earlier79 byte-bound goal cases,80 runtime/96 owner/36 mounted lanes,37 affected UI and final14 setup cases retain distinct receipts; counts overlap. Goal worker cleanup has zero retained owned DB handles. Current F4, schema24/exact recovery and ADR227 govern; historical source failures and baseline size/style debt stay labelled. No full suite, external model quality or physical terminal UAT was claimed. Root publication/required hosted checks/merge remain separate gates.

Hosted-failure follow-up Tasks3-5: class-scoped Goal CSS and first-use Settings helper pass unchanged274/558 ratchets; ordered atomic save/revert preserved. Proven pre-existing Notes same-thread timer context and shutdown-admission defect corrected with native empty-context/cross-thread controls; nine-case module enters required UI census with all179 existing entries/shards retained and floor tightened180. Corrected only synthetic-v15 fixture Goal shape before later dependencies are stripped; production schema/migration/immutability unchanged. Final targeted receipts: navigation/listener14passed70.80s; final orchestration/Goal migration/census74passed8.04s; other final affected lane21passed with one Changes-test race superseded by final14. Initial52pass/1history-race and all red receipts retained. Independent spec/quality review passed with zero unresolved findings; census gap closed. Changed widgets/listener/tests Ruff/format/diff pass; shared Settings119 and checker2 Ruff findings remain identical inherited debt. Actual preflight passed; later census change has67 targeted CI contracts and180/180 checker proof. Dev c23e0ba141 adds only three nonoverlapping task-note files, so reviewed/tested product bytes are preserved during rebase. No full sweep, physical terminal, external-model or hosted merge clearance claim; zero preimport module headroom remains explicit. Review/evidence are in external pr-review-wave2 directory; root owns fresh hosted CI and protected merge.
<!-- SECTION:NOTES:END -->
