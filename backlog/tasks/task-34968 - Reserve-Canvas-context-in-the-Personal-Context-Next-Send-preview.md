---
id: TASK-34968
title: Reserve Canvas context in the Personal Context Next Send preview
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-11 23:58'
updated_date: '2026-10-12 15:15'
labels:
  - memory
  - canvas
  - context-budget
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Canvas tools are enabled and the model input budget is tight, the Personal Context Inspector can report records as selected even though the dispatched request omits them. Make the preview reflect the complete request budget while preserving the disposable nature of inspection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A native production preview regression reproduces the Canvas-enabled budget mismatch before the fix.
- [ ] #2 With identical profile, model, message and tool inputs, the Inspector and dispatched request agree on selected records and their available input budget, including Canvas schemas and runtime guidance.
- [ ] #3 Preview does not register an executable Canvas run, create staging records, grant mutation authority, or change the selected Canvas.
- [ ] #4 Disabled, unavailable and stale Canvas states preserve their existing fail-closed behavior, with native Python >=3.12 targeted verification and a recorded ADR check.
- [ ] #5 Current Goal/Canvas preview budget parity and post-await Canvas ownership regressions execute in required PR checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md; backlog/decisions/202-dependency-aware-personal-context-forgetting.md
Reason: bounded correction of disposable first-request budgeting under existing Canvas ownership and Personal Context publication boundaries; document the pure-schema preview contract in ADR-121 before implementation.
1. Reproduce missing Canvas reservation through the real controller/bridge preview and real Personal Context selection, with synthetic private state and native Python 3.12.
2. Reuse the Canvas schema loader and shared disclosure planner for a data-only preview catalog. Preserve live registry authentication, schema order, persona narrowing, native/fenced protocol and progressive discovery; do not register a preview run or issue Canvas authority.
3. Reuse capture_interactive_owner/validate_interactive_owner and exact session identity to fence late publication against close, disable, durable/temporary replacement, promotion and branch changes. Carry the guard through project-preview assembly and Inspector async validation; release diagnostic sidecars only at final snapshot publication. Keep schema loading incremental inside the shared guarded probe, including denied and unreadable schemas.
4. Verify production preview versus dispatched first-request budgets/records and absent run/staging/selection effects, plus disabled/unavailable/stale controls. Run only affected native tests and scoped static checks.
5. Obtain one fresh final review, update receipts/task notes and the existing PR against dev. Record current-dev qualification debt and any newly advanced base separately.

Current-dev reconciliation (2026-10-11): repeat production preview/dispatch, current Goal narrowing, Canvas authority and private sidecar boundary qualification on the final integrated source. Preserve current budget/permission/custody owners and all original limits. Historical source evidence below is not a current-dev qualification claim.

Complete-head review correction: follow Task2 in Docs/superpowers/plans/2026-10-11-pr2862-current-dev-integration.md. ADR required: no new ADR. Existing191/102/150/161 and required-regression governance apply. Reject pending-first retired provenance with a cheap host identity fence, preserve off-thread bootstrap, use literal Text.assemble, gate bounded critical privacy/native/Canvas/currentness/original-retirement regressions, and correct living architecture inventory before independent re-review and actual sequential-base qualification. Existing source/receipt history remains sealed; native observation entries are not unique-owner counts.

Task4: reconcile independently cleared feature-only range onto actual merged Goal9595facbbb; preserve atomic/lazy Goal Settings with cached PC identity, source-built CSS and exact current required targets/census187; qualify only changed joint native/UI/Canvas/import/CI boundaries. Actual post-Focus/latest-dev, independent clearance and protected CI remain required before publication/completion. ADR required: no new ADR; direct existing Personal Context/Canvas/Goal227 integration.

Task5: preserve actualdev352 concurrentNotes-sync inode/containment fix and executable fast sandbox containment target while rebasing reviewed PC feature-only range; qualify only affected CI/nativeNotes/dependency controls and exact sourcebindings. Existing ADRs; no new ADR/authority/storage/permission/cap. FinalactualpostFocus/latestdev andprotectedhostedCI remain rootgates.

Task6: preserve accepteddf4985 Buddy/Library/Settings owners while retaining all reviewed PC contracts. ADR required:no new; direct existing boundary preservation. Full Task6plan in Docs/superpowers/plans/2026-10-11-pr2862-current-dev-integration.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Historical source implementation evidence (source49c5ebf4eec858df40b56c65e46b5d94414ccb7a; current integration remains In Progress):

Corrected disposable Personal Context budgeting using the existing Canvas owner profile, ordered catalog metadata and a lazy read-only schema loader inside the shared guarded planner. Live Canvas provider authentication and execution authority remain unchanged. One shared currentness callback fences exact session identity, Canvas enablement/controller/interactive owner and branch through final snapshot publication and later Inspector validation; diagnostic sidecars publish only after that fence.

Native Python 3.12.11 reproduced the original 123565 versus 121793 token mismatch on unchanged head 295ee2a02a. Seven fresh-review regressions failed before correction (unreadable/denied schemas, closure/disable/replacement during project preview, durable replacement and later Inspector validation). Final affected groups passed 93 context/Inspector cases and 367 agent/Canvas cases, followed by 19 shared-probe checks after simplifying the loop. All 41 exact private-profile children passed without skips; parent/child application and profile-library imports were verified against this PR worktree. Three legacy suites now use the existing bootstrap-profile marker; the unchanged control reproduced their config-source setup failure. Targeted lint, regression-file formatting, AST-preserving helper formatting and whitespace pass. ADR-121 and ADR-202 apply; no new policy or activation was selected. Current-dev integration qualification remains separately tracked.

2026-10-11 current-dev implementation on323c3b0c: passive Canvas catalog metadata/schema loader now participates in the shared first-request planner before current Goal narrowing. A production Goal+Canvas regression reproduced123844 vs122349 token reservation and passed after reconciliation, preserving actual authenticated live dispatch and data-only preview. Exact session/Canvas/service/selection/expiry guards remain; no run/staging/permission/selection mutation occurs in preview. Current Inspector/Settings integration tests and native-owner qualification are recorded separately from historical source receipts in Task1 report. ADR121/202 apply; no new ADR or activation authority. Current integration remains In Progress pending complete-head independent root review and latest-dev transplant.

2026-10-11 final Canvas handoff: passive production Goal+Canvas first-request planner regression and59-case helper/selection/snapshot group pass; affected Canvas fixture and feature cases have final joined410-node passing coverage with separate native retirement receipts. Preview authority/currentness remains data-only and native V2/schema8/AAD1/Forget/source/disclosure/server activation gates remain closed. Commits c0cb9f58ec and fa402213ac8aa78e99f64e1724bae46d59684b51; exact commands/source hash/node mapping in Task1 report and pr2862-final-integration receipt bundle. Existing ADR121/202 apply. Task remains In Progress until independent root review and actual post-Goal sequential-dev transplant; inherited guard/Goal deadline failures remain separately reported.

2026-10-11 Task2 correction gates named production Goal/Canvas/native-tight/fenced-tight/unreadable/denied planner and both publication-drift controls in the existing required sandbox lane, plus selection factories and lazy worker cancellation in UI census. Complete bounded minimal-dependency selection826 sandbox (62 feature additions),13 admission and46 UI cases passed with unchanged deadlines/caps. All original179 UI entries retained; seven additions floor186, preserve root post-Goal180+7 at transplant. First-read canonical provenance owner fence, literal export label and living architecture corrections are in Task2 report/new pr2862-review-fixes receipt seal. V1/schema8/AAD1 and all closed activation gates remain. In Progress pending independent root scoped review, actual current-dev transplant and hosted clearance; no publication or merge by implementer.

Task3 independent spec PASS and quality PASS at ea50008b3c, with38 affected CI/copy-probe controls passing and all prior/runtime seals unchanged. Task4 integration qualification begins against actual merged Goal; no Task Done or hosted clearance is claimed.

2026-10-11 Task4 current-Goal reconciliation: feature-only range transplanted onto exact merged9595facbbb, preserving current request/Goal/tool/root budgets and pure Canvas preview/currentness owners. Canvas/native boundary group54 pass/1 already retained Goal outer_timeout failure; that exact body also fails on isolated9595 source with unchanged two-second dispatch wait. Settings61, current Goal style/navigation2, native original-owner/Notes admission22 and executable CI/census105 pass; required target order/flags/deadlines retained, current180+7 census187, all12 offline preflight guards pass. Original seals untouched; new receipts/report are separate under pr2862-task4-current-goal-integration and Task4 SDD. No new cap, permission/schema/AAD/V1/V2/Forget/source/disclosure/server activation or GitHub mutation. Existing three component and broad lint/format failures retain exactbase evidence. Status/AC remain In Progress/open pending root independent clearance and actual post-Focus/latest-dev/hosted qualification. No new ADR; direct existing Canvas/PC/Goal227 integration.

Task4 final independent specPASS/qualityPASS at31eb with all227feature paths/current41base owners/source/evidence seals verified; zero new findings, inherited4RED remain documented. Actualdev advanced withPR3077 nativeNotes/requiredCI changes, prompting boundedTask5 beforepublications. No TaskDone claim.

2026-10-11 Task5: rebased only9595..e0d7b600 feature/governance onto exact concurrent Notes-fix352496f5, qualified runtime57745fb68416cf8b206ea300814a5973187eee31. All227 feature paths retained; all11 nonoverlapping Notes-fix paths match352; only append-only testing lesson conflicted, both bodies retained. Settings/CSS/PC/Goal/Canvas/packages/runtime feature bytes remain independently cleared31eb, except identified actual-base Notes changes. Actual containment target belongs fast sandbox: preserve38base+26PC targets,22base+originalcleanup admission module (13 cases), exact180+7 census187, four comment-aware parser function ASTs, flags/deadlines/caps. Fresh CI105, Notes context9, containment36pass/1APFSskip and complete original settlement15pass/2APFSskip; all12offlinepreflight passed with canonical cache. Exact352 settlement control15pass/2APFSskip establishes existing native retirement debt: candidate20dead-worker leases/86physicalFDs versus base21/88 over17Notes DB paths; no cleanup, GC or borrower repair. Initial copied observer17teardown errors remain diagnostic, not clean base evidence. Current CI Ruff check/format pass; prior4RED/121lint debt retained through unchanged sourcebindings. All32009 qualification-source hashes unchanged and1100immutable receipt files retain their seals; root live sequence-state is explicitly outside immutable history. New receipts/report separate under pr2862-task5-current-notes-integration and Task5 SDD. ADR required:no newADR; existing Notes/Console/PC/Canvas/Goal/required-regression decisions apply. Status/AC remain In Progress/open pending root independent/latestdev/postFocus/hosted clearance; no publication/merge/TaskDone claim.

Task5 independently cleared specPASS/qualityPASS onb750, no newsourcefinding. Inherited native Notes settlement debt reproduces exact352 and staysRED. Latestdevdf4985 preservation/boundedchanged-owner qualification nowTask6; no latestbase/hosted/Done claim.

Task6 private latest-owner transplant (2026-10-12): exact actual dev df4985eca6c3e7e09bd1eca60536411c6faa3720; runtime-qualified head 7142222f2df59f8e666e9b2ac155b53b471de886. All14 feature/governance patches range-diff identical,227 feature paths retained, all51 accepted Buddy/Library/Settings changes preserved; only Settings discovery method and append-only lessons overlap. Fresh bounded candidate78pass/1 inherited pre-body config-source refusal across79 selected cases; exactdf targeted controls15pass/1same refusal. NativeNotes9 zero process leases. Changed Library behavior15pass and factory8 original owners retire, while Notes12 dead-thread leases/34physicalFDs match exactdf on same4relative paths. Settings32/normalization16pass; six boot/import controls pass original caps (payload552/558,417200/425347LOC;boot678/686;UIready1031/1033). Fulloffline12preflight pass112.763s with verified Mermaidcache. Changed Settings Ruff119 per-file findings and formatRED match reviewed and actualbase; existing static/size/deadline/17-node Notes debt retained. One invalid native-observer path setup reached no tests and remains separately recorded. At the Task6 final audit, 1,200 of 1,201 inherited frozen files remain unchanged; the root-owned live sequence-state.json change is recorded separately. The original manifest is retained, immutable PC receipts remain unchanged, and original Task5 live-root chronology remains intact. ADR required:no new; existing accepted ownership/memory/custody decisions. No runtime repair, GC, cap change, full suite or hosted claim. Task6report and exact argv/raw/XML/source/native/seal evidence in pr2862-task6-current-owners-integration. Root independent review, actual post-Focus/latestbase, hosted qualification/publication/protected merge remain; status and open AC retained.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

The source Personal Context Canvas task used TASK-33165, which collides with the older Dreams query-angle task retained on dev. This new CLI-allocated identity preserves all four source acceptance criteria, the original plan and clearly historical implementation notes. Its criteria are reopened until the current integration is qualified. Source `49c5ebf4eec858df40b56c65e46b5d94414ccb7a` and the frozen source branch preserve the original completed record; live Personal Context references use this task's new identity.

## Renumbering provenance

Renumbered from task-34800 to TASK-34968 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
