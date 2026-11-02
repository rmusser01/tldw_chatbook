---
id: TASK-38544
title: Qualify the integrated prompt mix-in lifecycle and document delivery
status: In Progress
created_date: 2026-11-01 19:30
dependencies:
- TASK-38541
- TASK-38543
labels:
- prompts
- console
references:
- backlog/decisions/286-console-prompt-mix-ins.md
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
assignee:
- '@codex'
updated_date: 2026-11-02 04:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish that the complete mix-in workflow behaves as approved across real Console surfaces, storage lifetimes and supported provider boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Targeted integration journeys exercise the real source/store/controller/gateway and UI for edit, queue, off, retry-original, regenerate-current, reopen, temporary promotion, copy/fork and failure recovery, including exchange capture off.
- [x] #2 Isolated live UI evidence covers the existing Prompts action at normal and narrow terminal sizes with keyboard input; real provider/cache evidence is recorded when available and missing qualification is explicit.
- [x] #3 Relevant architecture/user documentation, migration packaging, scoped lint/format and UI token/bundle checks are current; no full suite, new rendering system, automatic follow-up action or unrelated refactor is introduced.
- [x] #4 Copy/fork after a same-row Retry chain retains exact empty or nonempty receipts, markers and lineage after source purge; live temporary and durable fork publication remains replayable and promotion preserves ownership with empty selections and cleared proof authority.
- [x] #5 Fresh current-choice Regenerate, edit-and-resend and Continue refuse source replacement during preparation without dispatch or draft loss; original Retry and accepted queue snapshots remain independent of current source availability.
- [x] #6 Opening unsupported server Mix-ins from ordinary Browse preserves its source/view and reports bounded capability feedback; expected unavailable-source Preview reports recovery guidance without an internal-error traceback.
- [x] #7 Archive privacy sentinel fixtures assert the intended ID and pass all five nodes; restored-generation assertions prove original scalar identity/count; delivery status and historical versus current QA provenance agree.
- [x] #8 The feature-only branch is based on current dev, preserves schemas 82–85, installs mix-ins at schema 86, and passes targeted native, migration/recovery, provider and UI checks with honest historical/current evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Qualify integrated capture-off journeys through the local source, store, controller, provider gateway and operational receipt anchors; audit selected-generation display projection.
2. Run affected focused modules and packaging/restore checks, preserving original-baseline parity evidence and scoped lint/format deltas.
3. Exercise the actual isolated app with production styles at the three required sizes; record available provider/cache qualification honestly.
4. Update architecture/user documentation and preserve bounded logs, captures and source-qualified manifests under committed QA before independent review.
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: qualification and documentation implement the existing approved ADR; any demonstrated defect fix preserves its boundaries.
Final combined fix wave: (1) reproduce and correct same-row Retry copy publication and immutable fork operational snapshot ownership at existing capture/transaction/publication seams; (2) preserve shared currentness witnesses through newly current-choice replay awaits; (3) refuse unsupported server Browse before navigation and classify expected Preview failures; (4) repair sentinel/restoration assertions and qualify only covering native/transaction/UI controls; (5) preserve the four exact root audit artifacts and original final review, update source/evidence manifests with historical live attribution to 247c9eb, attempt one bounded actual Preview check, self-review and commit locally. ADR required: no new ADR. ADR path: backlog/decisions/286-console-prompt-mix-ins.md. Reason: existing approved ownership/source contracts govern; document the explicit fork snapshot handoff detail. Keep In Progress for the single scoped re-review. M4 retirement logging deferred by controller.
Current-dev integration: reconstruct only the reviewed feature delta and explicit design foundation on pinned dev 2acbcd42; preserve upstream owners and historical schemas; allocate ChaChaNotes 86 and Prompts 6; run native lifecycle/fork/retry/source/provider/UI and 85-to-86 migration/recovery/packaging checks; qualify fresh manifests separately from historical captures and verify fresh-clone behavior; commit locally for fresh independent review. ADR required: no new ADR. ADR path: backlog/decisions/286-console-prompt-mix-ins.md. Reason: port existing approved behavior, amend actual migration allocation and evidence provenance.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Qualified the capture-off lifecycle with real local source/store/controller/gateway ownership and exact operational receipt anchors. Fixed the demonstrated selected-generation count projection gap with three existing-owner assignments; Retry authority stays in operational receipts. Final affected tests: 249 passed; predecessor tests: 540 passed; wheel/sdist migration packaging: 1 passed. New test Ruff/format pass; changed product files have zero introduced diagnostics versus dbc3819f (142/142) and identical existing formatter debt.

Actual isolated console-script UI exercised production styles at 160×45, 100×30 and 80×24, keyboard/order, literal preview, queue, Retry original while off, current regeneration, new/reopened chats, invalid-name save and selected-source loss with retained draft/no extra HTTP request. Provider is an explicitly deterministic local responder; external cache qualification unavailable from checked environment credentials. Module-served startup failed before normal launcher succeeded; generic preview failure and retirement log are preserved. No cloud savings claim.

ADR required: no new ADR; implements [ADR-286](../decisions/286-console-prompt-mix-ins.md). Exact commands, historical failure attribution, source-qualified captures, teardown limits and durable evidence: [verification](../../Docs/superpowers/qa/2026-11-01-prompt-mix-ins/verification.md). Keep In Progress until independent review. No full suite, push or merge.
Independent Task9 spec/quality review approved at 247c9eb19d. Native249, predecessor540 and packaging1 pass; committed542 evidence/source records and77 live hashes independently checked. Actual isolated PTY and HTTP receipts qualify controls, queue, original Retry while current off, regeneration off, reopen/new chat, invalid Save and missing source refusal/draft preservation. ADR-286 applies. Earlier cross-slice implementation proof resolves through reviewed slices1–8 and retained QA; whole-feature review follows separately. Three minor observations (restored-count assertion, expected preview feedback/traceback, retirement traceback) carried to final review. External model/cache proof unavailable, full suite not run; no push or merge.
Final whole-feature review at b69e911462 found four Important integration gaps: same-row Retry SQL copy markers, temporary/durable live fork receipts/projections, pre-acceptance source fences in current-choice Regenerate/edit-resend/Continue, and unsupported server Mix-in Browse refusal. Reopened for the single combined final fix wave and scoped re-review. Archive privacy fixture attribution also needs correction (original5pass/current5fail caused duplicate sqlite_sequence row), plus stronger generation assertion, expected Preview feedback and consistent delivery status. Existing retiring browser-read traceback remains a documented separate lifecycle follow-up. ADR-286 remains applicable; update verifiable ACs/plan before product fixes.
Final combined review fixes: same-row Retry SQL copies publish the exact source marker once; immutable fork history retains detached predecessors through live publication, source close/purge, nested temporary forks, Retry and atomic promotion. Fresh acceptance keeps strict live anchors; historical copies clear replacement/result/trace proof. Fresh current-choice replay keeps the existing source witness across awaits. Ordinary server Browse refuses before navigation; expected Preview uses bounded recovery copy. Archive sentinel and scalar generation assertions corrected. ADR-286 amended before the historical handoff implementation. Final covering checks: 51 passed in 25.78s; exact public inventories and held-save readiness pass. Per-file FIX_BASE Ruff: 427/427, zero introduced. Current actual private UI Preview and normal shutdown captured separately from historical whole-journey evidence. Known sparse-policy promotion fails original/FIX_BASE/current; prior 479-case run FD +761 remains unqualified, while 14 new fork cases observe +3 first setup then zero growth each. Full report and durable QA preserve exact outputs, iterations and qualification. Keep In Progress for one scoped re-review; M4 remains deferred. No full suite, push or merge.
Final combined fixes d7744a78e9 and evidence c086d88c56 passed the sole scoped re-review: I1–I4 and M1/M2/M3/M5 addressed, no new breakage; M4 explicitly deferred under controller ruling. Fresh acceptance retains strict live anchors; historical nullable fork state preserves exact receipts, lineage, marker classification, empty selections and no replacement proof authority. Source witnesses cover current-choice routes without changing original Retry/accepted queue. Final covering51 passed, fork14 passed, exact follow-up32 passed (overlapping counts); per-file Ruff427/427 zero additions. Current actual Preview/normal quit verified, historical/current source hashes separate. Original sparse-policy failure and earlier aggregate FD warning retained, not called green. ADR-286 amended and linked; durable final report/re-review/rulings in prompt-mix-ins QA. No full suite, external cache qualification, push or merge.
Focused current-dev port reconstructed on pinned dev 2acbcd42 without unrelated parent commits; backup codex/prompt-mix-ins-reviewed-backup retains reviewed mixed history. Current ChaChaNotes schemas82–85 preserved, mix-ins allocated86, Prompts6. Ported extracted trace-continuation and Library save-ack owners, preserved plugin/hook/compaction contracts. New held-Save raw literal regression passes; empty-list page label corrected from actual app observation. Final covering57 passes; feature native141, storage43, Prompt recovery20 and admitted ordinary Prompt Save8 pass. Current target baseline reproduces disclosed legacy owner/UI/migration/RAG failures; aggregate FD warning retained unqualified. Per-file lint2852 target/2850 current, zero introduced; current-source/historical-evidence provenance separated. Fresh profile86 actual workbench captured at160x45/80x24 with normal Quit. Keep In Progress for controller fresh independent review. ADR286 amended; see current integration report for exact commands, results, baseline limits and fresh-clone verifier.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
