---
id: TASK-38540
title: Compose current-send mix-ins and preserve exact retry behavior
status: Done
created_date: 2026-11-01 19:30
dependencies:
- TASK-38539
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
updated_date: 2026-11-01 23:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply captured modifiers to the current provider request while keeping the transcript clean and distinguishing original Retry from current regeneration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plain, character and agent payloads include enabled Before/user/After content once with deterministic separators and intact multipart attachments; commands, variables, retrieval and dictionaries never reinterpret mix-in bodies.
- [x] #2 Retry uses the original recorded set, including empty sets and capture-off restart, while regenerate, edit-and-resend and explicit new-turn Continue capture current choices; missing new-format snapshots visibly refuse exact Retry.
- [x] #3 Future history, summaries, retrieval and ordinary transcript exports exclude old injections and receipts; preview/token accounting includes current bodies; snapshot persistence failure blocks dispatch and trace projection retains existing redaction.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add literal composition and controller original/current replay regressions. 2. Compose at the verified current-user semantic owner, preserve exactly-once envelope custody, and bind trace/accounting. 3. Persist attempts before provider entry across initial, replay, and temporary promotion seams; select original/current snapshots by action. 4. Extend the existing trace replacement witness with a body-free exact operational attempt identity for clean canonical history after accepted regeneration; revalidate current markers, complete saved response, source ownership, actual parent and no pending dispatch without relaxing ordinary response-link equality. 5. Run targeted owner tests, update architecture, self-review and commit. ADR required: no new ADR. ADR path: backlog/decisions/286-console-prompt-mix-ins.md. Reason: direct implementation of accepted ADR-286.
Review round 1: share explicit synthetic Continue anchors across Retry/Regenerate/edit; pin cancellation drain. Extend existing unreleased v82 operational attempts with explicit replaces_attempt_id and set-once result_revision_id/trace_call_id, bind initial trace via exact dispatch checkpoint, seal replacement result in existing complete-generation transaction, and validate bounded replacement chain to exact origin call. Fork clears trace authority; temporary promotion carries replacement lineage without inventing unsealed result/trace proof. Update exact recovery catalog and focused ownership/restore tests. ADR required: no new ADR; amend existing backlog/decisions/286-console-prompt-mix-ins.md implementation contract.
Fix round 2: Validate the selected original operational snapshot before attaching optional Regenerate replacement lineage. Missing/corrupt old proof suppresses that lineage while current-choice capture and new attempt persistence remain mandatory; exact Retry stays fail-closed. Cover real controller capture On/Off, deleted old source with valid current choices, damaged records/items, and DB-wide write failure. ADR required: no new ADR; existing backlog/decisions/286-console-prompt-mix-ins.md governs current-choice recovery and proof authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-286 current-user literal composition with one semantic attempt marker, clean canonical history, original Retry/accepted Resend and current-choice Regenerate/edit/Continue. Attempts persist before dispatch through native acceptance and replay transactions; temporary sessions retain every full frozen batch and promote transactionally. The unreleased v82 marker trigger and exact recovery catalog now permit validated same-row retry lineage. Trace clean-source restoration revalidates a body-free operational replacement witness; preview includes bodies without publishing usage facts.

Validation: 75 focused composition/controller/agent/DB/recovery cases passed; 201 existing owner checks passed with 11 failures reproduced on exact BASE 19dc3dbb43e5e623d77efc91741b4c49835295fe using identical corrected test harness/admission. New tests lint/format clean, changed-line lint clean, undefined-name checks and git diff --check clean. Full evidence and baseline qualifications: .superpowers/sdd/2026-11-01-prompt-mix-ins-implementation/task-5-report.md. Status deliberately remains In Progress for independent controller review. No new ADR; existing backlog/decisions/286-console-prompt-mix-ins.md applies; architecture docs updated.
Review fix round 1 completed: Retry/Regenerate share the exact synthetic Continue anchor, including captureOff restart. Explicit accepted replacement links and set-once trace/result seals in the existing unreleased v82 attempt table replace marker-only trace authority. Initial call binding uses the exact checkpoint; complete result sealing shares generation settlement, so edits/missing or unequal original links cannot inherit proof. Fork clears authority, temporary promotion invents none, and diagnostic GC clears proof while retaining exact Retry snapshots. Cancellation during attempt persistence drains the write before publication/raising and never enters the adapter. ADR286 and architecture updated before/with schema change; exact recovery catalog matches. Final targeted changed-risk run: 95 passed in 58.33s. Exact per-file Ruff BASE/current code-message multiplicity: 448/448, no introduced diagnostics (SIM102 fixed). No broad owner/full suite rerun. Evidence appended to task-5-report.md; status remains In Progress for independent re-review.
Fix round 2: Current-choice Regenerate validates the optional original replacement link through the existing exact operational reader. Missing/corrupt originals omit proof lineage; Retry remains unavailable and current frozen snapshot persistence remains mandatory before dispatch. Real controller coverage uses deleted original source and distinct valid current B under capture On/Off, with missing attempt/missing or corrupt items and read+write DB failures. Targeted covering run: 35 passed, 22 deselected in 34.89s. Exact fix-BASE/current per-file Ruff diagnostic multiplicities: 180/180, no introduced diagnostics. Existing ADR286 applies; Console architecture updated. In Progress for independent re-review.
Independent review cleared after 8c551fe41f corrected Continue replay anchors and sealed accepted replacement/result/trace provenance, and 98c32189b6 restored current-choice Regenerate when original Retry proof is unavailable. Exact Retry remains fail-closed. Final fix rounds:95 and35 targeted pass respectively (overlap); exact per-file static comparisons introduce no diagnostics. The earlier201/11 owner run remains BASE-qualified, not green. ADR286 amended for three body-free proof columns on unreleasedv82; fork clears authority, promotion retains inert lineage only. No live-provider/full-suite claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
