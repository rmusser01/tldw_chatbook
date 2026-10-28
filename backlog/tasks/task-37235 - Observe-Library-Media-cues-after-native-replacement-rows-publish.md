---
id: TASK-37235
title: Observe Library Media cues after native replacement rows publish
status: Done
created_date: 2026-10-28 18:55
labels:
- textual
- testing
- ci
- library
priority: high
references:
- backlog/tasks/task-34000.18 - Library-Media-Read-later-trash-and-Edit-metadata-act-on-the-item-opened-by-link-not-row-1.md
- backlog/decisions/084-library-media-reader-ia.md
- backlog/decisions/055-library-destructive-action-reversibility-rule.md
assignee:
- '@codex'
updated_date: 2026-10-28 19:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Library Media action journey deterministic when a saved Reader title lands before the sibling Items canvas finishes native batched recomposition. A controlled held mount reproduces the empty DOM while canonical identity and the last eligible painted cue remain correct. This is a narrow test-publication repair linked to completed TASK-34000.18, not a duplicate identity or Reader-action implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Edit Save, the original journey waits for the current replacement rows and edited row title before making its unchanged separate highlight assertion; existing write-target and Reader identity assertions remain intact.
- [x] #2 A controlled native mount hold proves the publication condition cannot settle while replacement rows are absent and does settle after release, with normal guards and cleanup.
- [x] #3 A published list with the selected cue deliberately removed remains a failing semantic control; the publication wait does not wait on or weaken the highlight assertion.
- [x] #4 Targeted original and neighboring Media tests plus static checks pass; production source, native lifecycle, timeouts, profile guards and existing acceptance semantics remain unchanged.
- [x] #5 Existing TASK-34000.18 and ADR-084/055 are linked; actual RED/GREEN receipts and scoped Superpowers review are retained.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A (existing backlog/decisions/084-library-media-reader-ia.md and055-library-destructive-action-reversibility-rule.md).
Reason: test-only synchronization preserves the accepted TASK-34000.18 contract and product boundaries.
1. Read the task, native investigation and related test/architecture guidance; preserve source and actual receipt provenance.
2. Write a controlled native held-mount regression that fails the original early observer, and a cue-removed negative control distinguishing publication from semantic correctness.
3. Extend only the post-save readiness condition to the accepted current replacement rows and edited title; keep the existing exact highlighted-id assertion separate.
4. Run RED/GREEN, the exact original node and immediate file neighbors one App batch at a time through the native external-profile launcher; verify actual XML and focused static/baseline checks.
5. Retain receipts/source hashes and a concise implementation report; root owns fresh scoped Superpowers review, task closure, measured census enrollment and final current-head CI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Test-only repair for the native publication interval in completed TASK-34000.18. The existing Edit Save journey waits for the accepted current Media page, five attached replacement rows and row3 edited title, then keeps its exact highlighted-id assertion separate. All original service/Reader assertions and other original helpers/tests have identical AST. Product, native batching, profile guards and timeouts remain unchanged.

Authoritative RED03: both native held-mount controls fail the old Reader-only observer at the intended early-publication assertion,2failures0errors/skips. Earlier RED01/02 mixed diagnostic timeout/query errors remain disclosed and are not cleanRED. Final whole original file:12passed0failures/errors/skips,585.55s loaded localWindows session. Both native-hold and missing-cue arms execute; missing-cue uses real apply_reader_state and requires the unchanged semantic assertion to fail independently of readiness. Actual XML/hash/source snapshots and report are retained in .superpowers/sdd/2026-10-27-textual-37031-37034/ci-media-37235-report.md and .test-tmp/textual-37031-37034/media-37235-evidence/. SourceSHA66241205a4ef70430e4fb0feda973fde267febcdab939cebc0d162b7add33567. Ruff/diff checks pass; ten inherited formatter regions remain exact.

Existing ADR-084/055 and TASK-34000.18 apply; no newADR. No demonstrated identity loss or live-terminal defect is claimed: the controlled native probe preserves canonical3 and last render-eligible cue through batched emptyDOM; attribution of the original uninstrumented CI execution is an inference. Implementation/self-review complete; task remains InProgress with AC5 awaiting root independent Superpowers review, final source/evidence recording and actual successor-headCI delivery.
Independent scoped Superpowers spec/code-quality review approved (012ca47c to96c8a33e), Critical0/Important0/Minor0. Reviewer independently checked actual authoritative XML, all12 original assertions, RED/GREEN helper-only AST delta, final source hash, raw decoded frame hashes and all11 portable Media evidence files. All5AC complete; no production change. Actual current-head combined PR CI and delivery remain root-owned separate gates; no hosted pass or natural race frequency is inferred.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
