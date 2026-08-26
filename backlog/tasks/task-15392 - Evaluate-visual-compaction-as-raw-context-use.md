---
id: TASK-15392
title: Evaluate visual compaction as raw context use
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 18:53'
updated_date: '2026-08-11 19:05'
labels:
  - console
  - context
  - evals
dependencies: []
references:
  - backlog/tasks/task-15391 - Benchmark-visual-compaction-with-GPT-5.6-Terra.md
  - backlog/decisions/054-deterministic-visual-transcript-compaction.md
  - backlog/decisions/056-context-use-visual-compaction-evaluation.md
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the visual-compaction benchmark so models use deterministic transcript images as historical context for a downstream request instead of transcribing or extracting the image contents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Evaluator-v3 output and prompts contain downstream probe answers and safety state only, with no transcript extraction, summary, or restatement field
- [x] #2 The text and visual requests use the identical downstream active request and differ only in the representation of historical context
- [x] #3 Context-use readiness derives from measured input savings, code/math recovery, instruction recall, and adversarial safety without an OCR or transcription gate
- [x] #4 Evaluator-v1 and evaluator-v2 matrices remain exactly loadable, but legacy transcription evidence cannot make a v3 matrix eligible
- [x] #5 The QA guide and a new ADR supersede the misleading evaluator-v2 recommendation and document the raw-context contract
- [x] #6 Focused tests, static analysis, payload invariants, and mutation checks pass before any separately authorized billable rerun
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the evaluator-v2 contract and the production paired context request. 2. Create and link ADR-056 defining raw-image context-use evaluation; ADR required: yes; ADR path: backlog/decisions/056-context-use-visual-compaction-evaluation.md; Reason: this changes the long-lived evaluation contract and default-enablement evidence for visual compaction. 3. Implement evaluator-v3 context-use prompts, structured output, readiness gating, and v1/v2 compatibility. 4. Add focused schema, payload-invariant, backward-compatibility, and mutation tests. 5. Update QA guidance to supersede the transcription-based v2 recommendation. 6. Run focused tests and static checks, then self-review. 7. Stop before any billable live calls unless separately authorized.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented evaluator-v3 as a downstream context-use comparison: paired requests
share the exact active probe request and return only answers plus the adversarial
safety flag. Added a separate ADR-056 readiness gate using measured input-token
savings, code/math recovery, instruction recall, and safety; v3 reports omit OCR.
Schema-v1/v2 reports still round-trip exactly as legacy transcription evidence,
but cannot make a schema-v3 matrix eligible. Updated the QA guide to label the
checked-in v2 Terra result methodologically superseded; no billable calls were made.

Verification: 37 focused evaluator/renderer tests passed; Ruff check and format
check passed; compileall passed. Three deliberate mutants (reintroducing a
transcript output field, admitting a ready v2 report to v3 eligibility, and
removing the positive-savings gate) were each caught by the focused tests. An
adjacent combined run reached 50 passing tests before six prepared-request async
tests failed during fixture setup because the Windows Proactor event loop's local
socket pair was blocked by the repository network guard; no affected test body ran.

ADR required: yes. ADR path:
`backlog/decisions/056-context-use-visual-compaction-evaluation.md`. It amends
ADR-054 because the evidence contract and long-lived default-enablement semantics
changed.
