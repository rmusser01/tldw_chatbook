---
id: TASK-15482
title: Run evaluator-v3 Terra raw-context evidence
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 19:14'
updated_date: '2026-08-11 19:23'
labels:
  - console
  - context
  - evals
dependencies: []
references:
  - backlog/tasks/task-15392 - Evaluate-visual-compaction-as-raw-context-use.md
  - backlog/decisions/056-context-use-visual-compaction-evaluation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Produce the first live GPT-5.6 Terra evidence for deterministic visual transcript pages used directly as historical context under the ADR-056 evaluator-v3 contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly two billable requests run against the explicitly pinned official OpenAI GPT-5.6 Terra route: one text-history control and one raw-image-history request
- [x] #2 The run uses evaluator-v3 context_use output with no transcript extraction or OCR field and records complete provider usage or fails closed
- [x] #3 The normal Chatbook config and data profile fingerprints are unchanged by the isolated run
- [x] #4 A schema-v3 support matrix replaces the superseded v2 policy artifact while retaining truthful content-free evidence and eligibility derivation
- [x] #5 The QA guide reports the measured v3 result and clearly separates response max tokens from conversation context length
- [x] #6 Focused tests, static analysis, payload invariants, and a post-run self-review pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the current evaluator-v3 contract, official Terra image/Chat Completions/Structured Outputs support, and the exact isolated two-call CLI path. 2. Fingerprint the normal Chatbook config and data profile and construct a task-specific scratch config without printing credentials. 3. Run exactly one text-history control and one raw-image-history request against pinned official OpenAI GPT-5.6 Terra. 4. Validate schema-v3 context_use evidence, complete provider usage, profile invariants, and content-free persistence. 5. Update the support matrix and QA interpretation with the measured result. 6. Run focused tests, Ruff, format, payload invariants, and self-review. 7. Open a PR against dev, address Qodo comments, rebase latest dev, and merge while ignoring CI checks as requested. ADR required: no. ADR path: backlog/decisions/056-context-use-visual-compaction-evaluation.md. Reason: this task executes and publishes evidence under the existing ADR-056 contract without changing architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Ran the confirmation-gated evaluator-v3 once against the pinned official OpenAI
`gpt-5.6-terra` route: exactly one text-history control and one raw-image-history
request. Both returned valid `context_use` structured output with complete measured
usage, and neither the persisted payload nor response contract contains transcript
or OCR material. The normal config hash and data-profile fingerprint were unchanged.

The text request used 1,060 input / 84 output tokens; the two-page visual request
used 2,909 input / 94 output tokens, a -1.7443 reduction ratio (174.4% more visual
input). Visual code/math recovery was 1.0, instruction recall 0.8, and adversarial
safety passed. The schema-v3 matrix therefore truthfully records
`not_recommended` and no eligible model. Updated the QA guide and its scratch-data
isolation recipe, plus the live-verification lesson documenting why
`TLDW_CONFIG_PATH` alone does not relocate application data.

Verification: 37 evaluator/renderer tests passed after formatting; Ruff check and
format check passed; the production loader round-tripped the checked-in matrix
exactly; content-free, schema, eligibility, metric, and documentation invariants
passed. Self-review found no remaining actionable issue.

ADR required: no. Existing ADR:
`backlog/decisions/056-context-use-visual-compaction-evaluation.md`. This task
executes its evidence contract without changing architecture.
