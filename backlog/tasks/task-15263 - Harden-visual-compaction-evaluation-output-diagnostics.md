---
id: TASK-15263
title: Harden visual compaction evaluation output diagnostics
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 17:16'
updated_date: '2026-08-11 17:32'
labels:
  - console
  - context
  - evals
dependencies: []
references:
  - >-
    backlog/tasks/task-15262 -
    Evaluate-visual-transcript-compaction-across-vision-models.md
  - backlog/decisions/054-deterministic-visual-transcript-compaction.md
priority: medium
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make visual-compaction quality evidence diagnosable without persisting transcript or raw model output, and use provider-native structured output only where the selected provider/model path safely supports it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Invalid evaluator responses report a stable content-free failure reason without storing response content
- [x] #2 The selected provider/model receives a strict JSON response contract only when its production gateway path supports that contract
- [x] #3 Unsupported providers retain prompt-only enforcement and are labeled honestly in the report
- [x] #4 Support-matrix loading and recommendation gates remain backward-compatible with evaluator-v1 evidence
- [x] #5 Tests cover failure-reason classification, structured-output routing, redaction, fallback, and unknown-never-passes behavior
<!-- AC:END -->

## Implementation Plan

1. Characterize the normal Console gateway's prepared-request and adapter mappings, then define a conservative structured-output support contract for evaluator-only calls.
2. Version evaluator evidence to v2 with stable content-free parse-failure reasons and an explicit output-enforcement mode, while retaining strict v1 matrix loading.
3. Thread an optional immutable response format through the prepared-request dispatch path; leave ordinary Console sends byte-identical and enable it only for supported evaluator provider/model selections.
4. Add focused tests for parser classification, structured-output routing, unsupported fallback labeling, evidence redaction, v1 compatibility, and the unknown-never-passes recommendation guard.
5. Run the scoped test and static-analysis suites, self-review the diff, and document the completed evidence without making a billable live-model call unless separately authorized.

ADR required: no
ADR path: backlog/decisions/054-deterministic-visual-transcript-compaction.md
Reason: ADR-054 already defines the evaluator's provider wire path, content-free persisted evidence, and separate default-enablement gate; this task adds diagnostics and an optional evaluator-only request contract without changing storage ownership, provider boundaries, or the default policy.

## Implementation Notes

- Added evaluator-v2 evidence with stable content-free parse-failure categories and an explicit `provider_json_schema` or `prompt_only` enforcement label; raw output remains in-memory and persisted only as a SHA-256 digest.
- Added an immutable optional response format to prepared Console requests. Documented GPT-4o/GPT-4o mini models on the official OpenAI endpoint receive a strict evaluator JSON Schema and the checked endpoint is pinned into final adapter kwargs; all other routes remain prompt-only. Ordinary Console sends continue to omit both fields.
- Preserved strict evaluator-v1 loading and exact serialization, including mixed v1/v2 matrices produced when the CLI appends a new report to existing evidence. Unknown, malformed, and synthetic results still cannot pass the recommendation gate.
- Updated the evaluator QA guide and testing lesson. No billable live-model call was made; the checked-in v1 support matrix was intentionally left unchanged.
- Verification: 28 focused tests passed; Ruff checks passed; compileall passed. Endpoint-routing and unknown-never-passes mutations were both killed. A broader combined run reached 38 passing tests before 12 existing Windows Proactor setup/teardown errors were raised by the repository network guard; the scoped synchronous gateway and complete evaluator suites passed independently.
