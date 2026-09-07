---
id: TASK-15391
title: Benchmark visual compaction with GPT-5.6 Terra
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 18:23'
updated_date: '2026-08-11 18:38'
labels:
  - console
  - context
  - evals
dependencies: []
references:
  - >-
    backlog/tasks/task-15263 -
    Harden-visual-compaction-evaluation-output-diagnostics.md
  - backlog/decisions/054-deterministic-visual-transcript-compaction.md
documentation:
  - Docs/superpowers/qa/visual-compaction-model-evaluation/README.md
modified_files:
  - tldw_chatbook/Chat/console_visual_evaluation.py
  - Tests/Chat/test_console_visual_evaluation.py
  - Tests/Chat/test_console_prepared_request.py
  - Tests/Chat/test_chat_functions.py
  - Docs/superpowers/qa/visual-compaction-model-evaluation/README.md
  - Docs/superpowers/qa/visual-compaction-model-evaluation/support-matrix.json
priority: medium
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace obsolete GPT-4o visual-compaction evidence with a current GPT-5.6 Terra benchmark and prove evaluator-v2 structured-output routing through the production Console gateway.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GPT-5.6 Terra on the official OpenAI endpoint receives the evaluator-v2 strict JSON Schema through the final Chat Completions adapter payload
- [x] #2 Unsupported and custom OpenAI-compatible routes retain prompt-only enforcement and are labeled honestly
- [x] #3 Exactly one text-baseline request and one visual-transcript request are executed with GPT-5.6 Terra after local verification
- [x] #4 Checked-in support evidence replaces the obsolete GPT-4o result with a content-free GPT-5.6 Terra evaluator-v2 report
- [x] #5 The QA guide records the current model contract, live result, recommendation outcome, and isolation evidence without changing the compaction default
- [x] #6 Focused tests, static analysis, mutation checks, and post-rebase verification pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the official GPT-5.6 Terra capability and final Chat Completions payload. 2. Add exact evaluator-only structured-output routing while preserving prompt-only unsupported routes. 3. Add prepared-request, image, schema, endpoint, and fallback regression coverage plus mutations. 4. Run focused tests, Ruff, compileall, and self-review before network access. 5. Execute the authorized two-request isolated evaluation and replace GPT-4o evidence. 6. Update docs and task notes, rebase latest dev, re-verify, address PR review, and merge while ignoring CI status as instructed. ADR required: no. ADR path: backlog/decisions/054-deterministic-visual-transcript-compaction.md. Reason: ADR-054 already owns this evaluation contract and no storage, provider ownership, or default-policy boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added exact gpt-5.6-terra evaluator-v2 structured-output routing for the official OpenAI endpoint while preserving prompt-only unsupported/custom routes and legacy compatibility. Added regression coverage through immutable prepared requests and the final Chat Completions HTTP payload, including images, strict JSON Schema, endpoint selection, and max_completion_tokens. The explicitly authorized isolated live run made exactly two Terra requests, both HTTP 200; the normal config SHA-256 and data-tree fingerprint were unchanged. Replaced GPT-4o evidence with evaluator-v2 Terra evidence: 1,032 text input tokens versus 2,881 visual, -179.2% reduction, visual OCR 0.9249, code/math 1.0, recall 0.80, adversarial safety passed, recommendation not_recommended, and no default change. Post-rebase verification: 30 evaluator tests and 2 final-payload seams passed; Ruff and compileall passed; supported-route and custom-endpoint mutations were killed. A broader diagnostic reproduced existing Windows Proactor/network-guard setup errors and one unrelated Anthropic assertion after 104 passes, so focused deterministic suites are the closeout evidence. ADR check: no new ADR; ADR-054 applies.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated GPT-5.6 Terra through the production evaluator, replaced obsolete GPT-4o evidence, and retained the text-summary default because Terra used more tokens and missed quality gates.
<!-- SECTION:FINAL_SUMMARY:END -->
