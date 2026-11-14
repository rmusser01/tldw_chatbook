---
id: TASK-38735
title: "The custom-provider literal audit is red on dev"
status: To Do
created_date: 2026-11-04 03:08
labels:
- follow-up
- architecture-review-2026-10-25
- providers
priority: medium
references:
- Tests/Chat/test_custom_openai_execution_keys_constant.py
- tldw_chatbook/Chat/provider_catalog.py
dependencies:
- TASK-36946.1
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev d204ddce70 (2026-11-03), `Tests/Chat/test_custom_openai_execution_keys_constant.py::test_no_bare_custom_literals_outside_the_audited_allowlist` fails on four files:
- Chat/provider_catalog.py has 3/1 unexpected literals. TASK-36946.1 added these: the `custom`/`custom_2` display overrides and the `custom-openai-api` slot derivation.
- Chat/provider_readiness.py has counts 2/1 against a pin of 1/1.
- LLM_Calls/Summarization_General_Lib.py has counts 2/2 against a pin of 1/1.
- Library/library_rag_answer_service.py has 1/1 unexpected literals.

The test was already red before the architecture-review Phase 2 merge, which hid the provider_catalog additions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed file names custom providers through the shared custom-execution-key constants, or has an audited allowlist row with a one-line reason
- [ ] #2 The audit test passes on dev
- [ ] #3 The test file runs on a PR lane, or the task notes say why not
<!-- AC:END -->
