---
id: TASK-17383
title: Kobold and TabbyAPI summarizers read config sections that do not exist
status: To Do
assignee: []
created_date: '2026-08-17 09:05'
labels:
  - llm-calls
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The llama.cpp summarizer's defect is a family, not a one-off. Two sibling local summarizers index configuration sections that the loader never builds, so like llama.cpp they raise before contacting a server and report failure by returning an error string. Kobold reads an API-keys section and a local-endpoint-address section; TabbyAPI reads those two plus a models section. All three are absent at runtime, confirmed by probing the loaded configuration.

The evidence-contamination consequence is already closed for every provider: the deep-search caller now recognizes a returned provider error string as a failure and falls back to the source content. What remains is that these two summarization paths cannot work at all, and that the code reads section names nothing produces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Kobold summarization request reaches its server instead of failing on a configuration lookup
- [ ] #2 A TabbyAPI summarization request reaches its server instead of failing on a configuration lookup
- [ ] #3 Each summarizer resolves its endpoint and credential from a source the loader actually populates, preferring the modern per-provider settings table like the chat handlers do
- [ ] #4 The remaining local summarizers are audited for reads of absent sections or keys, and each is fixed or recorded as sound
- [ ] #5 Tests cover the configuration resolution for both providers without contacting a network
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence from the task-17382 investigation, before any fix here:

- Probing the loaded configuration: `api_keys`, `local_api_ip` and `models`
  are all MISSING, while `kobold_api`, `tabby_api`, `vllm_api`,
  `custom_openai_api` are present.
- `Local_Summarization_Lib.py:388,394,448` (Kobold) and `:876,883,884`
  (TabbyAPI) index the missing sections.
- Same failure shape as task-17382: the read sits above the HTTP call, and the
  bottom `except` converts the KeyError into a returned error string.
<!-- SECTION:NOTES:END -->
