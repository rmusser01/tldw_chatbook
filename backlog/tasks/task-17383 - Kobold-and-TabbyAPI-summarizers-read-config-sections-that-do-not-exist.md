---
id: TASK-17383
title: Kobold and TabbyAPI summarizers read config sections that do not exist
status: In Progress
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
- [x] #3 Each summarizer resolves its endpoint and credential from a source the loader actually populates, preferring the modern per-provider settings table like the chat handlers do
- [x] #4 The remaining local summarizers are audited for reads of absent sections or keys, and each is fixed or recorded as sound
- [x] #5 Tests cover the configuration resolution for both providers without contacting a network
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

## Implementation Notes (partial -- AC #1 and #2 blocked)

<!-- SECTION:NOTES:BEGIN -->
Both summarizers now resolve their endpoint, credential and model from tables
the loader actually builds -- the modern `api_settings` entry first
(`koboldcpp` / `tabbyapi`, both of which exist and carry `api_url`), then the
legacy section (`kobold_api` / `tabby_api`, which carry `api_ip`, `api_key` and
for Tabby `model`). Two shared helpers do the resolution by name rather than by
index, so a missing section can never raise again.

Credential resolution keeps a distinction the previous code had by accident and
the contract tests pin deliberately: a key declared as an empty string is
"configured but blank" (proceed with no Authorization header), while a key that
is absent or declared null is "absent" (refuse). Collapsing the two turned a
working blank-credential call into a failure, which those tests caught.
`api_key_env_var` is honoured because tabbyapi's modern table carries only that
field and two UI surfaces already read it -- following the repo's convention,
not inventing one.

AC #4, the audit: an AST pass over every `summarize_with_*` shows only these two
read sections the loader never builds. `local_llm` has no direct config reads,
`llama` was fixed in task-17382, and `oobabooga`, `vllm`, `ollama`,
`custom_openai` and `custom_openai_2` all read sections that exist.

**AC #1 and #2 remain open, blocked on task-17387.** Both functions are
GENERATOR functions -- a top-level `yield` in their streaming branches -- so a
call returns a generator and the body never runs; no request reaches either
server regardless of configuration. A prototype fix worked but re-attributed
roughly twenty diagnostics to nested functions and broke seven tests in the
security-reviewed ledger suite, so it was reverted and filed separately rather
than landed as a side effect of a configuration fix.

Verified: the diagnostic privacy suite fails exactly the 3 manifest-boundary
tests clean dev fails (3 failed / 254 passed on both sides), and 1106 tests pass
across the LLM_Calls, pipeline and research suites.
<!-- SECTION:NOTES:END -->
