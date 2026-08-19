---
id: TASK-18803
title: >-
  OpenAI, Moonshot and Z.ai request builders gate parameters on hand-maintained
  model-name checks
status: To Do
assignee: []
created_date: '2026-08-18 23:56'
labels:
  - llm
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-18414 replaced two hand-maintained Anthropic model-name lists with capability predicates in model_capabilities.py, after they went stale and made every send to claude-opus-5 an HTTP 400. A sweep of LLM_Calls/ found the same pattern still live for three other providers. All findings below are CODE-READ, not probe-verified -- each needs a reproduction before being fixed.

1. LLM_API_Calls.py:245/248 -- _OPENAI_REASONING_MODEL_FAMILIES / _is_openai_reasoning_model, a hand-maintained tuple ('o1','o3','o4','gpt-5'), consumed at :681 to omit temperature and top_p. Structurally identical to the Anthropic list that was just removed; a miss is a 400. Its own docstring cites the same historical incident (task-404).

2. LLM_API_Calls.py:218 -- _is_openai_gpt_5_6_model, matching exactly gpt-5.6 and gpt-5.6-*, deciding which key carries the token cap (max_output_tokens / max_completion_tokens / max_tokens) at :660, :695, :738. The sweep reports a live hole rather than only a future one: _openai_use_responses_api (:233) returns True only when a reasoning effort/summary/verbosity is set, so gpt-5, gpt-5.1, o3 and o4-mini with no reasoning effort configured are said to fall through to payload['max_tokens'] at :698, which OpenAI rejects in favour of max_completion_tokens. Reproduce this first -- it is the highest-value claim in the sweep and the repo has an openai-api-key.txt.

3. moonshot.py:527 -- an inline startswith('moonshot-v1-') allowlist that only ADDS temperature/top_p/n/presence_penalty/frequency_penalty for that prefix, so on kimi-k3 or any future Kimi id the user's sampling settings are silently dropped rather than rejected. Lower urgency (silent, not a 400) but the same staleness mechanism.

4. moonshot.py:516 and zai.py:292 -- reasoning_effort pinned to the single literals kimi-k3 and glm-5.2 respectively, each raising a client-side ChatBadRequestError for any other model. Worse than the Anthropic case in one respect: an exact-id pin rather than a family, so a kimi-k3-turbo or glm-5.3 release is rejected before a request is ever made.

Adjacent, no name check at all: zai.py:269 puts thinking={'type':'enabled',...} into every payload unconditionally.

model_capabilities.py now has the right shape to extend -- a prefix/suffix-tolerant family parser plus per-question predicates -- but nothing equivalent exists for OpenAI, Moonshot or Z.ai.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each claim above is reproduced (or disproved) against the real provider API and the finding recorded before any code changes
- [ ] #2 Per-model request capabilities for these providers are expressed as predicates in model_capabilities.py rather than name checks in the request builders
- [ ] #3 A new model release in a covered family does not require editing a marker list to avoid a 400
- [ ] #4 Models currently working are unchanged, pinned by regression tests
<!-- AC:END -->
