---
id: TASK-18803
title: >-
  OpenAI, Moonshot and Z.ai request builders gate parameters on hand-maintained
  model-name checks
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18 23:56'
updated_date: '2026-08-20 20:37'
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
- [x] #1 Each claim above is reproduced (or disproved) against the real provider API and the finding recorded before any code changes
- [x] #2 Per-model request capabilities for these providers are expressed as predicates in model_capabilities.py rather than name checks in the request builders
- [x] #3 A new model release in a covered family does not require editing a marker list to avoid a 400
- [x] #4 Models currently working are unchanged, pinned by regression tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce builder-side claims locally (payload capture at the HTTP seam): OpenAI gpt-5/o3/o4-mini emit max_tokens with no reasoning effort (:695-698); family-miss characterization for _OPENAI_REASONING_MODEL_FAMILIES; Moonshot reasoning_effort ChatBadRequestError off kimi-k3 and silent sampling drop off moonshot-v1-*; Z.ai reasoning_effort pin off glm-5.2 and unconditional thinking.\n2. Wire evidence: dump the exact built gpt-5 payload and curl it (known 400 expected); Moonshot model-list + param-acceptance probes with the real key; Z.ai recorded unprobeable (no key).\n3. Fixes: chat_with_openai consults openai_model_rejects_sampling_params / openai_model_requires_max_completion_tokens (18802 predicates); new family-tolerant Moonshot/Z.ai reasoning-effort + Moonshot sampling predicates in model_capabilities.py; builders consult them.\n4. Red-first pins, controls (gpt-4o/gpt-4.1, moonshot-v1, kimi-k3, glm-5.2 unchanged), mutation tests with Edit-based restores.\n5. Live: chat_api_call seam with real keys (gpt-5 no effort, gpt-5 with effort, gpt-4o, one Moonshot turn) + clean origin/dev control showing the gpt-5-no-effort 400.\n6. Hygiene: report, notes, PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
COMPLETE (2026-08-20, branch fix/task-18803-chat-path-model-gates; full report: Docs/superpowers/plans/2026-08-20-task-18803-report.md).

STEP 1 (AC #1, recorded in commit cc6a576bc BEFORE any fix): all findings reproduced at the HTTP seam with the production builders. Headline (finding 2) wire-confirmed: the EXACT payload chat_with_openai builds for gpt-5 with no reasoning effort -> HTTP 400 unsupported_parameter max_tokens. Finding 1's tuple mechanism reproduced (gpt-6/o5 family-miss re-injects temperature 0.7/top_p 0.95); API side cited from TASK-18802's probes, not re-run. Moonshot wire boundary mapped with the real key: GET /v1/models serves kimi-k2.5/k2.6/k2.7-code(+highspeed)/k3/kimi-latest + moonshot-v1-*; versioned kimi rejects non-default sampling VALUE-level ('invalid temperature: only 1 is allowed for this model'; top_p 'only 0.95'; presence_penalty 'only 0'; temperature=1 -> 200) while kimi-latest accepts the full five-param set (chatcmpl-6a872b9816ceb0c0ae780b1e) -- and reasoning_effort answers 200 on k2.6/k2.7-code/kimi-latest and 'medium' on k3/k2.6, DISPROVING the exact-id pin. Z.ai: client-side rejection of glm-5.3/glm-6/glm-5.2-air reproduced locally; wire UNPROBEABLE (no key in repo root), recorded; unconditional thinking payload reproduced (also on glm-4.6) and left unchanged -- no wire evidence, no local breakage.

FIX (AC #2/#3): chat_with_openai consults the existing 18802 predicates -- openai_model_rejects_sampling_params for the sampling omission (hand tuple _OPENAI_REASONING_MODEL_FAMILIES deleted) and openai_model_requires_max_completion_tokens for the token-cap key (closing the no-effort max_tokens fall-through); _is_openai_gpt_5_6_model kept ONLY for Responses-API selection and the chat-completions reasoning_effort knob -- a different question from the token cap. New immutable predicates in model_capabilities.py (outside the config tables, boundary-safe parsing): moonshot_model_supports_reasoning_effort (whole kimi series), moonshot_model_rejects_sampling_params (versioned kimi only; kimi-latest/v1/unknown now pass sampling through, replacing the frozen moonshot-v1- allowlist), moonshot_model_requires_min_temperature_for_multiple_choices (v1 interplay rule), zai_model_supports_reasoning_effort (GLM >= 5.2 version floor -- conservative liberalisation, not wire-verified, documented as such). Kimi effort value set gains 'medium' (wire-verified); Settings' kimi-k3 selector and the settings.md guide rows aligned.

EVIDENCE (AC #4): red-first 16 failed/176 passed with builders untouched -> 192 passed after; cluster gate 1186 passed (2 pre-existing TASK-18801 manifest reds, identical on clean base d851e7977; 2 opt-in live skips). Six mutations (invert each consultation, restore each literal gate, bump the floor) kill 8/1/5/3/3/15 pins respectively incl. every glm-5.2 control; all Edit-restored, git diff clean. LIVE through chat_api_call with real keys: gpt-5 no-effort 200 'OK' (chatcmpl-EF3guz2RLQSTaEVlTmA9OWmBzub2i, THE headline), gpt-5+effort 200 via /responses, gpt-4o control 200, kimi-k2.6+medium-with-temp-dropped 200; dev control on clean d851e7977: gpt-5 no-effort -> the exact 400, gpt-4o -> 200 identical.

FILED: TASK-19170 (response-side/UI kimi-k3//glm-5.2 literal pins -- continuation replay, keep_all, Settings selector scoping; CLI assigned colliding 19052, renumbered after ghost-check). Dev reds already tracked as TASK-18801.
<!-- SECTION:NOTES:END -->
