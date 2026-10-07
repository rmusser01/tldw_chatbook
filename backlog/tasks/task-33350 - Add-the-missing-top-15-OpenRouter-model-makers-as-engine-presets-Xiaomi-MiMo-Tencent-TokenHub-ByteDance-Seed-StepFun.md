---
id: TASK-33350
title: >-
  Add the missing top-15 OpenRouter model makers as engine presets (Xiaomi MiMo,
  Tencent TokenHub, ByteDance Seed, StepFun)
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-28 17:00'
updated_date: '2026-09-28 19:46'
labels:
  - providers
  - engine
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
OpenRouter's usage rankings (week of 2026-09-21, model authors by share of requests, plus its by-token model tables) name the model makers people actually use. Eleven of the top fifteen are already supported and xAI stays excluded by ADR-179. Meta's Llama API was retired on 2026-07-06. That leaves four makers with no first-party provider in tldw_chatbook: Xiaomi (MiMo), Tencent (Hy4, served through Tencent Cloud TokenHub International), ByteDance (Seed models via BytePlus ModelArk) and StepFun. Each should be selectable as a strict ADR-179 engine preset derived from its public documentation. Xiaomi documents only an `api-key` request header, so the engine's planned `api_key_header` auth scheme (ADR-179 Phase 3) is needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Xiaomi MiMo, Tencent TokenHub, ByteDance Seed (BytePlus ModelArk) and StepFun are selectable and dispatch through the hosted engine with no per-provider module
- [x] #2 Every base URL, env var, allowance and reasoning setting is traceable to a cited public doc page or a recorded unauthenticated probe
- [x] #3 A preset can authenticate with an `api-key` request header instead of `Authorization: Bearer`, and that credential header cannot be overridden by per-provider extra headers
- [x] #4 Every pre-existing engine preset sends byte-identical requests
- [x] #5 Providers with an undocumented or unusable models route ship a seeded model list and refuse discovery instead of probing it
- [x] #6 Settings/Console user guides and README list the new providers, env vars and region notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rank model makers by OpenRouter usage; drop retired (Meta) and excluded (xAI); research each first-party API's public docs and probe each host unauthenticated.
2. Engine: implement the ADR-179 Phase 3 api_key_header auth scheme (MiMo) and a mid-stream error-frame check (TokenHub).
3. Add MiMo, TokenHub, BytePlus and StepFun records + dispatch, parity lists, continuation pairings, config seeds/tables.
4. Pin every documented value with doc-shaped bodies and negative controls; mutation-check each engine change.
5. Settings/Console guides and README.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four strict engine presets for the OpenRouter top-15 model makers that lacked a first-party provider: Xiaomi MiMo, Tencent TokenHub (Hy4 preview), ByteDance Seed via BytePlus ModelArk, StepFun. Ranking: OpenRouter author request share (week of 2026-09-21) plus its by-token model tables; Meta dropped (Llama API retired 2026-07-06, confirmed via Meta's deprecation page), StepFun substituted; xAI stays excluded (ADR-179).

Engine: (1) api_key_header auth scheme (ADR-179 Phase 3) -- key sent as api-key, no Authorization; key required like bearer; api-key joins Authorization/Content-Type as headers extra_headers may never set; bearer header order unchanged (byte-identical for existing presets). (2) error_frame_key record field -- TokenHub documents a bare {"error": ...} SSE frame after the 200 header; it now raises a 502 provider error (provider text never copied) instead of a misleading malformed-response error. One composed provider_event_check replaces the MiniMax-only lambda.

Review-driven design changes: Tencent targets TokenHub International, not the direct Hunyuan API (China-only, newest documented model hunyuan-a13b, no Hy4). MiMo and BytePlus request streamed usage but tolerate its absence (unconfirmed key / user reports). MiMo, TokenHub, BytePlus reasoning is proprietary and round-tripped (MiMo FAQ and TokenHub Preserved Thinking require it back). MiMo/BytePlus are seeded-only (discovery_route=None): MiMo's models route is unconfirmed and discovery only speaks Bearer; BytePlus's route shape is undocumented. The Console accepts custom model IDs for unlisted models and BytePlus ep- IDs. StepFun ships tools off (docs show finish stop alongside tool_calls). TokenHub's array-valued reasoning_details/search_results stay strict (the allowance value rule rejects arrays).

Verification: Tests/LLM_Calls/test_model_maker_presets.py (49 tests) + 5 new auth tests in test_hosted_provider_engine_auth.py; six mutations (api-key header off, api-key not reserved, key not required, error-frame check off, TokenHub repetition_truncation removed, MiMo back to bearer) each turn tests red. Parity/catalog/readiness/engine/census suites pass; the 9 remaining LLM_Calls failures fail identically on the branch base (7cda... merge-base) by name. preflight clean.

Unverified live: no provider keys; every shape comes from docs and unauthenticated probes. MiMo's acceptance of Bearer is unknown (api-key is the documented header).
<!-- SECTION:NOTES:END -->
