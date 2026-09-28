---
id: TASK-33201
title: >-
  Six more inference-cloud engine presets from public docs (SambaNova, NVIDIA
  NIM, DeepInfra, Nebius, Novita, MiniMax)
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-28 02:37'
updated_date: '2026-09-28 03:30'
labels:
  - providers
  - engine
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the ADR-179 hosted-provider engine with six OpenAI-compatible inference clouds as data-only preset records, derived from each provider's public documentation (no captured fixtures, no API keys). Two engine gaps the docs exposed are closed along the way: streamed usage must be requested explicitly on OpenAI-semantics providers, and one provider documents streams without usage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SambaNova, NVIDIA NIM, DeepInfra, Nebius Token Factory, Novita AI and MiniMax are selectable and dispatch through the hosted engine with no per-provider module
- [x] #2 Every base URL, env var, allowance and reasoning setting is traceable to a cited public doc page
- [x] #3 Streamed replies from providers that only send usage on request (SambaNova, Nebius, Novita, MiniMax) receive usage instead of failing
- [x] #4 NVIDIA NIM streams without usage complete instead of failing, while every other strict preset still requires usage
- [x] #5 Existing presets (Databricks, Together, Fireworks, Cerebras, custom-hosted) send byte-identical payloads
- [x] #6 GitHub Models and Hyperbolic are documented as excluded (both retired)
- [x] #7 Settings/Console user guides and README list the new providers and env vars
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Research each provider's public docs in parallel (base URL, auth, env var, /models, tools, streaming usage, response extras, reasoning, finish reasons); drop retired services.
2. Close the engine gaps the docs expose (streamed usage on request; streams without usage) as default-off record flags.
3. Add six registry records + dispatch/param map + the parity-guarded hand lists + config tables + discovery gate.
4. Pin every documented value in a new test file with doc-shaped bodies and negative controls; mutation-check it.
5. Update Settings/Console guides and README.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Six strict engine presets built from public docs (read 2026-09-27; no keys, no fixtures): SambaNova, NVIDIA NIM, DeepInfra, Nebius Token Factory (renamed from AI Studio), Novita AI, MiniMax. GitHub Models (retired 2026-07-30) and Hyperbolic (serverless inference retired) were dropped after confirming on the providers' own pages.

Engine: the engine never sent stream_options, and strict records fail a stream that ends without usage, so providers that only stream usage on request could not stream. Added ProviderRecord.stream_include_usage (SambaNova, Nebius, Novita, MiniMax) and stream_usage_optional (NVIDIA; its chunk schema has no usage, finish reason still required). Both default off: Databricks/Together/Fireworks/Cerebras/custom-hosted payloads are byte-identical (pinned). Open question, not changed here: whether those existing presets stream usage without the option -- they are unverified the same way.

Allowances are exactly the fields each doc publishes (e.g. MiniMax base_resp + sensitivity flags + message name/audio_content; Nebius/DeepInfra service_tier; SambaNova choice logprobs + delta reasoning/channel). content_filter finishes (Nebius, MiniMax) map to a 502 provider error. Novita/MiniMax send separate_reasoning/reasoning_split so reasoning lands in reasoning_content instead of <think> text. MiniMax never reads OPENAI_API_KEY (its quickstart repoints it). /models: unauthenticated GETs confirmed OpenAI-shaped lists for NVIDIA (undocumented but live), SambaNova, Novita, DeepInfra (/v1/openai/models, added to the discovery path gate); Nebius and MiniMax answer 401. MiniMax documents no /models, so it ships seeded from its documented model enum with auto-refresh off.

Tests: Tests/LLM_Calls/test_doc_derived_presets.py (62 tests, mutation-checked: disabling each engine flag or a MiniMax allowance turns tests red). test_app_model_catalog_wiring now derives its expected provider list from AUTO_REFRESH_PROVIDER_LIST_KEYS instead of re-pinning it. Local failures match origin/dev by name (RecoveryRequired local gate); test_console_settings_modal_provider_round_trip_ignores_none_model_sentinel fails 3/3 on clean dev too (pre-existing).

Qodo review (3 findings, all fixed): a nonzero MiniMax base_resp status alongside valid choices used to be returned as a reply -- new status_envelope_key field checks it on every body and stream event (nonzero -> 502, code only); MiniMax manual discovery no longer probes an undocumented /v1/models (discovery_route=None = seeded-only, refused by the discovery gate); docstrings on the new tests. Each fix mutation-checked.

Follow-up not done: capture_cloud.py cannot capture these yet (needs /models, sends no stream_options/extra body fields).
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Created as TASK-33126 on branch feat/inference-cloud-presets-2 (PR #2872). PR #2817 (Guardian x Dreams) landed TASK-33126 on dev first (2026-09-27 23:14 PT), so per the TASK-19601 owner rule (older arrival keeps the id) this task renumbered to TASK-33201 -- the first id above every id on origin/dev (max 33162), open remote branches (33165/33175/33200) and local worktrees (33163-33165/33200) on 2026-09-28.
