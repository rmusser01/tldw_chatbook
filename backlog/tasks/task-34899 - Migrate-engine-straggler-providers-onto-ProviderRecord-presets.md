---
id: TASK-34899
title: Migrate engine-straggler providers onto ProviderRecord presets
status: To Do
created_date: 2026-10-12 17:53
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
groq deepseek mistral openrouter huggingface and the two custom_openai handlers are hand-written prototypes of the ADR-179 engine (build_hosted_chat_handler); every observed delta maps to an existing ProviderRecord field. Evidence-gated per-provider migration per ADR-062.

Evidence (verified on dev df4985e): groq.py deepseek.py mistral.py openrouter.py total 1351 LOC (~85 percent identical); huggingface handler ~624 LOC in LLM_API_Calls.py:4381; custom_openai twins ~380 LOC in LLM_API_Calls_Local.py:2031+2222 with the engine replacement already behind [console] custom_endpoints_use_engine. Mistral blocker: caller-supplied safe_prompt and random_seed need an engine param-projection mechanism. Tests/LLM_Calls/test_groq_openrouter_migration_characterization.py (16 tests) pins response shapes. May split per provider into sub-PRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 groq deepseek openrouter become ProviderRecords with characterization tests green before each cutover,mistral migrates after the caller-knob projection gap is filled,huggingface collapses to a preset,custom_openai and custom_openai_2 collapse onto the CUSTOM_HOSTED engine path once the kill-switch default is proven,PROVIDER_PARAM_MAP entries collapse into the engine map,moonshot and zai excluded per ADR-179 decision 4
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
