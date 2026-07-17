---
id: TASK-286
title: Audit remaining dead 'temperature' PROVIDER_PARAM_MAP keys
status: Done
assignee:
  - '@claude'
created_date: '2026-07-17 05:04'
updated_date: '2026-07-17 05:36'
labels:
  - providers
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The review-minors sweep fixed the dead 'temperature' generic key (the dispatcher's generic name is 'temp') for groq/deepseek/mistral/mistralai/google after verifying each handler's parameter name. Six more entries still carry dead keys and silently drop temperature today: cohere, openrouter, huggingface, koboldcpp, local_llamacpp, local_llamafile — huggingface and koboldcpp each have DUPLICATE 'temperature' keys in one dict literal (one mapping to 'temp', one to 'temperature'; Python keeps the last). Each fix needs the handler's actual parameter name verified before flipping the key, since a mismatched name would crash calls that currently just drop temperature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every remaining 'temperature'-keyed entry is either fixed to 'temp': '<verified handler param>' or documented why it must stay
- [x] #2 Duplicate temperature keys in huggingface/koboldcpp are deduplicated
- [x] #3 A forwarding test pins temp reaching each fixed handler
- [x] #4 Dead 'prompt' map keys (vestigial from the pre-messages_payload API, present in every provider entry) are removed,An invariant test pins that every PROVIDER_PARAM_MAP generic key is an actual chat_api_call parameter (kills the dead-key class permanently)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit executed programmatically: every PROVIDER_PARAM_MAP generic-side key diffed against chat_api_call's actual parameters. Findings + fixes: (1) all six remaining dead 'temperature' entries fixed to 'temp': 'temp' after per-handler signature verification (cohere/openrouter/huggingface/koboldcpp -> temp; local_llamacpp + local_llamafile route to chat_with_llama which takes temp — their old 'temperature'->'temperature' mapping would have CRASHED the handler had the key ever fired); (2) huggingface + koboldcpp duplicate temperature keys deduplicated (one entry each); (3) BONUS same-class finding, AC-extended before implementing: dead 'prompt' keys (vestigial from the pre-messages_payload API) removed from all 28 entries — including a vllm straggler caught only by the new invariant test; (4) forwarding pins extended to all six providers (parametrized, mocked handlers); (5) invariant test test_provider_param_map_has_no_dead_generic_keys pins every map key to inspect.signature(chat_api_call) — the dead-key class is now structurally impossible to reintroduce. 273 targeted + 95 provider-suite tests green.
<!-- SECTION:NOTES:END -->
