---
id: TASK-286
title: Audit remaining dead 'temperature' PROVIDER_PARAM_MAP keys
status: Done
assignee:
  - '@claude'
created_date: '2026-07-17 05:04'
updated_date: '2026-07-17 05:48'
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
- [ ] #5 Provider-side map TARGETS verified against handler signatures (tabbyapi/local-llm temp crash + five user_identifier->user dead targets fixed) and pinned by a second invariant test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AMENDED post-review: the reviewer's sweep exposed the provider-SIDE dead-target sibling class — 7 entries whose mapped target name doesn't exist on the handler: tabbyapi + local-llm mapped generic temp to 'temperature' (handlers take temp — a temperature-carrying call TypeErrored, WORSE than the silent drop), and oobabooga/vllm/aphrodite/custom-openai x2 mapped user_identifier to 'user' (handlers take user_identifier). All 7 fixed after authoritative inspect.signature verification (vllm/aphrodite legitimately take 'temperature' for temp — left as-is); second invariant test pins every map target to its handler's signature (or **kwargs). Also from review: llama_cpp + oobabooga temperature keys had been dropped (comment-bearing dict-open lines confused the rewrite script) — restored as temp:temp; unreachable cohere-prompt dispatcher branch deleted. Both dead-key classes (generic-side + provider-side) are now structurally pinned.
<!-- SECTION:NOTES:END -->
