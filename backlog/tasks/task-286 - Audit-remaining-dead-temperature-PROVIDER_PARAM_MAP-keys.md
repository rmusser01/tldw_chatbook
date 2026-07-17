---
id: TASK-286
title: Audit remaining dead 'temperature' PROVIDER_PARAM_MAP keys
status: To Do
assignee: []
created_date: '2026-07-17 05:04'
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
- [ ] #1 Every remaining 'temperature'-keyed entry is either fixed to 'temp': '<verified handler param>' or documented why it must stay
- [ ] #2 Duplicate temperature keys in huggingface/koboldcpp are deduplicated
- [ ] #3 A forwarding test pins temp reaching each fixed handler
<!-- AC:END -->
