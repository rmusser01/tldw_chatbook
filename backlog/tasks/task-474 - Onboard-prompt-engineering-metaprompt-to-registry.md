---
id: TASK-474
title: 'Onboard the prompt-engineering metaprompt to the Internal Prompts registry'
status: To Do
assignee: []
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - enhancement
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the program. Prompt_Management/Prompt_Engineering.py contains the large Anthropic 'metaprompt' used by generate_prompt(). Onboard it as a single PromptSpec so it is user-editable, preserving its exact text and its function-calling example blocks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The metaprompt is registered with a byte-identical default (parity test)
- [ ] #2 generate_prompt() resolves it via the registry
- [ ] #3 Its placeholder/token contract is captured so edits cannot break generate_prompt()
<!-- AC:END -->
