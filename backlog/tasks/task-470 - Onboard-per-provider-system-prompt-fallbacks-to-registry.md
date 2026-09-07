---
id: TASK-470
title: 'Onboard per-provider system_prompt fallbacks to the Internal Prompts registry'
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
Deferred from the Internal Prompts program. The per-provider 'You are a helpful AI assistant' style system_prompt fallbacks in Summarization_General_Lib / Local_Summarization_Lib and the [api_settings.<provider>].system_prompt defaults are not yet registry-backed. Onboard them as PromptSpecs (one-liners) so they are user-editable, following the established byte-parity + transport-boundary-test pattern.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The provider system_prompt fallbacks are registered as PromptSpecs with byte-identical defaults (golden-parity tests)
- [ ] #2 Call sites resolve via the registry; existing programmatic overrides still win
- [ ] #3 A transport-boundary test proves an override reaches the payload
<!-- AC:END -->
