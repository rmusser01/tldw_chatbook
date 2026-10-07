---
id: TASK-33501
title: Console offers reasoning effort for engine presets that refuse it
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:27'
updated_date: '2026-09-29 20:02'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every engine preset whose registry record refuses reasoning effort (NVIDIA NIM, Fireworks, Together, Cerebras and others) still shows a Reasoning-effort choice in the Console settings as 'support not verified'. Picking a level makes every send fail locally with 'reasoning effort is unsupported'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console reports reasoning effort as unsupported for every engine preset whose record refuses it
- [x] #2 Presets and providers that accept reasoning effort keep their current support answer
- [x] #3 Tests pin both cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In console_generation_control_support, answer unsupported for reasoning_effort when the provider is an engine preset whose record refuses it (unless the model has a documented thinking toggle, TASK-33502).
2. Leave every other provider's answer unchanged.
3. Parametrized tests: refusing presets unsupported; accepting presets and non-engine providers unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
console_generation_control_support answered from PROVIDER_PARAM_MAP only, and ENGINE_PROVIDER_PARAM_MAP maps reasoning_effort for every engine preset, so the Console offered the control (as "support not verified") for presets whose record refuses it -- and every send with a level then failed locally. A new _engine_reasoning_effort_support helper (Chat/console_provider_support.py) returns "unsupported" for an engine preset with reasoning_effort=False (unless the model has a thinking toggle, TASK-33502) and None for everything else, so other providers keep their existing answer. Tests: parametrized rows for nvidia/fireworks/together/cerebras, a registry-wide sweep of every refusing preset, and a check that accepting presets are never hidden (Tests/Chat/test_console_provider_support.py). Mutation (guard removed) turns 6 tests red.
<!-- SECTION:NOTES:END -->
