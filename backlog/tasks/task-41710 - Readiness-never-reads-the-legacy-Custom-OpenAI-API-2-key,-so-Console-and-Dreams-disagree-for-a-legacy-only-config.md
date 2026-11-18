---
id: TASK-41710
title: "Readiness never reads the legacy Custom OpenAI API 2 key, so Console and Dreams disagree for a legacy-only config"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- providers
- config
priority: medium
references:
- tldw_chatbook/config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-41523 (round 16). With only the legacy `[API] custom_openai2_api_key` set, Settings shows no key and Console sends keyless (Console passes readiness's key), while Dreams and import analysis send the legacy key. Nine providers already have their legacy key bridged into `api_settings.<provider>.api_key` at load time (`_LEGACY_PROVIDER_API_KEY_BRIDGE`, `_normalize_legacy_provider_api_key`); Custom 2 does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A legacy-only Custom 2 key is seen by readiness, Settings and Console the same way Dreams and import analysis see it, or is retired with a named migration
- [ ] #2 A test pins a legacy-only config across readiness and a non-Console send, red before the change
- [ ] #3 The User Guide and `Docs/Architecture/llm-providers.md` legacy-key sentences match the new behaviour
<!-- AC:END -->
