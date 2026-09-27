---
id: TASK-33092
title: Finish the shared settings vocabulary beyond sampling params
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, settings]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-147's Chat/sampling_params.py already unified the sampling keys, bounds, and enums across 13 consumers — the right pattern, half applied. The rest of the settings vocabulary is still declared per surface and has measurably drifted: the endpoint-key tuple has 5 keys in the Settings hub (PROVIDER_ENDPOINT_KEYS) versus 8 in console_settings_defaults.py (missing api_endpoint, router_base_url, huggingface_router_base_url on the hub side), provider-capability maps exist three times (settings_screen, console_settings_defaults, model_capabilities predicates), and coercion helpers have 30+ private copies repo-wide despite canonical ones in config.py. Extend the sampling_params pattern to the remaining vocabulary so every surface reads one schema.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The endpoint-key tuple is defined once and consumed by both the Settings hub and Console defaults.
- [ ] #2 Provider capability maps are sourced from one table.
- [ ] #3 The Settings hub sees all eight endpoint config keys — the measured drift is eliminated.
- [ ] #4 Per-family curated enum lists and model_capabilities version-floor predicates either consume the shared table or document why they cannot.
- [ ] #5 Targeted settings tests pass.
<!-- AC:END -->
