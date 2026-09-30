---
id: TASK-33510
title: Settings cannot save a key or endpoint for any engine preset except Databricks
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 20:31'
updated_date: '2026-09-29 20:40'
labels:
  - providers
  - settings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provider setup persistence validates the provider against a hand-kept allowlist that never gained the engine presets (Together, Fireworks, Cerebras, the inference clouds, the model makers, the gateways and the TASK-33505..33509 presets). Saving a changed key or endpoint for any of them in Settings fails with 'Provider settings are invalid', and the first-run wizard cannot persist them either. Found by Qodo on PR #2916.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings and first-run setup can save the key and endpoint of every engine preset
- [x] #2 Each preset's display name resolves to its canonical provider key
- [x] #3 Unknown or malformed provider names are still rejected
- [x] #4 A test fails if a future engine preset is missing from setup persistence
- [x] #5 A preset's documented base URL, or a bare host for a per-account preset, is saved exactly as the engine needs it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Derive setup persistence's engine-preset keys and display-name aliases from the registry.
2. Teach the endpoint contract engine presets' own API bases (no forced /v1; bare host -> record suffix or documented path).
3. Sweep tests over every preset: ownership, exact round-trip of the documented URL, bare hosts and pasted chat URLs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two layers blocked saving an engine preset in Settings/first-run. (1) provider_setup_persistence._CANONICAL_PROVIDER_KEYS was a hand list that never gained the presets (only Databricks), so ProviderSetupDraft raised "Provider is not supported" and Settings showed "Provider settings are invalid". Now the engine presets (minus the custom-hosted execution key) and their display config keys are derived from provider_registry. (2) Behind it, provider_endpoint_contract modeled every URL as <root>/v1/...: DeepInfra's /v1/openai was rejected ("ambiguous API suffixes"), BytePlus/Kilo/Qianfan saved with a bogus /v1 appended, and an Azure/Databricks bare host saved as /v1 instead of /openai/v1. _engine_preset_resolution treats the entered path as the API base (dropping a pasted /chat/completions or /models); a bare host takes the record suffix, else the documented path on the preset's own host, else /v1. Every documented URL now round-trips exactly (tests over all 32 URL-shipping presets + bare-host and pasted-chat-URL cases). Mutations (derivation removed, contract branch removed) turn 95 / 7 tests red. Found via Qodo on #2916, which reported only the five new keys.
<!-- SECTION:NOTES:END -->
