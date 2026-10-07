---
id: TASK-33511
title: >-
  Readiness and first-run look for the wrong env var when a preset's settings
  table is absent
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 20:32'
updated_date: '2026-09-29 20:40'
labels:
  - providers
  - settings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a provider's api_settings table has no api_key_env_var (an existing config.toml predating the preset, or a partial first-run config), readiness and the first-run wizard derive <KEY>_API_KEY, while the engine reads the record's documented variable. Presets whose documented name differs (Vercel AI_GATEWAY_API_KEY, Ollama Cloud OLLAMA_API_KEY, BytePlus ARK_API_KEY, Azure AZURE_OPENAI_API_KEY, Cloudflare CLOUDFLARE_API_TOKEN, OpenCode Zen OPENCODE_API_KEY) show as unconfigured although the key is exported, and Console blocks the send. Found by Qodo on PR #2916.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With only the documented env var set and no settings table, readiness reports every engine preset as configured
- [x] #2 The first-run wizard detects the same variable
- [x] #3 Non-engine providers keep their current conventional names
- [x] #4 A test fails if a preset's documented variable and the fallback disagree
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. default_api_key_env_var returns an engine preset's documented api_key_env_var from the registry.
2. normalize_provider_config_key maps presets' display config keys to their registry keys (OllamaCloud -> ollama_cloud, OpenCodeZen -> opencode_zen).
3. Sweep tests: readiness and first-run detection with only the documented env var and no settings table.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
default_api_key_env_var derived <KEY>_API_KEY with only three aliases, so with no [api_settings] table (every existing config.toml predates the newer presets) readiness and first-run detection looked for VERCEL_API_KEY, OLLAMA_CLOUD_API_KEY, BYTEPLUS_API_KEY, AZURE_API_KEY, CLOUDFLARE_API_KEY, OPENCODE_ZEN_API_KEY while the engine reads AI_GATEWAY_API_KEY, OLLAMA_API_KEY, ARK_API_KEY, AZURE_OPENAI_API_KEY, CLOUDFLARE_API_TOKEN, OPENCODE_API_KEY. It now returns the registry record's api_key_env_var for engine presets (other providers unchanged). Writing the sweep test exposed a second bug: normalize_provider_config_key lowercased "OllamaCloud"/"OpenCodeZen" to ollamacloud/opencodezen, so readiness by config key answered "Unknown provider"; config.py now maps engine presets' display config keys to their registry keys (identity for every other preset; non-engine mappings untouched). Tests: readiness and read_provider_secret_presence over every engine preset with only the documented env var set. Mutations turn 12 / 2 tests red.
<!-- SECTION:NOTES:END -->
