---
id: TASK-41145
title: "MLX-LM's Settings endpoint field writes api_url, which the MLX adapter never reads, and the shipped config carries an unused api_url placeholder"
status: To Do
created_date: 2026-11-15 06:30
labels:
- follow-up
- settings
- providers
- config
priority: low
references:
- tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Config_Files/default_config.toml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top d4e8d51ca8 (endpoint stream b3c75c8234, on dev ecad404bd4); line numbers are that tree's.

**Why.** A user who sets the MLX-LM endpoint in Settings ▸ Providers sees it saved, but chats still go to 127.0.0.1:8080. Review r14ep-rev4 (C-1) made Console check and pin the adapter's own URL; the Settings field and the adapter still disagree, and changing the adapter needs a config migration.

**Evidence.**
- `settings_screen.py:1686-1700`: `API_URL_PROVIDER_KEYS` includes `local_mlx_lm`, so its Endpoint row writes `[api_settings.local_mlx_lm] api_url`.
- `chat_with_mlx_lm` (`LLM_API_Calls_Local.py:2522-`) uses only the `api_base_url`/`api_url` arguments; otherwise it builds `http://{host}:{port}/v1` from the section's `host` and `port` (defaults 127.0.0.1 and 8080, `:2557-2564`). It never reads the section's `api_url`.
- `Config_Files/default_config.toml:1429-1430` ships `[api_settings.local_mlx_lm] api_url = "http://localhost:5678/v1/chat/completions"`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings' MLX-LM endpoint and the adapter use one setting: either the adapter reads the saved endpoint (with a config migration for existing host/port values) or Settings edits host and port; the notes record which and why
- [ ] #2 The shipped default config matches the setting the adapter reads
- [ ] #3 A test saves the endpoint through Settings and checks the URL the adapter calls, red before the change
<!-- AC:END -->
