---
id: TASK-41145
title: "MLX-LM's Settings endpoint field writes api_url, which the MLX adapter never reads, and the shipped config carries an unused api_url placeholder"
status: In Progress
assignee:
- '@claude'
updated_date: '2026-11-15 19:12'
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

## Implementation Plan

1. Map every writer and reader of the MLX-LM endpoint (Settings, first-run setup, Chat settings Make default, adapter, Console pinning, shipped template).
2. If agreeing needs a config migration or a choice of which side wins, stop and report NEEDS_CONTEXT with the options and their effect on existing user files (brief).

## Implementation Notes

**NEEDS_CONTEXT — not implemented.** Making the field and the adapter agree needs a choice of which side wins, and one of the two choices needs a config migration; the round-15 brief says to stop and report both, and not to change the round-14 Console MLX pinning without that decision. No criterion is met yet.

Findings (round-15 tree, base cc6006fcf5):
- Writers of `[api_settings.local_mlx_lm] api_url`: Settings ▸ Providers' Endpoint row (`settings_screen.API_URL_PROVIDER_KEYS` -> `_provider_endpoint_setting_key`), first-run setup and Chat settings' Make default (both through `provider_setup_persistence.provider_endpoint_key`, which is `api_url` for `local_mlx_lm`/`mlx_lm`). Settings also *shows* that value as the endpoint.
- Reader: `chat_with_mlx_lm` uses only a request's `api_base_url`/`api_url`; otherwise `http://{host}:{port}/v1` from `[api_settings.<local_mlx_lm|mlx_lm>]` (default 127.0.0.1:8080). Console's send and readiness, approval summaries and the sub-agent freeze check and pin that built URL (round 14, `ADAPTER_ONLY_ENDPOINT_KEYS`); TASK-41131's transport check checks the same URL.
- The template ships `api_url = "http://localhost:5678/v1/chat/completions"` and no host/port, and first run writes the whole template into config.toml, so profiles created from it carry that placeholder verbatim. Every such user sees localhost:5678 in Settings while chats go to 127.0.0.1:8080.

Option A — the adapter reads `api_url` (Settings, first-run setup and Make default win). The adapter and Console's pin read `api_url` first and host/port only when it is unset; the template ships `api_url = "http://127.0.0.1:8080/v1"`. Needs a migration: read as-is, the placeholder already in existing files would move every MLX user from 127.0.0.1:8080 to localhost:5678. The migration would drop (or rewrite to `http://<host>:<port>/v1`) an `api_url` equal to the exact shipped placeholder and keep any other value. A user who once saved a real URL in Settings (which silently did nothing) starts sending to it at upgrade: their stated intent, but a destination change. Changes the round-14 Console MLX pinning.

Option B — Settings edits host and port (the adapter wins). The MLX endpoint field in Settings, first-run setup and Make default parses the URL into host and port, writes those, and refuses what they cannot express (https, a path other than /v1, user info); it shows `http://host:port/v1`. The template drops the unused `api_url` (optionally ships `host`/`port`). No send changes for anyone and no migration for sends; existing files keep a dead `api_url` unless a cleanup removes it. A user who saved an `api_url` in Settings sees the field show 127.0.0.1:8080 and must re-enter it; an https or path-bearing MLX URL cannot be expressed. Console's pinning is unchanged.

Related, found while doing TASK-41143: Custom OpenAI API 2 has the same split. Settings, first-run setup and Make default write `[api_settings.custom_2] api_url` (Console reads and pins it), but `chat_with_custom_openai_2`, unpinned, reads the top-level `[custom_openai_api_2] api_ip`, which the template does not ship, so unpinned callers (Dreams, ingest analysis, Summarization) fail with "API URL (api_ip) required" or use a stale value.
