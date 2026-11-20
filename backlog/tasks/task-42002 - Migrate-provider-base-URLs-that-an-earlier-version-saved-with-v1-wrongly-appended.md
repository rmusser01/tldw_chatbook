---
id: TASK-42002
title: "Migrate provider base URLs that an earlier version saved with /v1 wrongly appended"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 15:58
labels:
- follow-up
- providers
- settings
priority: medium
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/config.py
- Docs/User_Guide/settings.md
- Docs/Architecture/llm-providers.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41704 (round 17). Before TASK-41704, Settings appended `/v1` to a first-party cloud base with no `v1`, so configs may hold known-broken values such as Google `.../v1beta/v1` or `https://api.cohere.com/v1`; TASK-41704 only stops new saves. Owner decision, 2026-11-19: "Migrate on load": when the app loads one of these known-broken shapes for a first-party provider, it rewrites it to the documented base. This changes those configs' project-instruction keys once.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On the provider's own host only, these saved shapes are rewritten to the documented base when config loads, and the rewrite is saved once: Google `…/v1beta/v1` to `…/v1beta`; Cohere `https://api.cohere.com/v1` to the bare host; Groq, OpenRouter and QwenCloud `<own bare host>/v1` to `/openai/v1`, `/api/v1` and `/compatible-mode/v1`
- [x] #2 A value that is not one of those shapes is never rewritten, including OpenAI's correct `<host>/v1`, any foreign or gateway host, and any custom base. Google's `https://generativelanguage.googleapis.com/v1` is kept too: v1 is a real Gemini API version, and whether the adapter's requests need v1beta cannot be checked without contacting Google
- [x] #3 The rewrite is logged without the URL's credentials or query, and the notes list the shapes covered
- [x] #4 Tests cover each shape and a non-matching control, red before the change
- [x] #5 `Docs/User_Guide/settings.md` (the paragraph that says such values stay as they are) and `Docs/Architecture/llm-providers.md` ("it is not rewritten") describe the migration
<!-- AC:END -->

## Implementation Plan

1. Contract (`provider_endpoint_contract.py`) owns the rule: a table of the shapes an earlier Settings saved with `/v1` appended (Google `v1beta/v1`; Cohere, Groq, OpenRouter, QwenCloud `v1`), read on the provider's own host (the registry default's) as its documented path, and a public `v1_appended_documented_base(provider, value)` returning the documented base or None.
2. Config load: register it as config migration 2 in the existing `_CONFIG_MIGRATIONS` framework (in-memory on every load, persisted once at boot by `migrate_config_file_if_needed`). The migration reads only `api_settings.<key>.api_base_url` (what the five adapters read) and imports the contract only for a value on a provider's own host that is not its default (ADR-097). Log provider and documented base only.
3. `migrate_config_file_if_needed` rewrites the file only when a migration changed content, not for a stamp alone, so configs without these shapes are not rewritten.
4. Tests in `Tests/Chat/test_local_server_discovery.py` (lane-run): each shape migrates in memory and on disk once; controls (OpenAI `/v1`, Google `/v1`, gateway hosts, a custom base, DeepSeek `/v1`, a query/credential value) are never rewritten; red before.
5. Docs: User Guide settings.md and llm-providers.md describe the migration.

## Implementation Notes

One rule, owned by the endpoint contract, serves the load migration and every save. `Provider_Contract/provider_endpoint_contract.py` gains `_V1_APPENDED_PATHS`. On the provider's own host (the registry default's host), each of these saved paths reads as the documented path, before any other path rule runs. The public `v1_appended_documented_base(provider, value)` returns that base, or None for any other value and for any value the contract refuses (user info, a query or a fragment).

**Shapes covered (AC#3):** on the provider's own host only:
- Google `https://generativelanguage.googleapis.com/v1beta/v1` -> `.../v1beta`
- Cohere `https://api.cohere.com/v1` -> `https://api.cohere.com`
- Groq `https://api.groq.com/v1` -> `.../openai/v1`
- OpenRouter `https://openrouter.ai/v1` -> `.../api/v1`
- QwenCloud `https://dashscope-intl.aliyuncs.com/v1` -> `.../compatible-mode/v1`

Nothing else is rewritten: OpenAI's `/v1`, Google's `/v1`, DeepSeek's `/v1`, gateway and foreign hosts (QwenCloud's China host included), presets and custom endpoints.

**Load and save once.** `config.py` registers `_migrate_v1_appended_provider_bases` as config migration 2 in the existing TASK-26040 framework (`_CURRENT_CONFIG_SCHEMA_VERSION` 1 -> 2; the template's stamp is 2 too). Every load runs it in memory, and the boot step `migrate_config_file_if_needed` (app_entry) persists it once, with the framework's backup. The migration reads only `api_settings.<key>.api_base_url`, which the five adapters read. It imports the contract only for a value on a provider's own host that is not the registry default (ADR-097: no import for an ordinary config). It logs the provider and the new documented base. A value the contract refuses is never rewritten, so no credentials or query reach the log.
- Decision: `migrate_config_file_if_needed` now rewrites the file only when a migration changed content, not for the stamp alone. Without that, registering any migration would rewrite every existing config once at boot, which strips its comments and makes a backup. The framework's own rule already said a bare stamp is not worth a rewrite.
- Known gap (pre-existing, framework-wide): `_install_bootstrap_cache_from_raw` and `_atomic_config_values_from_raw` do not run migrations. If the boot persist never ran, the in-memory view after a settings write is the unmigrated raw file until the next full load. Listed as a follow-up.
- Rewritten on purpose: `test_a_project_instruction_key_saved_before_the_v1_fix_still_resolves` (TASK-41704 AC#2) now expects a save of Google's and Cohere's own-host values to write the documented base. The owner's decision changes their keys once. Its gateway and DeepSeek rows are unchanged. The template's sha pin in `Tests/test_config_load_settings_table_guard.py` was re-pinned for the stamp.

**Tests** (`Tests/Chat/test_local_server_discovery.py`, lane-run):
- `test_config_load_migrates_a_base_saved_with_v1_appended_once`: 15 nodes. For each of the 5 shapes: real loader on a scratch v1 config, the loaded value is the documented base, the boot step writes it once (a second call writes nothing), and the log names the provider and the base. For each of 10 controls: the value is loaded unchanged, the file is byte for byte untouched, and nothing is logged. The controls are OpenAI `/v1`, Google `/v1`, a Google gateway `.../v1beta/v1`, a Groq gateway `/v1`, QwenCloud's China host, DeepSeek `/v1`, a custom `api_url`, a preset's own host `/v1`, and own-host shapes with a query or user info. No log line holds the secret, and a save keeps every accepted control.
- `test_a_base_saved_with_v1_appended_saves_its_documented_base`: 5 nodes (TASK-42019).
- Red with the three sources at d85e90bc9e: 11 failed / 10 passed. All 5 save nodes failed (saved `/v1`), all 5 shape load nodes failed (value unchanged), and so did the rewritten key test. The 10 controls pass at base (an invariant). Negative controls: persisting stamp-only migrations made 10 of 10 controls red (file rewritten). Dropping the contract's own-host check made the Groq-gateway and QwenCloud-China controls red on the save. Green after reverting each.

**Covering run** (`-n 4`, load 9-10): the migration, contract, save, setup-facts, discovery, first-run endpoint, registry, import-closure, private-bootstrap and QwenCloud adapter files, 18 in all: 2244 passed and 1 failed. The failure was the template sha pin, re-pinned (file: 12 passed). The module-size ratchet passed (serial, 209): `config.py` 8837 -> 8871 (+34), with a dated row comment.

**Docs:** `Docs/User_Guide/settings.md` (the paragraph that said such values stay) and `Docs/Architecture/llm-providers.md` ("it is not rewritten") now describe the migration.

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `tldw_chatbook/config.py`, `tldw_chatbook/Config_Files/default_config.toml`, `Tests/Chat/test_local_server_discovery.py`, `Tests/test_config_load_settings_table_guard.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/settings.md`, `Docs/Architecture/llm-providers.md`.
