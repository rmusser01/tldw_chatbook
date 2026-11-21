---
id: TASK-42153
title: "Decide whether the /v1 base migration also covers QwenCloud's China host and Cohere's old api.cohere.ai"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 21:10
labels:
- follow-up
- providers
- settings
- decision
priority: low
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/config.py
- tldw_chatbook/provider_registry.py
- Docs/User_Guide/settings.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the providers stream's TASK-42002 (round 18; owner note). TASK-42002 rewrites bases an earlier Settings saved with `/v1` appended, and its AC#1 limits the rewrite to the provider's own host: on the round-18 providers branch, `_documented_own_path` in `Provider_Contract/provider_endpoint_contract.py` returns nothing unless the saved host equals the host of the registry's `default_base_url`. Two saved shapes are therefore left as they are:
- QwenCloud's China host, `https://dashscope.aliyuncs.com/v1`. The registry default is `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (`tldw_chatbook/provider_registry.py:1649`), and the China host's compatible base also ends in `/compatible-mode/v1`.
- Cohere's old host, `https://api.cohere.ai/v1`. The default is `https://api.cohere.com` (`provider_registry.py:1582`); with `/v1` kept, the adapter posts to `/v1/v2/chat`, the shape TASK-42019 covers for `api.cohere.com`.

Whether to extend the migration to these hosts is an owner decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision is recorded: whether the migration also rewrites these two shapes
- [x] #2 If extended: each shape is rewritten to its documented base when config loads and saved once, any other host is still left alone, and tests for both are red before the change
- [x] #3 `Docs/User_Guide/settings.md` lists the shapes the migration covers
<!-- AC:END -->

## Implementation Plan

1. Record the owner's decision (2026-11-21, "153 - yes") in the notes.
2. Contract: a `_V1_APPENDED_OTHER_HOSTS` table (QwenCloud `dashscope.aliyuncs.com`, Cohere `api.cohere.ai`) beside `_V1_APPENDED_PATHS`; `_read_v1_appended_path` reads a shape on one of those hosts as on the provider's own host, so a save writes the documented path on the host typed.
3. Config: a frozen kernel copy of the same table and config migration 3 (schema 2 -> 3, template stamp 3), reusing migration 2's matcher with those hosts; no new import, no allowlist row. A new number, because a config already at schema 2 (stamped by migration 2's boot write or created from the v2 template) never re-runs migration 2.
4. Tests in `Tests/Chat/test_local_server_discovery.py` (lane-run), red first: save and load rows for both hosts with their tails, a v2 config migrated by migration 3, controls kept, and the parity test run over every migration and the new hosts.
5. Docs: `Docs/User_Guide/settings.md` and `Docs/Architecture/llm-providers.md`; re-pin the template sha; module-size ratchet.

## Implementation Notes

**Owner decision, 2026-11-21 (verbatim): "153 - yes".** The TASK-42002 `/v1`-appended migration and Settings save rule now also cover QwenCloud's China host and Cohere's old host:
- `https://dashscope.aliyuncs.com/v1` -> `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `https://api.cohere.ai/v1` -> `https://api.cohere.ai`

Each also with the tails TASK-42002 drops (`/chat/completions`, `/models`; QwenCloud `/responses`; QwenCloud's `/v1/models` is still refused and kept). The host typed is kept: the documented path goes on that host, as with a typed port. Nothing moves to the registry default's host.

**Contract (save rule).** `Provider_Contract/provider_endpoint_contract.py` gains `_V1_APPENDED_OTHER_HOSTS` (`qwencloud: dashscope.aliyuncs.com`, `cohere: api.cohere.ai`) beside `_V1_APPENDED_PATHS`. `_read_v1_appended_path` reads a shape on that host as on the provider's own host. `v1_appended_documented_base` follows from it.
- The bare-host completion follows too (see "Coordinator correction" below): `_documented_own_path` treats the listed host as the provider's own, so `_read_v1_appended_path` needs no host mapping of its own.

**Config (load migration), migration 3 and a schema bump.** `config.py` keeps a frozen kernel copy of the table (`_V1_APPENDED_OTHER_HOSTS`). Migration 3 is `partial(_migrate_v1_appended_provider_bases, hosts=_V1_APPENDED_OTHER_HOSTS)`: the same matcher, reading the listed host instead of the registry default's. Migration 2 behaves exactly as before (`hosts=None`). `_CURRENT_CONFIG_SCHEMA_VERSION` is 2 -> 3 and the template's stamp is 3 (sha pin re-pinned).
- Why a new number rather than widening migration 2: a config already at schema 2 never runs migration 2 again. That covers a config migration 2's boot write stamped (it held one of the five own-host shapes) and every config created from the v2 template. A v2 config can hold the China shape, because a save of the China `/v1`, or of the bare China host, writes it today. Widening migration 2 would never reach those configs, and it would also change a frozen migration's result.
- The rules TASK-42002 set still hold: migrate on load, saved once by the boot step (`migrate_config_file_if_needed`). It matches only a spelling a save wrote (the listed host, `http`/`https`, an optional canonical port, kept), so user info, a query, a fragment, a gateway or any other host never match. The log line is migration 2's: the provider and the new base only. No new import (`partial` comes from the existing `functools` import), no allowlist row; `Tests/Architecture/test_layer_direction.py` passes.

**Tests** (`Tests/Chat/test_local_server_discovery.py`, lane-run):
- `test_a_base_saved_with_v1_appended_saves_its_documented_base`: 6 new nodes (`qwencloud-other-host`, `cohere-other-host`, each bare and with its two tails), including the adapter posting under the saved base (`.../compatible-mode/v1/responses`, `https://api.cohere.ai/v2/chat`).
- `test_config_load_migrates_a_base_saved_with_v1_appended_once`: 7 new shape rows (both hosts with their tails, and `https://api.cohere.ai:443/v1` keeping its port). The old `qwencloud-china-host` control is now a shape. New controls, each loaded unchanged with the file left byte for byte: another QwenCloud region's host (`dashscope-us`), a Cohere gateway path, and on the new hosts a query, user info and QwenCloud's `/v1/models`.
- `test_a_config_already_at_schema_2_still_migrates_the_added_hosts[qwencloud|cohere]`: a v2 file is rewritten on load and saved once, stamped 3.
- `test_config_migration_2_agrees_with_the_endpoint_contracts_rule`: adds a structural pin (config's `_V1_APPENDED_OTHER_HOSTS` equals the contract's). The migrations now run as a v1 config's load runs them (`migrate_config_forward`, so 2 then 3), over the generated rows for every own host and both added hosts. `test_the_boot_migration_step_takes_no_lock_for_a_config_with_nothing_to_rewrite` gains a v3 row.
- Red at base (dc371f0842 sources, new tests): 16 failed and 50 passed. All 6 save nodes and all 7 load shape nodes failed (`/v1` kept), both v2 nodes failed (`(..., False, None) == (..., True, None)`), and the parity test failed with an AttributeError because config had no table. With the config half only, the parity test was still red (the contract had no table). Negative controls after the change, each reverted: removing Cohere from the contract's table failed the structural pin, and skipping the new host read in `_read_v1_appended_path` failed the parity assertion (`('cohere', 'http://api.cohere.ai/v1', 'http://api.cohere.ai', 'http://api.cohere.ai/v1')`, ...). The controls pass at base, an invariant. After: 66 passed (the TASK-42002/42019/42153 selection, `-n 4`).

**Covering run** (`-n 4`, load 33-58): the 40 test files that name the contract, the migration framework, the schema version, the two hosts, `persisted_endpoint_for_key` or the template (the paid live QwenCloud file excluded), plus `Tests/test_config_hot_reload.py` and `Tests/test_config_persistence_owner.py`: 412 failed, 4720 passed, 1 xfailed, 8 errors. All of those reds were in 8 files: `Tests/UI/test_console_native_chat_flow.py`, `test_settings_qwencloud_api_mode.py`, `test_settings_saved_provider_return.py`, `test_console_session_settings.py`, `test_settings_provider_test_draft.py`, `test_console_endpoint_discovery.py`, and `Tests/Chat/test_cohere_native_tools.py`, `test_console_provider_gateway.py`, `test_console_session_settings.py`. Most were the local `RecoveryRequired: raw_source_selection_changed` trap (323), or stale tests (`SimpleNamespace` has no `get_message_exchanges`, a missing `_console_visible_draft_session_id`). The same 8 files run with dev's sources (dc371f0842) gave 410 failed and 8 errors, the same node set except 2. Both of those passed when re-run at the head: `test_conversation_settings_return_claims_exact_revision_and_restores_mounted_draft` at `-n 2`, and `test_console_original_attempt_delete_clears_parent_and_descendant_previews` serially (it timed out waiting for `#console-delete-receipt-done` at `-n 2` under load 47). Neither touches config migrations.

`config.py` 8901 -> 8913 (+12), with a dated row comment in `Tests/Architecture/test_module_size_ratchet.py`.

**Docs:** `Docs/User_Guide/settings.md` now lists the shapes the migration covers, the two added hosts included, and says the host and any port typed are kept. `Docs/Architecture/llm-providers.md` describes `_V1_APPENDED_OTHER_HOSTS`, migration 3 and why it is a new schema version.

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `tldw_chatbook/config.py`, `tldw_chatbook/Config_Files/default_config.toml`, `Tests/Chat/test_local_server_discovery.py`, `Tests/test_config_load_settings_table_guard.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/settings.md`, `Docs/Architecture/llm-providers.md`.

**Coordinator correction (2026-11-21): the bare-host completion is in scope of "153 - yes".** The first version left `_documented_own_path` (review r17cu-rev2) knowing only the registry default's host. A bare `https://dashscope.aliyuncs.com` (or `.../compatible-mode`) therefore saved as `.../v1` (or as typed), and the next save of that provider rewrote it to `.../compatible-mode/v1`, so the save was not stable.
- `_documented_own_path` now treats the provider's `_V1_APPENDED_OTHER_HOSTS` host as its own. Both rules read it from there, and the host mapping added earlier to `_read_v1_appended_path` is gone. A bare China host, or one with `/compatible-mode`, saves as `https://dashscope.aliyuncs.com/compatible-mode/v1`, and the adapter posts `.../compatible-mode/v1/responses`. `https://api.cohere.ai` already saved bare, because Cohere's bare host is its base on any host, and it still does.
- Test: `test_a_cloud_base_without_v1_saves_a_base_its_adapter_posts_under` (`Tests/Chat/test_local_server_discovery.py`, lane-run). The QwenCloud case gains the China host bare, `/compatible-mode` and `/compatible-mode/` rows; the Cohere case gains `https://api.cohere.ai`. Every row of the test now also asserts that a save is stable: saving what was saved writes it unchanged.
  - Red before the change: `[qwencloud-api_base_url]` failed (`https://dashscope.aliyuncs.com/compatible-mode` saved as typed), and the other 5 cases passed, the new stability assertion included. The Cohere old-host row passed before the change too; it is a pin, not a fix.
  - After: the TASK-41704/42002/42019/42153/42134 selection, 75 passed.
- Config's frozen table and migration 3 are unaffected: a bare host is not a `/v1`-appended shape. The parity test is unchanged and green.
- Docs: `Docs/User_Guide/settings.md` (the own-host completion sentence) and `Docs/Architecture/llm-providers.md`.
