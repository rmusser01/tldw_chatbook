---
id: TASK-42002
title: "Migrate provider base URLs that an earlier version saved with /v1 wrongly appended"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-21 17:03
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

**Load and save once.** `config.py` registers `_migrate_v1_appended_provider_bases` as config migration 2 in the existing TASK-26040 framework (`_CURRENT_CONFIG_SCHEMA_VERSION` 1 -> 2; the template's stamp is 2 too). Every load runs it in memory, and the boot step `migrate_config_file_if_needed` (app_entry) persists it once, with the framework's backup. The migration reads only `api_settings.<key>.api_base_url`, which the five adapters read. It hands every `api_base_url` to the contract (already in the boot import set; review r18-rev1 dropped the host pre-filter). It logs the provider and the new documented base. A value the contract refuses is never rewritten, so no credentials or query reach the log.
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

**Review r18-rev1 fixes:**
- Boot cost: the boot step took the config write lock and parsed the whole file on every launch (~35 ms under load), and a v1 file with nothing to rewrite is never stamped, so it paid that forever. The load now records, per config path, the file stamp at which migration found nothing beyond the version stamp to persist (`_CONFIG_MIGRATION_SETTLED`, shared `_changed_beyond_stamp`); `migrate_config_file_if_needed` returns before the lock while the file still has that stamp, and keeps the re-read under the lock for a real write (a file changed since the load, or never loaded, takes the old path). Measured after: 0.0055 ms per call for a v2 config. The `_has_pending_config_migration` docstring ("rides the next natural save") and the app_entry comment are corrected.
  - Test: `test_the_boot_migration_step_takes_no_lock_for_a_config_with_nothing_to_rewrite` (v2 and a v1 no-change control; `_config_write_lock` patched to raise). Red 2/2 at the previous commit (`AssertionError: took the config write lock`), green after.
- Shape plus a pasted tail: `_read_v1_appended_path` matches the shape as a prefix followed by a tail the reading drops (none, `chat/completions`, `models`, and `responses` for QwenCloud) and returns the documented path plus that tail, so QwenCloud's `/v1/models` still reaches TASK-42021's `/models` refusal. Decision: the load migration rewrites a stored shape plus tail too, to exactly what a save writes (`v1_appended_documented_base` now returns the save's base, `chat_url` minus `/chat/completions`); QwenCloud's `/v1/models` is refused by both and kept.
  - Tests: 10 new save rows in `test_a_base_saved_with_v1_appended_saves_its_documented_base` and 10 new load rows (plus a `qwencloud-v1-models` control) in `test_config_load_migrates_a_base_saved_with_v1_appended_once`; 20 red at the previous commit (the save or load kept `.../v1` plus the tail), green after.
- `_migrate_v1_appended_provider_bases` drops its default-equality and raw-hostname pre-filter; the contract decides alone. Boot budget re-run: boot-import count unchanged (687/688).
- `config.py` 8875 -> 8884 (+9), dated ratchet comment. Docs: `Docs/Architecture/llm-providers.md` and `Docs/User_Guide/settings.md`.

**Review r18-rev2:** `config.py`'s unused `RECORDS_BY_KEY` import (only the removed pre-filter used it) is gone; ratchet 8884 -> 8883. `test_the_boot_migration_step_writes_a_file_changed_after_the_load_settled_it` pins that a file changed after the load settled it (a different size, now holding Groq's `<host>/v1`) takes the locked path and is rewritten; red with the stamp comparison temporarily ignored (`assert None is not None`), green after.

**Round-18 hotfix (2026-11-21):** migration 2's lazy import of `Provider_Contract.provider_endpoint_contract` from `config.py` was a kernel -> domain edge (`Docs/Architecture/layer-map.md`), so `Tests/Architecture/test_layer_direction.py::test_no_new_upward_import` failed on dev f864f00762 and every PR's required check. Coordinator decision: a config migration is a frozen historical fix and does not call live domain code.
- `config.py` keeps its own table, `_V1_APPENDED_SHAPES` (provider -> the path saved with `/v1` appended), and the migration matches only a spelling a save wrote: the provider's registry default host (`RECORDS_BY_KEY`, already imported from the kernel `provider_registry`; stdlib `urlsplit`) with an optional port (review rev1, below), `http` or `https`, with no tail, a pasted `/chat/completions`, or `/models` (QwenCloud: `/responses`; its `/v1/models` is kept, as the contract refuses it). It rewrites to the registry default base on the value's scheme and netloc, which is what a save writes. User info, a query, a fragment or any other host never match, so they are left untouched; a leading-zero port, an upper-case host and a trailing slash are left too (the contract reads them, but no save wrote them). The log line is unchanged: provider and new base only. No allowlist row was added.
- The documented base is built with `SplitResult._replace(...).geturl()`, not an f-string over `.path`: the production diagnostic inventory reads `.path` as path-shaped and flagged the log call as a new path-privacy candidate.
- Parity test: `test_config_migration_2_agrees_with_the_endpoint_contracts_rule` (`Tests/Chat/test_local_server_discovery.py`, lane-run) runs the migration and the contract's `v1_appended_documented_base` over every `_V1_APPENDED_LOAD_ROWS` value (the 5 shapes, their 10 tails, and the 11 controls including QwenCloud `/v1/models`), each also as `http://`, and asserts the same result. It is green at base (an invariant dev keeps by construction), so red was shown under temporary source edits: removing Groq from the contract's `_V1_APPENDED_PATHS` failed it (`('groq', 'http://api.groq.com/v1')`), and dropping the kernel's `/models` tail failed it (`('cohere', 'http://api.cohere.com/v1/models')`); both reverted. The existing TASK-42002/42019 tests are unchanged and green (46 passed, `-n 4`).
- The contract's comments and `Docs/Architecture/llm-providers.md` now say config keeps a frozen copy pinned by the parity test. Boot import weight unchanged (687/688). `config.py` 8883 -> 8895 (+12), dated ratchet comment.
- Same hotfix, second red: `Tests/Architecture/test_app_string_lookup_census.py::test_app_string_lookups_match_their_pin[UI]` measured 1315 against a pin of 1316. TASK-42015 (bd343b444a) deleted `ChatScreen._record_console_first_send`, and with it `getattr(self.app_instance, "app_config", None)`; the census over the tree with that one file restored to bd343b444a^ measures 1316. The pin in `Tests/Architecture/app_lookup_census.tsv` is lowered to 1315.

**Review rev1 of the round-18 hotfix (2026-11-21):**
- An explicit port was no longer migrated, though the save kept a typed port (`_normalized_netloc(canonical_host, port)`), so `https://api.groq.com:443/v1` and the like were really saved; that regressed dev's previous migration for all 5 providers. The migration now matches the host with an optional canonical port (1-65535, no leading zero, at most 5 digits, so `int` never sees a long string) and keeps the netloc in the result. The log line still carries only the provider and the new base.
- The parity test is widened: a structural pin (`_V1_APPENDED_SHAPES` equals the contract's `_V1_APPENDED_PATHS` joined), and 180 generated rows (5 shapes x `https`/`http` x no port, the scheme's default port, 8443 x no tail, `/chat/completions`, `/models`, `/responses`, `/embeddings`, `/chat`) on top of the load rows; only `https://` load rows get an `http://` twin (the custom control became `http://http://...`). Red at 985ceb88ae: 60 of the 180 generated rows disagreed, every one with a port; green after.
- `v1_appended_documented_base`'s docstring now says it is the parity oracle and that a save goes through `_read_v1_appended_path`. `Docs/Architecture/llm-providers.md` and `Docs/User_Guide/settings.md` say a typed port is kept and that a hand-edited spelling is rewritten on its next save, not on load. `config.py` 8895 -> 8901 (+6), dated ratchet comment.
