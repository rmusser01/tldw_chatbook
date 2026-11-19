---
id: TASK-41704
title: "Saving a native cloud provider's own base URL in Settings appends /v1 and breaks its requests"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 05:33
updated_date: 2026-11-19 17:24
labels:
- follow-up
- providers
- settings
priority: high
references:
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
dependencies:
- TASK-41700
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r16cu-rev2 of TASK-41700. `_persisted_endpoint_for_owner` reads a cloud provider's path that has no `v1` segment through the endpoint contract's generic reading, which appends `/v1`. Saving Google's own `https://generativelanguage.googleapis.com/v1beta` writes `.../v1beta/v1`, and the adapter then posts to `.../v1beta/v1/models/m:generateContent` (404). Cohere's `https://api.cohere.com` is saved as `.../v1`, and the adapter posts to `/v1/v2/chat` (404).

The same happens to a gateway base without a `v1` (`https://litellm.corp/gemini/v1beta`), and to DeepSeek's origin, which DeepSeek tolerates. It triggers on an explicit Settings entry, on "Save endpoint & use model", and on an existing configured value.

Confirmed to predate TASK-41522 and TASK-41700 (review r16cu-rev2, identical at 45b1949ca3). Fixing it changes the canonical identity of values accepted today, so project-instruction keys must be handled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Saving each native cloud provider's own documented base, or a gateway base without `v1`, writes it unchanged, and the adapter posts to the right URL
- [x] #2 Project-instruction keys for values saved before the fix still resolve, or are migrated, with a test
- [x] #3 Tests pin Google, Cohere and one gateway base, red before the fix
- [x] #4 The User Guide's known-gap sentence about the `/v1` append is removed
<!-- AC:END -->

## Implementation Plan

1. Contract (`provider_endpoint_contract.py`), one reading for every caller: a first-party cloud key's path with no `v1` segment is its API base (the relaxed reading TASK-41700 added for a mid-path `v1`), not an origin the generic rule appends `/v1` to. A bare host stays the base for the providers whose documented base is a bare host (Cohere, DeepSeek); for the others a bare host still gets `/v1`, as today. Values whose `v1` is last are untouched.
2. Settings' `_api_base_from_chat_url` writes a bare-host base without a trailing `/`.
3. Project-instruction keys: they hash the Console resolution's `base_url`, the stored value, not the contract's identity; pin that a value saved before the fix (`.../v1beta/v1`, `https://api.cohere.com/v1`) still resolves Ready with the same base, so its key is unchanged, and an unedited re-save writes it back unchanged.
4. Tests in `Tests/Chat/test_local_server_discovery.py` (lane-run): Google, Cohere, DeepSeek and a gateway base without `v1` save unchanged and the adapter posts to the right URL; red at base.
5. Docs: remove the User Guide's known-gap sentence; update `Docs/Architecture/llm-providers.md`.

## Implementation Notes

Fixed in the endpoint contract, which Settings' **s**, Console's "Save endpoint & use model", first-run setup and Make default all save through. In `Provider_Contract/provider_endpoint_contract.py`, a first-party cloud key's path with no `v1` segment is no longer an origin the generic rule appends `/v1` to: it goes through the relaxed reading TASK-41700 added for a mid-path `v1` (the path is the API base, a pasted `/chat/completions` or `/models` is dropped, a route is refused). A bare host is the base for the providers whose documented base is a bare host (Cohere on any host, DeepSeek on its own host; see the review r17cu-rev1 round below); any other cloud provider's bare host still reads as `<host>/v1`, as before. `_engine_preset_resolution` gained `bare_host_base`, and `provider_setup_persistence._api_base_from_chat_url` writes a bare host without a trailing `/`. A path whose `v1` is last resolves exactly as before. Gateway bases (owner decision "Allow gateway paths") and every host and shape refusal are unchanged.

Of every shipped default (`_BUILTIN_PROVIDER_ENDPOINTS` and the registry's default bases), only three save differently, all as intended: Google `.../v1beta` (was `.../v1beta/v1`), Cohere `https://api.cohere.com` (was `.../v1`) and DeepSeek `https://api.deepseek.com` (was `.../v1`); compared base vs head with a probe through the wrapper.

**Project-instruction keys (AC#2).** They hash the Console send resolution's `base_url`, which for a cloud provider is the stored value, not the contract's canonical identity, so they are unaffected by the reading change. A value an earlier version saved with `/v1` appended (`.../v1beta/v1`, `https://api.cohere.com/v1`) still reads as before (its `v1` is last): Console sends to it as stored, its notice key is the one consented under, and an unedited re-save writes it back unchanged. It is not migrated (it stays broken until the user re-enters the address; the User Guide says so). The contract's canonical connection identity does change for the no-`v1` values, but every holder of it (test evidence, the first-run discovery key, Settings' CAS identity, Console's probe identity) is process-local or recomputed, and continuation checkpoints compare the raw base first.

**Tests** (`Tests/Chat/test_local_server_discovery.py`, lane-run):
- `test_a_cloud_base_without_v1_saves_unchanged_and_posts_there` (Google, Cohere, DeepSeek; renamed and extended in the review round below): Google's own base and `https://litellm.corp/gemini/v1beta`, Cohere's own base and `https://litellm.corp/cohere`, DeepSeek's own base save unchanged, and the adapter, reading the saved value, posts `<base>/models/m:generateContent`, `<base>/v2/chat`, `<base>/chat/completions`. Red with c176f553b3's sources: 3 of 3 on `saved == base` (`.../v1beta/v1`, `https://api.cohere.com/v1`, `https://api.deepseek.com/v1`).
- `test_a_project_instruction_key_saved_before_the_v1_fix_still_resolves`: for four values saved with `/v1` appended, Console is Ready and sends to the stored value, the notice key equals the digest computed at c176f553b3 through the same resolution, and an unedited re-save writes it back. Green at base (an invariant); red under a temporary edit that made Console pin a cloud provider's contract-resolved endpoint (key `7e271d91...` instead of `a9aee8f3...`), green after reverting it.
- Serial cost of the 18 new nodes in this file: 3.1 s (load 17-21).

**Covering run** (150 files: every test file that calls the endpoint contract, the Settings setup persistence or card, the Cohere/Google/QwenCloud adapters or names their URLs; `-n 4`, load 19-67): 663 failed / 9832 passed / 1 error. Re-run of the 663 failing ids with this change's two source files reverted to c176f553b3 (same command): 658 fail the same way (the known `RecoveryRequired: raw_source_selection_changed` family: 289 in `test_console_native_chat_flow.py`, 87 in `test_console_session_settings.py`, the Cohere/Google native-tools files, the plain Settings UI files), and the error is `test_sensitive_llm_logging.py::test_auxiliary_pins_default_openai_endpoint_before_config_changes`, red at base too. The 5 that passed at base in that batch, run alone with the same serial command: 2 fail at head and at base alike (`test_google_native_tools.py::test_list_content_with_tool_calls_keeps_text_parts`, RecoveryRequired; `test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh`), 3 pass at both (order-dependent). `Tests/Chat/test_local_server_discovery.py` whole file 1046 passed; save-scope, connect-key-rows and the QwenCloud contract/adapter files 193 passed; `Tests/CI` 1811 passed; architecture size and identity ratchets (serial) 323 passed; `Tests/lint/ruff_gate.py` OK (the 2 ruff findings left in the test file are at base); preflight passed. Neither source module has a size-ratchet row.

**Docs.** User Guide `settings.md`: the known-gap sentence and its config-file workaround are gone; the paragraph now says a no-`v1` address saves as typed (Google, a gateway, Cohere's and DeepSeek's bare hosts), that another provider's bare host still gets `/v1`, and that an address an earlier version saved with `/v1` added must be entered again. `Docs/Architecture/llm-providers.md`: the base-URL section describes the new reading, `_BARE_HOST_CLOUD_KEYS`, and why project-instruction keys are unaffected.

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `tldw_chatbook/Chat/provider_setup_persistence.py`, `Tests/Chat/test_local_server_discovery.py`, `Docs/User_Guide/settings.md`, `Docs/Architecture/llm-providers.md`.

**Review r17cu-rev1 round (2026-11-19).**
- I-1 (Important): reading every no-`v1` cloud path as its base also changed an own-host prefix of the documented path: Groq's `https://api.groq.com/openai`, OpenRouter's `https://openrouter.ai/api` and QwenCloud's `https://dashscope-intl.aliyuncs.com/compatible-mode` (and their trailing-`/` forms), saved before as `.../v1` and posted to the documented route, saved bare and posted `/openai/chat/completions` etc. Now, on the provider's own host (the registry default's host), a path that is a prefix of the documented path reads as the documented path (`_documented_own_path`, applied in `_engine_preset_resolution` after a pasted `/chat/completions` or `/models` is dropped). Google's `/v1beta` (equal to its default), the bare-host rules and every gateway host are unchanged.
- M-1: DeepSeek's bare-host reading applied on any host, so a proxy under its key (`http://localhost:4000`) saved bare and posted `/chat/completions`. It is now scoped to DeepSeek's own host (it falls out of the rule above: its documented path is empty); a foreign bare host gets `/v1` again. Cohere stays bare on any host (`_BARE_BASE_ANY_HOST_KEYS`, its adapter appends `/v2/chat`).
- `test_a_cloud_base_without_v1_saves_unchanged_and_posts_there` is now `test_a_cloud_base_without_v1_saves_a_base_its_adapter_posts_under`, with rows of (typed, saved, posted): new rows for Groq `/openai`, OpenRouter `/api`, QwenCloud `/compatible-mode` (each also with a trailing `/`) and DeepSeek on `http://localhost:4000`. Red at 5a78acaf36: 4 of 7 nodes (Groq, OpenRouter, QwenCloud saved bare; DeepSeek's proxy saved bare); the trailing-`/` forms also saved bare there (probe). Green after.
- Typed forms (bare host, host with `/`, each prefix of the documented path with and without `/`, the default, `/chat/completions`, a gateway, `http://localhost:4000`) compared for every cloud key with c176f553b3's sources: the only differences left are the intended ones (no-`v1` gateway paths and Google's `/v1beta` saved as typed, Cohere's bare hosts and DeepSeek's own bare host saved bare, an own-host `/chat/completions` paste now reading as the documented base).
- Docs: the User Guide sentence now says the provider's own documented address and a gateway's are saved as typed, an own-host prefix of the documented path is completed (Groq's example), and any other bare host gets `/v1` except under Cohere; `llm-providers.md` likewise.
- Reruns: `test_local_server_discovery.py`, `test_provider_endpoint_contract.py` and the other contract callers outside Tests/UI, the QwenCloud, hosted-chat, DeepSeek, Groq/OpenRouter, Moonshot, Z.ai, Anthropic and custom-credential adapter files, and `test_settings_provider_save_scope.py` (26 files, `-n 4`, load 17-22): 2788 passed. `Tests/CI` 1811 passed. Preflight passed.
