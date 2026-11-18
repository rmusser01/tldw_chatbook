---
id: TASK-41522
title: "First-party cloud adapters send to an unchecked base-URL override in their own config table when the caller pins nothing"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 05:35
labels:
- follow-up
- security
- providers
priority: high
references:
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
- tldw_chatbook/Provider_Contract
dependencies:
- TASK-41131
- TASK-41132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the endpoint review (round 15) and repeated in endpoint fix r1: the first-party cloud adapters (OpenAI, Anthropic, Cohere, Google, HF, Groq, OpenRouter, DeepSeek, Mistral, Moonshot, Z.ai, QwenCloud) send to a base-URL override in their own config table without going through the provider endpoint contract when the caller pins nothing. This was out of scope for the endpoint round. Related: TASK-41131 and TASK-41132.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed adapter validates a configured base-URL override through the endpoint contract before any request leaves
- [x] #2 A refused override surfaces a named refusal, not a silent fallback
- [x] #3 A test per adapter family is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Establish which adapters already check: engine presets and custom-hosted (`hosted_provider_engine._resolve_base_url`) and every local/custom adapter (TASK-41131) do; none of the twelve first-party cloud adapters does.
2. Red tests in `Tests/Chat/test_local_server_discovery.py` (lane-run), one node per cloud dispatch name, looping over the refused shapes from the adapter's own table (and pinned): no request leaves (requests' `Session.send` stubbed), a status-less `ChatConfigurationError` with the contract's reason, no echo. Plus a keeps-working check: the shipped default and an accepted override still post where they did.
3. Fix with the existing helper (`LLM_API_Calls_Local._refuse_a_refused_endpoint`) on the base each adapter is about to use, before any session or request:
   - the strict hosted transport `hosted_chat.owned_json_post` (Groq, OpenRouter, DeepSeek, Mistral, Moonshot, Z.ai), with the transport's provider key;
   - the bespoke adapters at their own base resolution: OpenAI, Anthropic, Cohere, Google, HuggingFace (router and standard modes), QwenCloud (its normalised base).
4. Covering tests, preflight, size ratchets; notes.

## Implementation Notes

Every first-party cloud adapter now checks the base URL it is about to send to, from its own table or pinned, against the provider endpoint contract before any session or request, with TASK-41131's helper `_refuse_a_refused_endpoint` (no second checker). A refusal is a status-less `ChatConfigurationError` "<Name> API URL: <contract reason>" carrying the adapter's key (`mistral` for both Mistral dispatch names); the reason names no part of the value.

**Who already checked (base 45b1949ca3).** The engine presets and custom-hosted (`hosted_provider_engine._resolve_base_url`) and every local and custom adapter (TASK-41131). None of the twelve cloud adapters: OpenAI, Anthropic, Cohere, Google and Hugging Face had no check at all (every refused shape was sent); Groq, OpenRouter, DeepSeek, Mistral, Moonshot, Z.ai and QwenCloud had only their structural normalisers, which refuse a query, fragment, user info or zone id but sent `127.1`, `0x7f.1`, `-bad.corp`, `localhost..` and an IDN host.

**Where.**
- `hosted_chat.owned_json_post`, the strict transport Groq, OpenRouter, DeepSeek, Mistral, Moonshot and Z.ai (and the engine, which already checks) route through, before its own normaliser, with the transport's provider key and the registry display name ("Groq", not "groq"; existing transport copy is unchanged).
- `chat_with_openai`, `chat_with_anthropic`, `chat_with_cohere`, `chat_with_google` and `chat_with_huggingface` (router base in router mode, the configured base in standard mode) where each resolves its base.
- `chat_with_qwencloud` on its normalised base, not the raw value: the raw check refused a pasted `.../compatible-mode/v1/responses` endpoint that `test_qwencloud.py` pins as working (the contract's generic reading calls it ambiguous).
- Moonshot and Z.ai normalise their base in the resolver their readiness and catalog share, so a malformed base is still refused there first with their own copy ("Moonshot API base URL is invalid."); left alone so readiness is unchanged. QwenCloud's normaliser likewise keeps its own copy for those shapes.

**Tests** (`Tests/Chat/test_local_server_discovery.py`, the lane step that already runs it; requests' `Session.send` and the session factories stubbed, every request recorded and answered 400, nothing reaches a socket):
- `test_a_cloud_adapter_never_reaches_a_base_url_the_contract_refuses`, one node per dispatch (14: the twelve, `mistralai`, and Hugging Face router mode), each looping over TASK-41131's refused shapes plus the hex host, from the adapter's own table and pinned. Red with base sources: 14 of 14 failed on `assert sent == []` (a request recorded on the first shape); 14 of 14 green after.
- `test_a_cloud_adapter_still_posts_where_it_did`: every adapter's shipped base and an accepted override (`http://gpu-box.lan:8000/v1`) post to the same literal URL; green with base sources and after.
- Serial cost of the 15 new nodes: 26 s at load 61-65 (`-p no:xdist --timeout=180`).
- `Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_maps_invalid_base_url_to_redacted_transport_error` fed a user-info base, which the contract now refuses first (a `ChatConfigurationError`, not the transport error). It now feeds a base only the transport refuses (`.../RAW-URL-CANARY/v1/models`), keeping its redaction check; the user-info case is covered (no echo) by the new test.

**Covering run** (87 files that call the cloud adapters, the strict transport or `chat_api_call` with them; `-n 4`, load 51-90): 177 failed / 4674 passed. Re-run of the 177 with base sources: 172 failed the same way (139 are the known `RecoveryRequired: raw_source_selection_changed` trap; the others fail at base for other reasons). The five head-only reds: the hosted-chat test above (updated) and the two QwenCloud pasted-`/responses` nodes (fixed by checking the normalised base), both green now; `test_settings_provider_key_check.py`'s `[401]`/`[env]` nodes, listing-only UI tests that timed out under load 57-92 and pass 33 of 33 at head at load 36-73. After the QwenCloud change, the 15 covering files that touch QwenCloud: 3 failed / 2488 passed, all 3 red at base.

**Other checks.** `Tests/Architecture/test_module_size_ratchet.py` and `test_screen_size_ratchet.py` 232 passed (LLM_API_Calls.py row 5189 -> 5202, the change's own +13, owner rule). `Tests/Performance` boot budgets: the first-send warm-up (95 > 94) and UI-ready census (1041 > 1040) fail identically with base sources (modules from dev: `RAG_Search.scope_evidence`, `DB.character_conversation_search`, `Provider_Contract.provider_setup_facts`); the pre-import LOC pass is green. `ruff_gate.py` OK. Preflight passed.

**Behaviour change.** A non-Console caller with a cloud base the contract refuses is now refused, as Console already refused it since TASK-40300; for example OpenAI behind a Cloudflare AI Gateway (`https://gateway.ai.cloudflare.com/v1/<acct>/<gw>/openai`, "Endpoint path has ambiguous API suffixes."). Superseded by TASK-41700 (owner decision 2026-11-18): a cloud provider's gateway base is now accepted, in Console and everywhere else.

Files: `tldw_chatbook/LLM_Calls/hosted_chat.py`, `LLM_API_Calls.py`, `qwencloud.py`, `LLM_API_Calls_Local.py` and `Provider_Contract/provider_endpoint_contract.py` (docstrings), `Docs/Architecture/llm-providers.md` (gotcha 6), `Tests/Chat/test_local_server_discovery.py`, `Tests/LLM_Calls/test_hosted_chat.py`, `Tests/Architecture/test_module_size_ratchet.py`.

**Review r16cu-rev1 round (2026-11-18).**
- Minor 1: `Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_refuses_a_user_info_base_without_rendering_it` puts back a traceback no-echo check for a user-info base at the transport (`https://user:CRED-CANARY@example.com/v1`): a status-less `ChatConfigurationError`, and the canary absent from the whole rendered traceback, chained exceptions included. Red under a temporary edit that raised the refusal `from ValueError(<url>)` (the canary rendered in the chained `ValueError`), green after reverting it. That file runs only in the nightly whole-suite job, so review r16cu-rev2 M-3 moved the test to `Tests/Chat/test_local_server_discovery.py::test_the_strict_transport_refuses_a_user_info_base_without_rendering_it` (lane-run; same negative control, re-run there).
- Minor 2: the status, provider and no-echo assertions now run for every case, including Moonshot's, Z.ai's and QwenCloud's own malformed-base refusals. Moved, they were red for those three (status 500: the families' configuration-error default), so the three base-URL refusals are now status-less like the contract's (nothing was sent): `moonshot._resolve_base_url`, `zai._resolve_base_url`, `qwencloud.normalize_qwencloud_base_url`. Their callers catch by type. 1790 passed across the discovery, hosted-chat, Moonshot, Z.ai, QwenCloud, Kimi/Z.ai contract and readiness files.

