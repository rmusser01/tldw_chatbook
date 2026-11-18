---
id: TASK-41523
title: "Custom OpenAI API 2 reports ready without a key, but its handler requires one"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-18 04:09'
created_date: 2026-11-17 05:27
labels:
- follow-up
- providers
- settings
priority: medium
references:
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
- tldw_chatbook/Provider_Contract/provider_setup_facts.py
dependencies:
- TASK-41143
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in endpoint fix r1 (round 15). Readiness says Custom OpenAI API 2 is ready without a key, but `chat_with_custom_openai_2` requires one unless the caller resolved it, so keyless Dreams and ingest sends fail with "API Key required" (pre-existing as "api_ip required").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Readiness and the handler agree on whether a key is needed for a configured endpoint
- [x] #2 A keyless endpoint that does not need a key sends from Dreams and ingest, or readiness says Needs key
- [x] #3 A test covers both
<!-- AC:END -->

## Implementation Plan

1. Check at base: readiness reads custom_2 as keyless (Ready, no key); `chat_with_custom_openai_2` raises "API Key required" when the caller passes no key and the legacy `[API] custom_openai2_api_key` is unset, while Custom OpenAI API (1) resolves its key from its slot with readiness's own resolver and sends keyless.
2. Red first in `Tests/Chat/test_local_server_discovery.py` (lane file): a keyless custom / custom_2 slot is Ready with no key, and a Dreams send and an ingest analysis both post to the slot's endpoint (custom is the control).
3. Fix in `chat_with_custom_openai_2`: resolve the key from `[api_settings.custom_2]` with `resolve_provider_credential`, as Custom 1 and readiness do; keep the legacy `[API] custom_openai2_api_key` as the fallback only while the slot records no credential decision; drop the missing-key refusal.
4. Re-point the pins that recorded the old "custom_2 requires a key" premise (the ciphertext test keeps its no-ciphertext-on-the-wire guarantee).
5. Covering tests, preflight, ratchets.

## Implementation Notes

Checked at base 45b1949ca3, after TASK-41143 and TASK-41145: readiness reads `custom_2` as a keyless provider (`custom_2` is in `KEYLESS_PROVIDER_KEYS`, so a slot with no key is Ready with no key). `chat_with_custom_openai_2` read its key only from the legacy `[API] custom_openai2_api_key` (`load_settings`' `custom_openai_api_2.api_key`) and raised "custom_openai_api_2 API Key required." when that was unset and the caller had not resolved the key. Console passes `api_key_resolved=True` and was unaffected; Dreams (`readiness.api_key`, None) and ingest analysis (`resolution.api_key`, None) were not. Custom OpenAI API (1) already resolves its key from its slot with readiness's own `resolve_provider_credential` and sends keyless, so it agrees with readiness and needed no change.

**Fix** (`chat_with_custom_openai_2`): a key the caller passes or resolved wins, as before. Otherwise the key comes from `[api_settings.custom_2]` through `resolve_provider_credential("custom_2", ...)`, the resolver readiness and Custom 1 use (a recorded `credential_source`, else the stored key, else the variable `api_key_env_var` names). The legacy `[API] custom_openai2_api_key` stays the fallback, but only while the slot records no credential decision, so an explicit "none" or "stored" never brings it back. Only non-Console sends (Dreams, ingest analysis) can reach it: Console always passes readiness's key with `api_key_resolved=True`, and readiness never reads the legacy key, so Settings shows no key and Console sends keyless for a config that sets only the legacy key. With no key from any of these, the send is keyless. The missing-key refusal is gone. No setting was added.

Decision: the legacy key is kept as a fallback (not dropped as Custom 1 drops its own) by analogy with the owner's TASK-41145 ruling that kept the legacy `api_ip`; a config that set only that key keeps sending it from Dreams and ingest. For the keyless config the task is about, both slots now behave the same.

**Red/green** (lane file `Tests/Chat/test_local_server_discovery.py`):
- `test_a_keyless_custom_slot_sends_from_dreams_and_ingest[custom|custom_2]`: readiness Ready with no key, then a Dreams send and an ingest analysis each post to the slot's endpoint with no Authorization header. `custom_2` red at base ("custom_openai_api_2 API Key required."); `custom` is the control, green at base and after.
- `test_custom_2_reads_its_legacy_key_only_without_a_credential_decision[no-decision|slot-key-first|none|stored-without-a-key]`: with base sources `none` and `stored-without-a-key` sent the legacy key (`['Bearer legacy-key-0000'] == [None]`) and `slot-key-first` sent the legacy key over the slot's (`== ['Bearer slot-key-0000']`); `no-decision` is the control (green at base: the legacy key keeps working). All four green after.
- `Tests/Chat/test_encrypted_api_key_never_sent.py`: `test_custom_openai_2_refuses_ciphertext` pinned the old "custom_2 requires a key" premise (ciphertext had to be the missing-key refusal). Custom 2 now joins `test_local_handlers_never_forward_ciphertext` (key optional: ciphertext in its slot or its legacy key reads as no key, never sent); `_local_config` gains the `custom_2` slot so the slot path carries ciphertext too. `[chat_with_custom_openai_2]` red with base sources (`ChatConfigurationError('custom_openai_api_2 API Key required.')`, nothing forwarded), green after; the other six handlers are unchanged controls.

Not changed, proposed as follow-ups: (a) with the slot's stored key still `enc:` ciphertext, readiness blocks ("still encrypted") while both Custom handlers send keyless, and Dreams sends without checking `readiness.ready` (ingest analysis checks it); fixing it is the same code in both handlers, but it is a locked-key question, not the needs-a-key one this task is about. (b) Readiness never reads the legacy `[API] custom_openai2_api_key`, so with only that key set readiness reports no key while Dreams and ingest spend it (pre-existing; the old handler read only that key).

Docs: `Docs/Architecture/llm-providers.md` (Credential precedence: a Custom OpenAI API 1 and 2 row), `Docs/User_Guide/settings.md` (Custom 2's key). Review r16ky-rev1 (Minor 1): the User Guide paragraph said the older key "is still read" without saying by what; it and the precedence row now say only non-Console sends (Dreams, import analysis) read it, and that Settings and Console do not see it.

Files: `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/Chat/test_encrypted_api_key_never_sent.py`, the two docs.

**Runs** (all through the wrapper; load average 50-100 unless noted):
- New tests, one command (`-n 4`, `-k` over the two files): base sources 7 failed / 8 passed (every red named above), this branch 15 passed.
- `Tests/Chat/test_local_server_discovery.py` serially with its lane step's flags (`--storage-lease-guard --timeout=180`): 982 passed in 477 s at load 66-102 (before the `slot-key-first` case was added); every new node took under 1 s.
- Covering, `-n 4`: the lane file, `Tests/LLM_Calls/test_custom_openai_credential_resolution.py`, `test_encrypted_api_key_never_sent.py`, `test_custom_endpoint_engine_swap.py`, `test_custom_openai_execution_keys_constant.py`: 1 failed / 1073 passed. The failure, `test_no_bare_custom_literals_outside_the_audited_allowlist`, names four files this branch does not touch and fails with the identical message with base sources.
- Covering, `-n 4`: every other file that dispatches custom_2 (`Tests/Chat/test_chat_integration_APIs.py`, `test_console_chat_fork.py`, `test_console_provider_support.py`, `test_local_adapter_thinking_dispatch.py`, `test_local_thinking_wire_formats.py`, `test_provider_endpoint_contract.py`, `test_provider_readiness.py`, `Tests/Library/test_library_rag_answer_provider_resolution.py`, `Tests/LLM_Calls/test_summarization_chat_dispatch.py`, `Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py`, `Tests/Scheduling/test_automation_execution.py`, `Tests/test_provider_registry.py`, `Tests/Dreams`, `Tests/Library/test_ingest_analysis.py`): 8 failed / 2473 passed / 129 skipped. The 8 are `Tests/Scheduling/test_automation_execution.py` nodes failing with the local `RecoveryRequired: raw_source_selection_changed` trap; the same 8 fail with base sources.
- UI, `-n 4`: `Tests/UI/test_settings_provider_save_scope.py` (the custom_2 Settings saves), `test_llamacpp_consumers.py::test_verified_console_adoption_rebases_a_dropped_sampler`, and the TASK-41525 pins: 32 passed, 1 failed (`test_console_saved_defaults_flow.py::test_a_blank_top_p_applies_summarises_and_sends`, "the Custom OpenAI 2 handler was never called"; it replaces the handler with a stub, and fails the same way with base sources).
- `./scripts/preflight.sh`: all checks passed. `Tests/Architecture/test_module_size_ratchet.py`, `test_screen_size_ratchet.py`, `test_agents_chat_boundary.py` serially: 235 passed (neither touched module has a row; both are under the 4,000-line guard).
