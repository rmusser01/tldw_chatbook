---
id: TASK-41709
title: "Dreams sends without checking provider readiness, so keyless and still-encrypted keys reach the handler"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: '2026-11-19 15:25'
labels:
- follow-up
- providers
priority: medium
references:
- tldw_chatbook/Dreams/story_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-41523 (round 16). `story_service.resolve_dreams_chat` takes the readiness result's `api_key` without checking `ready` (~story_service.py:257). For a slot whose stored key is still `enc:` ciphertext, readiness blocks ("still encrypted") while Dreams sends keyless; with an `enc:` slot key plus a plaintext legacy `[API] custom_openai2_api_key` and no `credential_source`, Dreams sends the legacy key while readiness says the key is still encrypted. Ingest checks readiness; Dreams does not. It affects every keyless local provider, not only Custom.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Dreams send whose provider readiness is not ready is refused with readiness's reason and sends nothing
- [x] #2 Tests cover a still-encrypted slot key and the enc-plus-legacy case, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Check at base: `resolve_dreams_chat` takes `readiness.api_key` from the bootstrap view (`load_cli_config_and_ensure_existence`) and never checks `ready`; a still-encrypted Custom 2 slot key sends keyless, and with a legacy key beside it the handler sends the legacy key.
2. Red first in the lane file `Tests/Chat/test_local_server_discovery.py`: a Dreams send on a still-encrypted slot key, and on an encrypted slot key plus a plaintext legacy key, is refused with readiness's reason and posts nothing.
3. Fix in `Dreams/story_service.resolve_dreams_chat`: read readiness over `load_settings()` (the view the handlers spend from, where legacy `[API]` keys are bridged, so a legacy-only key for the nine bridged providers is not newly refused), and raise the cycle's RuntimeError with readiness's reason when it is not ready.
4. Update the Dreams test doubles that stub readiness without `ready`, covering tests, preflight, ratchets.

## Implementation Notes

Checked at base c176f553b3: `resolve_dreams_chat` read readiness over the bootstrap view (`load_cli_config_and_ensure_existence(force_reload=True)`) and took `.api_key` without looking at `.ready`. With Custom 2's slot key still `enc:` ciphertext readiness says "Saved API key is still encrypted", yet Dreams sent with no key; with a plaintext legacy `[API] custom_openai2_api_key` beside it, the handler's own legacy fallback sent that key.

**Fix** (`Dreams/story_service.resolve_dreams_chat`): readiness that is not ready raises the cycle's existing `RuntimeError`, now `"Dreams provider is not ready (<readiness.reason>)"`, before the closure exists, so nothing is sent. The cycle already degrades on that error (no synthesis or story call; the note "query synthesis failed: ..." reaches the Collection notice), and the track check logs the error type. Not a product decision: the AC names the refusal, and the cycle's skip-with-a-note path is the one it already takes for an unresolvable provider.

Decision: readiness (and the provider/model lookup) now read `load_settings()` instead of the bootstrap view. `config._normalize_legacy_provider_api_key` copies the nine providers' legacy `[API]` keys into `api_settings` only in `load_settings`, which every handler reads; gating on bootstrap-view readiness would have refused a legacy-only OpenAI or Anthropic key that Dreams sends today. Lesson recorded in `backlog/docs/lessons-testing-evidence.md`.

**Red/green** (lane file `Tests/Chat/test_local_server_discovery.py`, real scratch config through `select_config_source`, ciphertext from `Utils/config_encryption.ConfigEncryption().encrypt_value`, transport stubbed): `test_dreams_refuses_a_still_encrypted_key_and_sends_nothing[alone]` red with base sources (`(None, [POST], [None])`: sent keyless), `[with-a-legacy-key]` red (`(None, [POST], ['Bearer legacy-key-0000'])`); both green after.

Test doubles: `Tests/Dreams/test_story_service.py`'s readiness stubs gain `ready=True`, and its and the lane file's Dreams tests patch `story_service.load_settings` instead of the bootstrap loader.

Docs: `Docs/User_Guide/dreams.md` (Dreams sends only while its provider reads ready; the Collection notice gives the reason). `Dreams/cycle_service.py`'s comment no longer says the getter force-reloads.

Files: `tldw_chatbook/Dreams/story_service.py`, `tldw_chatbook/Dreams/cycle_service.py` (comment), `Tests/Chat/test_local_server_discovery.py`, `Tests/Dreams/test_story_service.py`, `Docs/User_Guide/dreams.md`, `backlog/docs/lessons-testing-evidence.md`.

**Runs** (all through the wrapper; load average 20-55):
- New tests, one command (`-n 4`, `-k` over the three tests): base sources 4 failed / 5 passed (every red named above), this branch 9 passed. The same three tests plus every Dreams node in the lane file, serially with the lane step's flags (`--storage-lease-guard --timeout=180`): 15 passed in 8.4 s at load 51-55.
- Covering, `-n 4`: `Tests/Chat/test_local_server_discovery.py`, `Tests/Dreams`, `test_encrypted_api_key_never_sent.py`, `test_provider_readiness.py`, `test_legacy_api_key_credential_source.py`, `test_provider_setup_persistence.py`, `Tests/LLM_Calls/test_custom_openai_credential_resolution.py`, `test_custom_endpoint_engine_swap.py`, `test_console_provider_gateway.py`, `Tests/Utils/test_config_api_key_resolution.py`, `Tests/Library/test_ingest_analysis.py`, `Tests/LLM_Calls/test_summarization_chat_dispatch.py`, `Tests/test_ciphertext_never_sent_outside_chat.py`, `test_stranded_key_local_provider_readiness.py`: 14 failed / 3307 passed. Two were this branch's (`test_custom_openai_credential_resolution.py`'s custom_2 legacy-credential pins, re-pointed below; that file then 15 passed). The other 12 fail identically with base sources: `Tests/Dreams/test_query_synthesis.py::test_no_wrapping_lets_a_mark_through_or_loses_its_goal` (a Hypothesis falsifying example replayed from the worktree's example database; the test and `query_synthesis.py` are unchanged), three `test_console_provider_gateway.py` nodes (qwencloud, deepseek, openai stop; filed on dev as TASK-40413), and eight `test_config_api_key_resolution.py` nodes on the local `RecoveryRequired: raw_source_selection_changed` trap.
- Covering, `-n 4`: every other file that dispatches custom_2 (`test_chat_integration_APIs.py`, `test_console_chat_fork.py`, `test_console_provider_support.py`, `test_local_adapter_thinking_dispatch.py`, `test_local_thinking_wire_formats.py`, `test_provider_endpoint_contract.py`, `Tests/Library/test_library_rag_answer_provider_resolution.py`, `Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py`, `Tests/test_provider_registry.py`, `Tests/UI/test_settings_provider_save_scope.py`, `Tests/test_config_raw_snapshot.py`, `test_custom_openai_execution_keys_constant.py`): 1190 passed, 129 skipped.
- Serially: `Tests/Architecture/test_module_size_ratchet.py` (config.py stays at its 8,837 ceiling), `test_layer_direction.py`, `test_screen_size_ratchet.py`, `test_agents_chat_boundary.py`: 242 passed.
- `./scripts/preflight.sh`: all checks passed.
