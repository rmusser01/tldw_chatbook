---
id: TASK-41710
title: "Readiness never reads the legacy Custom OpenAI API 2 key, so Console and Dreams disagree for a legacy-only config"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: '2026-11-19 16:34'
labels:
- follow-up
- providers
- config
priority: medium
references:
- tldw_chatbook/config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-41523 (round 16). With only the legacy `[API] custom_openai2_api_key` set, Settings shows no key and Console sends keyless (Console passes readiness's key), while Dreams and import analysis send the legacy key. Nine providers already have their legacy key bridged into `api_settings.<provider>.api_key` at load time (`_LEGACY_PROVIDER_API_KEY_BRIDGE`, `_normalize_legacy_provider_api_key`); Custom 2 does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A legacy-only Custom 2 key is seen by readiness, Settings and Console the same way Dreams and import analysis see it, or is retired with a named migration
- [x] #2 A test pins a legacy-only config across readiness and a non-Console send, red before the change
- [x] #3 The User Guide and `Docs/Architecture/llm-providers.md` legacy-key sentences match the new behaviour
<!-- AC:END -->

## Implementation Plan

1. Check at base: `load_settings` bridges nine providers' legacy `[API]` keys into `api_settings.<provider>.api_key`; Custom 2's `custom_openai2_api_key` reaches only `chat_with_custom_openai_2`'s own fallback, so readiness, Settings and Console never see it.
2. Red first in the lane file: a real scratch config with only the legacy key reads Ready with that key in readiness, a Dreams send spends it, and the loader names `custom_2` among the tables it filled (so a Settings "stored" save writes the key into the table).
3. Bridge it in the loader beside the nine (helper in `Utils/legacy_custom_2_endpoint.py`, config.py stays at its size ceiling), in memory only, under TASK-41523's rule: only while the slot records no credential decision, holds no key of its own (ciphertext included, so a still-encrypted key keeps blocking) and its named variable is unset. Drop the handler's own legacy fallback so readiness and the handler read one value.
4. Re-point TASK-41523's legacy-key pins to the real loader; update the User Guide and `Docs/Architecture/llm-providers.md` sentences; covering tests, preflight, ratchets.

## Implementation Notes

Checked at base c176f553b3: `load_settings` copies nine providers' legacy `[API] <provider>_api_key` into `api_settings.<provider>.api_key` (and names them in `legacy_api_key_providers`); Custom 2's `custom_openai2_api_key` reached only `chat_with_custom_openai_2`'s own fallback. With only that key set, readiness was Ready with no key, so Settings showed none and Console sent keyless, while Dreams and import analysis sent the legacy key.

**Fix:** the key is bridged, not retired, matching the owner's TASK-41145 ruling that kept the legacy `custom_openai2_api_ip` (carried into the slot in memory). `Utils/legacy_custom_2_endpoint.bridge_legacy_custom_2_api_key`, called by `load_settings` beside the nine-provider bridge, copies the legacy key into `api_settings.custom_2.api_key` in memory and adds `custom_2` to `legacy_api_key_providers`, so a Settings save that keeps the stored key writes it into the slot (the TASK-33001.15 path). It keeps TASK-41523's rule: only while the slot records no `credential_source`, holds no key of its own (ciphertext included, so a still-encrypted key keeps readiness at "still encrypted" instead of being shadowed by the older key) and its `api_key_env_var` variable holds no key (the variable still outranks the legacy key, as before). `chat_with_custom_openai_2` drops its own legacy fallback: the key now arrives through the slot, so the handler and readiness read one value (the one case where they differed was an encrypted slot key plus a legacy key, where the handler sent the legacy key while readiness blocked). The helper lives in `Utils/` because `config.py` is at its module-size ceiling; config.py's call site is offset by a shorter comment (net 0 lines).

**Red/green** (lane file, real scratch config through `select_config_source`):
- `test_a_legacy_only_custom_2_key_reads_the_same_in_readiness_and_every_send`: readiness Ready with the legacy key from `config:api_settings.custom_2.api_key`, a Dreams send and a Console-shaped send (`api_key_resolved=True`) both send `Bearer legacy-key-0000`, the loader names `custom_2`, and a Settings "stored" save writes the key into the slot. Red with base sources (`(True, None, None) == (True, 'legacy-key-0000', ...)`), green after. Settings' API key row reads **saved in config** from the same source (`settings_screen._provider_saved_api_key_present` calls `provider_credential_source`, readiness's resolver over `app_config`), so the asserted `config:` source is the one that row shows.
- `test_custom_2_reads_its_legacy_key_only_without_a_credential_decision` (TASK-41523's pin, moved from a hand-built dict to the real loader, so it now covers the bridge): `[slot-key-encrypted]` is new and red with base sources (`['Bearer legacy-key-0000'] == [None]`); `[no-decision]`, `[slot-key-first]`, `[env-before-legacy]` (new), `[none]` and `[stored-without-a-key]` are controls, green at base and after.
- `Tests/LLM_Calls/test_custom_openai_credential_resolution.py`: two custom_2 pins put the "configured credential" only in the legacy dict, the place the handler no longer reads; they now put it in the slot, where the loader does. The two stale-credential pins also hold it in the slot, so they still prove a resolved keyless call reloads nothing.

Docs: `Docs/Architecture/llm-providers.md` (Credential precedence row), `Docs/User_Guide/settings.md` (the tenth legacy key; Custom 2's key paragraph).

Files: `tldw_chatbook/Utils/legacy_custom_2_endpoint.py`, `tldw_chatbook/config.py`, `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/LLM_Calls/test_custom_openai_credential_resolution.py`, the two docs.

**Review r17rd-rev1 (three fixes):**
- Minor 1: the User Guide said any save with Custom 2 chosen copies the legacy key; only a connection save does (`category_panels.py` `connection_dirty`, then `build_provider_setup_mutation(keep_stored_credential=True)`). Reworded to match the nine-key paragraph: a change of provider, model or endpoint copies it, a save of other rows alone copies nothing.
- Minor 2: under `credential_source = "environment"` with its variable unset the send is keyless whether or not the bridge copies, so no auth assertion caught dropping `"environment"` from the bridge's set. `test_custom_2_reads_its_legacy_key_only_without_a_credential_decision` gains `[environment-unset]` and, for every case, asserts the loader's own record (`legacy_api_key_providers`, and whether the slot holds the legacy key). Red with `"environment"` temporarily removed from the set (`(('custom_2',), True) == ((), False)`), green restored; the other six cases are unchanged controls.
- Recommendation: the bridge's hand-written `{"none", "stored", "environment"}` now names `Chat.provider_readiness._PERSISTED_CREDENTIAL_SOURCES` as its source of truth (imported by name it would be an upward Utils -> Chat import).
- Runs (wrapper, `-n 4`, load 40-44): the three new tests, 10 passed; `./scripts/preflight.sh`, all checks passed.

Not changed, proposed as a follow-up: the nine-provider bridge still copies a legacy key over a still-encrypted `api_settings.<provider>.api_key` (its modern check treats ciphertext as absent), so readiness reports Ready with the older key instead of "still encrypted".

**Runs** (all through the wrapper; load average 20-55):
- New tests, one command (`-n 4`, `-k` over the three tests): base sources 4 failed / 5 passed (every red named above), this branch 9 passed. The same three tests plus every Dreams node in the lane file, serially with the lane step's flags (`--storage-lease-guard --timeout=180`): 15 passed in 8.4 s at load 51-55.
- Covering, `-n 4`: `Tests/Chat/test_local_server_discovery.py`, `Tests/Dreams`, `test_encrypted_api_key_never_sent.py`, `test_provider_readiness.py`, `test_legacy_api_key_credential_source.py`, `test_provider_setup_persistence.py`, `Tests/LLM_Calls/test_custom_openai_credential_resolution.py`, `test_custom_endpoint_engine_swap.py`, `test_console_provider_gateway.py`, `Tests/Utils/test_config_api_key_resolution.py`, `Tests/Library/test_ingest_analysis.py`, `Tests/LLM_Calls/test_summarization_chat_dispatch.py`, `Tests/test_ciphertext_never_sent_outside_chat.py`, `test_stranded_key_local_provider_readiness.py`: 14 failed / 3307 passed. Two were this branch's (`test_custom_openai_credential_resolution.py`'s custom_2 legacy-credential pins, re-pointed below; that file then 15 passed). The other 12 fail identically with base sources: `Tests/Dreams/test_query_synthesis.py::test_no_wrapping_lets_a_mark_through_or_loses_its_goal` (a Hypothesis falsifying example replayed from the worktree's example database; the test and `query_synthesis.py` are unchanged), three `test_console_provider_gateway.py` nodes (qwencloud, deepseek, openai stop; filed on dev as TASK-40413), and eight `test_config_api_key_resolution.py` nodes on the local `RecoveryRequired: raw_source_selection_changed` trap.
- Covering, `-n 4`: every other file that dispatches custom_2 (`test_chat_integration_APIs.py`, `test_console_chat_fork.py`, `test_console_provider_support.py`, `test_local_adapter_thinking_dispatch.py`, `test_local_thinking_wire_formats.py`, `test_provider_endpoint_contract.py`, `Tests/Library/test_library_rag_answer_provider_resolution.py`, `Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py`, `Tests/test_provider_registry.py`, `Tests/UI/test_settings_provider_save_scope.py`, `Tests/test_config_raw_snapshot.py`, `test_custom_openai_execution_keys_constant.py`): 1190 passed, 129 skipped.
- Serially: `Tests/Architecture/test_module_size_ratchet.py` (config.py stays at its 8,837 ceiling), `test_layer_direction.py`, `test_screen_size_ratchet.py`, `test_agents_chat_boundary.py`: 242 passed.
- `./scripts/preflight.sh`: all checks passed.

**2026-11-20 (round-17 final checks).** The stack-top boot budget run found `Utils.legacy_custom_2_endpoint` newly eager on the boot path (`import tldw_chatbook.app` loaded 688 modules against a limit of 688, "NEW modules (1)"), because `load_settings` imported it unconditionally. The import and the bridge call now run only when `[API] custom_openai2_api_key` is set: the bridge does nothing without a key, so behaviour is unchanged. The change is line-neutral (config.py stays at 8837). After it, `Tests/Performance/test_app_import_weight.py` passes; `test_local_server_discovery.py`, `test_custom_openai_credential_resolution.py` and `test_story_service.py` give 1057 passed at -n 4.
