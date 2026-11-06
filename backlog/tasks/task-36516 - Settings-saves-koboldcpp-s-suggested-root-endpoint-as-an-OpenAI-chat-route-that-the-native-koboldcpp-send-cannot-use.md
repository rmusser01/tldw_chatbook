---
id: TASK-36516
title: Settings saves koboldcpp's suggested root endpoint as an OpenAI chat route that the native koboldcpp send cannot use
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: '2026-11-05 22:16'
labels:
- follow-up
- settings
- providers
- bug
priority: medium
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py
dependencies:
- TASK-36380
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36380's review round 2 (PR #3224, 2026-10-23). That PR fixed only the Discover models hint. The bug is older than the PR.

A user who types the endpoint Settings suggests for koboldcpp gets a provider that cannot send:
- **The placeholder is a bare root.** `PROVIDER_ENDPOINT_PLACEHOLDERS["koboldcpp"]` is `http://127.0.0.1:5001` (settings_screen.py:1710).
- **A bare root is saved as the OpenAI chat route.** The endpoint contract reads a root as form `origin`, with `chat_url = <root>/v1/chat/completions` and `persisted_endpoint = chat_url` (provider_endpoint_contract.py:290-292). The Settings save writes that value to `[api_settings.koboldcpp] api_url` (provider_setup_persistence.py:889-897 and :1363-1373). A probe on dev 5b5920b8e2: `resolve_provider_endpoint("koboldcpp", "http://127.0.0.1:5001")` gives `persisted_endpoint == "http://127.0.0.1:5001/v1/chat/completions"`.
- **The koboldcpp send is KoboldAI-native.** `chat_with_kobold` (LLM_API_Calls_Local.py:944) reads that `api_url` (:978). It posts the native generate payload (`prompt`, `max_length`, …; :1071) to the URL unchanged (:1127), then requires a `results` list (:1157-1164) and otherwise raises ChatProviderError "Unexpected response structure". An OpenAI chat completions route answers with `choices`.

Only the native route `/api/v1/generate`, the template default (config.py:5262), works end to end. Not checked against a running koboldcpp server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every koboldcpp endpoint Settings suggests or saves leads to a Console send that posts a payload the saved route accepts; otherwise Settings refuses the endpoint and names the native /api/v1/generate route
- [x] #2 A koboldcpp endpoint already on the native route is saved unchanged, and a send to it still posts KoboldAI's native payload
- [x] #3 A test pins the URL and payload shape a koboldcpp send uses after Settings saves a bare root
- [x] #4 Other providers' saved endpoints are unchanged
<!-- AC:END -->

## Implementation Plan

1. Settings refuses, the contract stays as it is. The owner-ruled contract (TASK-35010) and the first-run wizard both treat a typed koboldcpp root as an OpenAI chat URL, and the wizard already refuses every koboldcpp endpoint whose chat URL is not the native route (`first_run_endpoint_refusal`, pinned by `Tests/Wizards/test_provider_review_contracts.py`). Settings gets the same check, so nothing outside Settings' koboldcpp save changes (AC#4).
2. `SettingsScreen._validate_provider_endpoint` takes the provider key; for koboldcpp it resolves the endpoint through the contract and returns the wizard's refusal ("KoboldCpp needs its native /api/v1/generate address. ..."), which the save shows before any write.
3. `PROVIDER_ENDPOINT_PLACEHOLDERS["koboldcpp"]` becomes the native route, `http://127.0.0.1:5001/api/v1/generate` (the wizard's `ENDPOINT_EXAMPLES` value).
4. Test (AC#3) in a whole-file UI-census file: on a private profile, Settings refuses a typed bare root and writes nothing; the placeholder's route saves unchanged; a koboldcpp send then posts KoboldAI's native payload to that URL (transport stubbed). Show it red before the fix.
5. Update the Settings User Guide page if it documents the koboldcpp endpoint.

## Implementation Notes

Settings now refuses a koboldcpp endpoint its native send cannot use, the way the first-run wizard already did; the endpoint contract is unchanged.
- Why refuse rather than map: the contract reading a typed root as an OpenAI chat URL is the TASK-35010 owner ruling, and the wizard's `first_run_endpoint_refusal` and its tests (`test_kobold_wizard_draft_refuses_non_native_routes`, `test_kobold_wizard_save_and_dispatch_use_native_url_with_prompt`) are built on it. A contract change would also have moved koboldcpp's connection identity and the wizard's decisions. This keeps the change to Settings' koboldcpp save (AC#4).
- `SettingsScreen._provider_endpoint_route_refusal(endpoint, provider_key)`: for koboldcpp it resolves the endpoint through the contract and returns the wizard's refusal, "KoboldCpp needs its native /api/v1/generate address. For an OpenAI-compatible server, choose Custom instead." The save checks it inside its `connection_dirty` branch, before `build_provider_setup_mutation`, so only a save that writes the endpoint refuses (a provider, model, endpoint or credential change). The save shows the refusal and stops before any write. A bare root, `/v1`, `/v1/chat/completions` and a prefix root are all refused; a URL ending in `/api/v1/generate` saves unchanged (AC#2). Both helpers are imported lazily, so Settings' UI-ready import set is unchanged.
- `PROVIDER_ENDPOINT_PLACEHOLDERS["koboldcpp"]` is now `http://127.0.0.1:5001/api/v1/generate`, the wizard's `ENDPOINT_EXAMPLES` value.
- AC#3, as the brief's "Settings refuses" option reads it: Settings no longer saves a bare root, so the test pins what a send uses after Settings is given one. `test_koboldcpp_saves_only_the_native_route_its_send_posts_to` (`Tests/UI/test_settings_provider_save_scope.py`, a whole-file UI-census entry, private profile) types `http://127.0.0.1:5001`, saves, and checks the refusal copy and that `api_url` on disk is unchanged. It then saves the placeholder's route and checks it was written unchanged. Last, `chat_api_call("koboldcpp", ...)` with the session stubbed posts once, to that URL, with `prompt` and no `messages`.
- Red before the fix: `assert 'Provider set... with Alt+M).' == 'KoboldCpp ne...stom instead.'`, because the bare root saved. Negative control with only the placeholder reverted: `assert 'http://127.0.0.1:5001' == 'http://127.0...i/v1/generate'`. Green after the fix.
- Runs (`-n 4`, run.sh, load average 39-64 on 18 CPUs): save-scope file, the hub's endpoint-validation test, `test_settings_provider_switch_atomic.py`, the koboldcpp Discover journey and three wizard endpoint files gave 102 passed and 8 failed. All 8 are `test_settings_provider_switch_atomic.py` plain tests failing with `RecoveryRequired: raw_source_selection_changed` (TASK-38111). The same 8 fail the same way with the base `settings_screen.py`.
- `Docs/User_Guide/settings.md` says which koboldcpp endpoints **s** saves and gives the refusal copy.
- Ratchet rows raised to the measured size, each with a dated comment naming this task (owner waiver): `settings_screen.py` 35,642 -> 35,669 (+27), `providers_models_card.py` 2,398 -> 2,401 (+3).

Review r8e-rev1:
- The round-1 check ran before Save knew what it writes. So a koboldcpp table holding a `/v1/chat/completions` route from the old bug refused every Providers & Models save, including a context window or a model default, which never write the endpoint. The check now runs in the `connection_dirty` branch. New `test_a_stored_koboldcpp_chat_route_refuses_only_saves_that_write_it` (save-scope file, private profile) pins both sides. A context-window save over the stored route is accepted and leaves `api_url` alone. A model change is still refused, and the model on disk stays as it was. Red at fd45ecd087: the first save returned the refusal.
- Endpoint guidance: `endpoint_requirement` (`UI/Settings_Modules/providers_models_card.py`, which feeds the Endpoint row help and the Inspector's Validation row) says "the server's /api/v1/generate address" for koboldcpp instead of "the server's base URL". `test_the_focused_endpoint_guide_says_required_as_its_row_does` gained a koboldcpp case. It is census-listed by function, so the case runs with no census edit. Red at fd45ecd087: `"required: th...er's base URL" == 'required: th...erate address'`. The Connect row of `Docs/User_Guide/settings.md` gives the koboldcpp copy.
- The refusal toast passes `markup=False` (TASK-1513).

Covering run after review r8e-rev1 (`-n 4`, load average 7-23): the Console defaults, endpoints, discovery, contract, setup persistence, model discovery and switcher probe files; the Settings provider test-draft, Connect key rows, save-scope, card-fit and Connect-paint files; the hub guidance and endpoint-validation tests; the koboldcpp Discover journey; and three wizard endpoint files. Result: 1206 passed, 2 failed. Both failures are in `test_settings_provider_test_draft.py`: `test_discovery_row_reads_a_fresh_models_capability_from_its_listing` (`AttributeError: 'SettingsScreen' object has no attribute '_nodes'`) and `test_t_hotkey_does_not_run_test_while_input_focused` (`RecoveryRequired`). Both fail the same way with all five source files at base 29592ecc9b.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/UI/Settings_Modules/providers_models_card.py`, `Tests/UI/test_settings_provider_save_scope.py`, `Tests/UI/test_settings_connect_key_rows.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/settings.md`.

