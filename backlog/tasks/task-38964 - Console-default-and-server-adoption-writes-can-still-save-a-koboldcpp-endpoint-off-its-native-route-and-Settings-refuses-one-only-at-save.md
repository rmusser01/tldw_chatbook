---
id: TASK-38964
title: "Console default and server-adoption writes can still save a koboldcpp endpoint off its native route, and Settings refuses one only at save"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: '2026-11-06 08:31'
labels:
- follow-up
- providers
- console
- settings
priority: medium
references:
- tldw_chatbook/UI/Wizards/first_run_setup_state.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/Chat/console_settings_defaults.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Wizards/first_run_provider_detection.py
- Tests/UI/test_settings_provider_save_scope.py
dependencies:
- TASK-36516
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-36516/36517 branch (the state after it merges), 2026-11-05; line numbers are from that branch. TASK-36516 made Settings refuse a koboldcpp endpoint that is not on the native route. The predicate it uses, `first_run_endpoint_refusal`, lives in the wizard module (`tldw_chatbook/UI/Wizards/first_run_setup_state.py:357-365`, `endswith("/api/v1/generate")`), and Settings imports it from there lazily (`settings_screen.py:16132-16146`). Two Console write paths never call it:

- **Console ▸ Settings modal ▸ "Make default for new chats".**
  - `_endpoint_patch_is_authorized` (`tldw_chatbook/Chat/console_settings_defaults.py:1095-1110`) checks the action, the field mask, the dirty and checked flags, the provider, and `parse_console_endpoint_preview` (authority and network class).
  - `:1254-1257` then writes the raw `patch.value.strip()` to the koboldcpp section of `api_settings`. Nothing in the file checks the route.
- **Model-switcher server adoption.**
  - `model_switcher.py:959` writes `provider_values["api_url"] = server.base_url` with no route check.
  - koboldcpp discovery candidates come only from the koboldcpp section's own endpoints (`local_server_discovery.py:431-445`).
  - `normalize_probe_base_url("http://127.0.0.1:5001/api/v1/generate")` is None, while a bare root normalizes to itself. So adoption can only ever write back a root that is not the native route.

A non-native endpoint then fails at send. `chat_with_kobold` (`LLM_API_Calls_Local.py:965`) posts the native payload to the saved URL. An OpenAI-style route answers with `choices`, and the send fails with "Unexpected response structure" (`:1216-1224`).

Settings also shows the refusal later than the wizard does:
- Its endpoint `Input.Changed` handler (`settings_screen.py:30621-30653`) only stages the value, and the refusal runs only inside Save's `connection_dirty` branch (`:33380-33392`).
- The wizard's `_refresh_endpoint_status` (`first_run_provider_detection.py:366-411`) shows "✗ KoboldCpp needs its native /api/v1/generate address…" while the user types.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One koboldcpp native-route rule lives in Provider_Contract and the first-run wizard, Settings, the Console defaults write and the model-switcher adoption all use it; no copy of the /api/v1/generate check remains outside it
- [x] #2 Console "Make default for new chats" with a koboldcpp endpoint off the native route is refused with the native-route copy and writes nothing; a test is red on dev (the root is written)
- [x] #3 Model-switcher server adoption never writes a non-native koboldcpp api_url (it keeps the saved endpoint or uses the native route); a test is red on dev
- [x] #4 Settings shows the native-route refusal on the endpoint row while the user types, as the wizard does, and still refuses at save; a test types a bare root and reads the row before any save, and is red on dev
- [x] #5 A koboldcpp endpoint already on /api/v1/generate saves unchanged on every path, and other providers' writes are unchanged
<!-- AC:END -->

## Implementation Plan

1. Contract: add `koboldcpp_route_refusal(provider, endpoint)` (the wizard's copy and its `/api/v1/generate` test, moved from `first_run_setup_state.first_run_endpoint_refusal`) and `koboldcpp_native_endpoint(endpoint)` (the native route at an endpoint's root). It takes an endpoint or a chat URL alike, so the wizard's callers pass what they pass today.
2. Wizard: its four callers and the Settings save use the contract's predicate; delete `first_run_endpoint_refusal` and Settings' `_provider_endpoint_route_refusal` wrapper.
3. Console defaults: `_endpoint_patch_is_authorized` refuses a koboldcpp patch off the native route, so `apply_console_default_intent` writes nothing; Chat settings' Make default shows the refusal copy before it submits.
4. Model-switcher adoption: a detected koboldcpp root is adopted as its native route, for the config write and for this chat's endpoint.
5. Settings: the Endpoint row's help shows the refusal while the field holds a koboldcpp endpoint off the route; Save still refuses.
6. Tests red on base: the Console defaults write (pure, in `Tests/Chat/test_local_server_discovery.py`), Chat settings' Make default copy (`Tests/UI/test_console_settings_saved_defaults.py`), the adoption write (`Tests/Chat/test_local_server_discovery.py`), and the Settings row while typing (`Tests/UI/test_settings_provider_save_scope.py`). Native routes save unchanged on each path.

## Implementation Notes

KoboldCpp's native-route rule lives in `Provider_Contract`, and every path that writes a koboldcpp endpoint uses it.
- Contract (`provider_endpoint_contract.py`):
  - `koboldcpp_route_refusal(provider, endpoint)` returns `KOBOLDCPP_NATIVE_ROUTE_REFUSAL` (the wizard's copy) for a koboldcpp endpoint the contract reads off `/api/v1/generate`. It returns "" for one on the route, for one the contract refuses outright and for any other provider. It takes the endpoint or the chat URL the contract resolved it to: a chat URL resolves to itself, so the wizard's callers pass what they passed before.
  - `koboldcpp_native_endpoint(endpoint)` returns the native route at the endpoint's root.
  - The path is spelled once, as `_KOBOLD_NATIVE_PATH`.
- Callers:
  - Wizard: `first_run_setup_state.first_run_endpoint_refusal` is deleted. Its four callers (`first_run_setup_state`, `first_run_provider_step`, and twice in `first_run_provider_detection`) call the contract.
  - Settings: the `_provider_endpoint_route_refusal` wrapper is deleted, and the save calls the contract (lazily, as before).
  - Console defaults: `_endpoint_patch_is_authorized` (`console_settings_defaults.py`) refuses an off-route koboldcpp patch, so `apply_console_default_intent` fails BEFORE_REPLACE and writes nothing.
  - Chat settings' `_submit` (`console_settings_modal.py`) shows the refusal in its error banner and submits nothing for Make default when the endpoint it would save is off the route. That is the copy AC#2 asks for. The pure layer still refuses for any other caller.
  - Model-switcher adoption: discovery finds koboldcpp only at a root, so `adopt_detected_local_server` adopts that root as its native route, for the `api_url` write and for this chat's endpoint (session or session-only), so the chat's sends post to the same route. The keep-saved compare still uses the detected root, so the same server is still recognized.
- AC#4: `endpoint_row_copy` (`providers_models_card.py`) returns the refusal as the Endpoint row's help while the field holds an off-route koboldcpp endpoint, so the row says it as the user types; Save still refuses.
- Tests, each red with the callers' changes reverted (the new contract functions kept so the imports resolve), green after:
  - `test_make_default_writes_a_koboldcpp_endpoint_only_on_its_native_route` (`Tests/Chat/test_local_server_discovery.py`, dreams-boundary): a scratch config through `select_config_source`. Root, `/v1` and `/v1/chat/completions` are refused and the file is byte-identical; the native route is written unchanged. Red: `assert True is False` (`file_replaced`) for the three off-route values.
  - `test_saving_a_koboldcpp_root_as_the_default_is_refused_with_its_copy` (`Tests/UI/test_console_settings_saved_defaults.py`, whole-file census entry): Save endpoint & use model on a root shows the copy and submits nothing; a native route then submits unchanged. Red: the root was submitted (`assert [ConsoleSettingsCommittedSubmission(...)] == []`).
  - `test_server_adoption_writes_koboldcpp_s_native_route` (4 configured shapes, including a `/kobold` prefix root) and `test_server_adoption_writes_other_providers_detected_endpoint_unchanged` (llama.cpp, AC#5), on the stand-in screen from TASK-38963. Red: `assert 'http://127.0.0.1:5001' == 'http://127.0...i/v1/generate'`.
  - `test_the_endpoint_row_refuses_a_koboldcpp_root_as_it_is_typed` (`Tests/UI/test_settings_provider_save_scope.py`, whole-file census entry, private profile): types the root key by key, reads the row before any save, types the route, reads the requirement, then saves the root and gets the refusal. Red: `assert "the server's...erate address" == 'KoboldCpp ne...stom instead.'`.
  - `test_koboldcpp_route_rule_lives_in_the_contract_alone` (AC#1): the contract's decisions, and no `endswith("/api/v1/generate")` or `first_run_endpoint_refusal` left under `tldw_chatbook/`. Red: it named the three wizard files.
- AC#5: TASK-36516's `test_koboldcpp_saves_only_the_native_route_its_send_posts_to`, the wizard's `test_kobold_wizard_draft_refuses_non_native_routes` and `test_kobold_wizard_save_and_dispatch_use_native_url_with_prompt`, and the native cases above stay green. Other providers' writes go through unchanged code (every new check returns "" for them), and the llama.cpp adoption test pins that.
- Covering runs (`-n 4`, run.sh):
  - The UI run in TASK-38963's notes, which held this change.
  - The wizard provider files (review contracts, review native, detection, discovery state, endpoints, check, skip, setup integration) with the discovery, QwenCloud, TLS, Console defaults, contract and Console endpoint files: 693 passed, 1 failed, 17 errors. The same 1 failure (`test_nothing_is_an_error_before_the_user_acts`) and 17 `RecoveryRequired` errors (`test_first_run_setup_integration.py`, TASK-38111) occur at base.
  - `Tests/Architecture` size ratchets: 205 passed.
- Ratchet rows set to the measured size, each with a dated comment (owner waiver): `providers_models_card.py` 2,401 -> 2,405, `console_settings_modal.py` 6,176 -> 6,191, `first_run_provider_step.py` 1,690 -> 1,691. `settings_screen.py` shrank, so its row is lowered: 35,669 -> 35,654.
- Review r10e-rev1: `koboldcpp_native_endpoint` read a root at KoboldAI's `/api` base as a prefix. A detected `…:5001/api`, from a configured `/api`, `/api/` or `/api/v1`, became `…/api/api/v1/generate`. It now drops a trailing `/api` path segment (by segment, so a host named `api` is untouched) before adding the route, and returns an endpoint already on the route unchanged. The three shapes are added to `test_server_adoption_writes_koboldcpp_s_native_route`, which now states each expected route. Red first: three `…/api/api/v1/generate` mismatches. Green after.
- Review r10e-rev3 (fold-in, every endpoint write):
  - The Console default writer authorized any authority `parse_console_endpoint_preview` parsed, and that took a bare key for a "Remote/unknown" host. For llama.cpp, readiness did not stop it either, so at abe944955d Make default wrote 8 of the 10 key shapes and `gpu-box.lan:8000/v1` to `api_url`. That is 9 red cases of `test_make_default_never_writes_an_endpoint_the_contract_refuses` (5 providers × 12 endpoints, laned file). The other 4 providers already failed readiness.
  - `_endpoint_patch_is_authorized` now also needs `_contract_accepts_authority`: the contract's scheme rule (`is_exact_schemeless_host`) and host rule (`canonical_endpoint_host`).
  - The preview takes Console display's rule: a single label other than localhost needs a port, or a path after a typed scheme. So Chat settings' "Not written to disk" summary no longer names a bare key (`test_the_default_recovery_summary_never_names_a_pasted_key`, red for 7 shapes; the other two are over 63 characters and were already refused).
  - The preview pins in `test_console_settings_defaults.py` stay green, and so do the 6 Console settings UI files. Their 1 failure, `test_shared_evidence_change_refreshes_the_console_once`, fails at base too.
- Review r10e-rev4:
  - rev3's `_contract_accepts_authority` copied only the contract's scheme and host rules. The writer still took 9 shapes **s** refuses (the reviewer's probe16): a query, a fragment, `..`, `//`, `%2F`, port 0 and three ambiguous paths. Chat settings let them through, so Save endpoint & use model ended at "Not written to disk", and Retry refused again while the guard still asked for Save endpoint.
  - The helper is replaced by `console_default_endpoint_refusal(provider, endpoint)`: the contract's first error (`resolve_provider_endpoint`; a registry id reads as Custom), then koboldcpp's route rule. `_endpoint_patch_is_authorized` and Chat settings' `_submit` both use it, so the modal shows the contract's own reason and submits nothing.
  - The reviewer's llama.cpp shapes (`gpu-box.lan:8000/v1`, `192.168.1.20:8080`) never reach Save endpoint & use model. llama.cpp's direct path raises no `endpoint_not_saved` blocker, so its footer offers Save as model default, which carries no endpoint patch. The reachable loop is vllm or ollama with a refused path or query, which reads `endpoint_not_saved`.
  - Tests:
    - `test_make_default_never_writes_an_endpoint_the_contract_refuses` gained the 9 shapes and a check that the writer and **s** decide alike. 41 cases were red at 632197c702 (`file_replaced` True).
    - New `test_saving_an_endpoint_the_contract_refuses_shows_its_reason` (`Tests/UI/test_console_settings_saved_defaults.py`; vllm with a query holding a key, and an ambiguous path) shows the reason, never the key, and submits once fixed. Red: the endpoint was submitted.
    - probe16 now reports no mismatch for 17 shapes × 4 providers.
    - `test_make_default_atomically_patches_profile_globals_and_checked_endpoint` wrote `https://new.example.test:8443/v1?ignored=yes`. It is rewritten on purpose without the query; the query case is pinned as refused.
  - `Docs/User_Guide/console.md` now says the action shows the reason **s** gives.
- Docs: `Docs/User_Guide/settings.md` (the row says the refusal as you type) and `Docs/User_Guide/console.md` (Save endpoint & use model refuses a KoboldCpp root; the Get started card saves KoboldCpp's native route).

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `tldw_chatbook/Chat/console_settings_defaults.py`, `tldw_chatbook/Widgets/Console/console_settings_modal.py`, `tldw_chatbook/UI/Console_Modules/model_switcher.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/UI/Settings_Modules/providers_models_card.py`, `tldw_chatbook/UI/Wizards/first_run_setup_state.py`, `tldw_chatbook/UI/Wizards/first_run_provider_step.py`, `tldw_chatbook/UI/Wizards/first_run_provider_detection.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/UI/test_console_settings_saved_defaults.py`, `Tests/UI/test_settings_provider_save_scope.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/settings.md`, `Docs/User_Guide/console.md`.
