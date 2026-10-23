---
id: TASK-36380
title: Settings Discover models refuses koboldcpp root and prefix-root endpoints that Switch model probes
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 17:44
labels:
- follow-up
- settings
- providers
- model-discovery
priority: low
references:
- tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py
- tldw_chatbook/Chat/provider_endpoint_contract.py
dependencies:
- TASK-35832
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the reviews of TASK-35832 (PR #3213, 2026-10-22). Once TASK-35832 lands, Discover models and Switch model agree on koboldcpp's native route `/api/v1/generate`. They still disagree on two other endpoint shapes the endpoint contract accepts:
- **Bare root.** For `http://localhost:5001`, Switch model probes the root's `/v1/models`. Discover refuses it, because koboldcpp is not in `_BASE_URL_INFERABLE_PROVIDER_KEYS`.
- **Prefix roots.** For `/kobold` or `/API/V1/GENERATE`, the contract reads the path as a prefix root and Switch model probes `<root>/v1/models`. Discover's generic path rules refuse it.

The refusal also shows Settings' generic copy ("This endpoint is not a valid OpenAI-compatible models endpoint…"), even where the contract has a specific reason, such as "Endpoint path has ambiguous API suffixes." So a user can see koboldcpp's models in Switch model but can't save them from Settings, and the message doesn't say why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discover models for koboldcpp lists from the same models URL Switch model probes, for a bare root and for a prefix root, or the task records why a shape stays refused
- [x] #2 Where Discover refuses an endpoint the contract refuses with a reason, the Settings message names that reason
- [x] #3 Other providers' discovery eligibility and models URLs are unchanged
<!-- AC:END -->

## Implementation Plan

1. In `openai_compatible_model_discovery.py`, let the endpoint contract decide koboldcpp's discovery whole, as Switch model does: eligible exactly when `connection_probe_availability` says MODELS_ROUTE, listing at the contract's `models_url`. That covers the bare root, prefix roots and the native route TASK-35832 added, so its native-only special case (and the contract's `is_native_generate_route`, used only there) goes. Other providers keep the generic rules.
2. Expose the contract's refusal reason for koboldcpp, and have the local catalog service (the Settings path) use it as the unsupported-endpoint message instead of the generic copy. The Settings status row is `markup=False`.
3. Tests in `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py` (root, prefix roots, refusal reason) and an assertion in the census-listed koboldcpp journey in `Tests/UI/test_settings_saved_lists_follow_file.py`; red before, green after.
4. Differential probe through run.sh: eligibility and models URL at base vs head for every catalog provider key over a varied URL set; only koboldcpp rows may change.
5. Update Docs/User_Guide/settings.md; run covering tests, ratchets, markup check and preflight.

## Implementation Notes

koboldcpp's Discover models now follows the endpoint contract alone, so for every endpoint that passes discovery's own URL checks it lists exactly where Switch model probes.
- `supports_openai_compatible_model_discovery`: for koboldcpp, once discovery's own URL parse passes, eligibility is `connection_probe_availability("koboldcpp", endpoint) is MODELS_ROUTE`, the decision Switch model makes. `build_models_url` returns the contract's `models_url`. So `http://localhost:5001` lists at `/v1/models`, `/kobold` at `/kobold/v1/models`, `/API/V1/GENERATE` at `/API/V1/GENERATE/v1/models`, and the native route as before.
- This replaces TASK-35832's native-route special case. `_contract_native_resolution` and the contract's `is_native_generate_route` (whose only caller it was) are deleted.
- Other providers keep the generic rules; `_BASE_URL_INFERABLE_PROVIDER_KEYS` is unchanged.
- AC#1, the shapes that stay refused: discovery's own URL parse (`_parse_endpoint`) still runs first and is kept, as a safety check on what discovery requests. It refuses a few shapes the contract accepts and Switch model probes:
  - a request tail that is not final, such as `/chat/completions/x`, `/responses/kobold` or `/models/chat/completions`;
  - a double escape, such as `/%252541`;
  - an endpoint over 2,000 characters (the contract allows 4,096).

  None of these is a route koboldcpp's native send uses. Settings shows the generic copy for them, because the contract gives no reason to name. `test_kobold_shapes_discoverys_url_checks_refuse_stay_refused` pins them.
- Because the contract is now the whole rule, koboldcpp rows the generic rules used to accept also move to the contract's reading:
  - `/models`, `/completion(s)`, `/chat/completions`, `/foo/chat/completions`, `/api/gateway` and `/api/paas/v4` are prefix roots to the contract. They now list at `<path>/v1/models`, which is Switch model's URL.
  - The contract refuses `/v1/openai`, a schemeless remote host (`example.com/v1`), an IPv6 zone id, a query, a fragment, user info and port 0. Discover now refuses them too, and gives the contract's reason.
- AC#2: the new `contract_refusal_reason(provider, endpoint)` returns the contract's first error for a koboldcpp endpoint it refuses. The local catalog service, which is the Settings path, uses that reason as the `unsupported_endpoint` message in place of "This endpoint is not a valid OpenAI-compatible models endpoint." With a reason, the hint is `KOBOLDCPP_ENDPOINT_HINT`: "Use the server's /api/v1/generate route, such as http://localhost:5001/api/v1/generate." Only the native route keeps sending, because `chat_with_kobold` posts KoboldAI's native payload to the saved endpoint. Both the generic "/v1 models endpoint" hint and a bare root would break sending, since Settings saves a root as `<root>/v1/chat/completions` (the contract's `chat_url`). Review round 2 corrected round 1's hint, which offered the bare root. Without a reason, the generic message and hint are unchanged. Settings shows the line in `#settings-model-discovery-status`, a `markup=False` Static, after `redact_secret_text`, so it is markup-safe.
- AC#2 is scoped to koboldcpp. Other providers refuse under discovery's own path rules, where the contract's reason can be wrong. Example: a hyphenated registry key reads "Select a valid provider.".

AC#3 differential probe, through run.sh on a scratch profile:
- Inputs: 126 provider identities (registry keys and aliases, `[providers]` config keys, the contract's URL-based keys, discovery's key sets, template `api_settings`/`providers` keys, and three koboldcpp spellings) × 441 URLs (9 hosts × 43 paths, edge cases, every preset default URL, template endpoints) = 55,566 rows.
- Between base 52b3207ff1 and head, 621 rows changed, all koboldcpp: 207 each for `koboldcpp`, `Koboldcpp` and `KoboldCpp`.
- Per spelling: 114 went from refused to eligible, 42 from eligible to refused, and 51 stayed eligible with a new models URL.
- No row changed for any other provider.
- koboldcpp's Discover agreed with Switch model's contract decision on 234 of 441 URLs at base, and on 441 of 441 at head.

Tests:
- In `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`:
  - `test_kobold_uppercase_generate_is_a_prefix_root_that_generic_rules_refuse` became `test_kobold_root_and_prefix_roots_list_where_switch_model_probes`, a rewrite on purpose: it pins the six literal URLs.
  - New `test_kobold_discovery_lists_exactly_where_switch_model_probes` checks the invariant on 12 endpoints.
  - New `test_a_kobold_endpoint_the_contract_refuses_gives_the_contracts_reason`.
  - New `test_kobold_shapes_discoverys_url_checks_refuse_stay_refused` (5 shapes).
- `Tests/UI/test_settings_default_model_picker.py` gained `test_koboldcpp_root_and_prefix_root_list_where_switch_model_probes`. `scripts/ui_pr_gate_census.txt` lists that file whole, so a PR lane runs the test. It was first written in `test_settings_saved_lists_follow_file.py`, but the census lists that file test by test, so the test would not have run in any lane; review round 1 moved it.
  - It runs the real discovery service on a private profile, with only the transport stubbed. It discovers from a saved root, then from a prefix root typed into Endpoint, and asserts the MockTransport saw exactly the two contract URLs.
  - For the ambiguous `/v1/api/v1/generate`, it waits for that line (`_until`, not a single pause, review round 2) and asserts it whole: "Endpoint path has ambiguous API suffixes. Use the server's /api/v1/generate route, such as http://localhost:5001/api/v1/generate."
- Red before the fix: 13 unit cases failed, for example `assert False is True` for `supports_...('koboldcpp', 'http://127.0.0.1:5001')` and the `/models` URL mismatch. The UI journey failed in `_press_discover` with "This endpoint is not a valid OpenAI-compatible models endpoint. ...".
- Negative control, reverted: with `contract_refusal_reason` returning None, the unit test failed (`assert None == 'Endpoint path has ambiguous API suffixes.'`), and so did the UI journey (the generic copy is shown). In review round 1, with the generic hint forced back, the moved UI test failed on its line ("... ambiguous API suffixes. Configure an explicit http:// or https:// /v1 models endpoint ...").

Runs, all `-n 4` through run.sh:
- Covering set: `Tests/LLM_Provider_Catalog`, the contract, local-server-discovery and switcher-probe tests, the provider preset tests, five Settings discovery UI files, and the boot, import and keystroke census tests. Result: 1269 passed, 10 failed.
  - 9 of the failures fail the same way on dev source: `test_probe_handles_recursive_json_as_bounded_failure`, 4 `test_inference_cloud_presets` registry-allowance tests, and 4 keystroke-census tests.
  - The 10th, `test_settled_400_message_typing_traverses_no_history`, has the same `messages_for_session` signature as its census sibling that fails on dev. It passed alone at head.
- Wider Settings, Console and wizard UI run (16 files): 95 failures (RecoveryRequired, network policy). The same 95 nodes and exception mix (53 AssertionError, 40 RecoveryRequired, 1 AttributeError) occur on dev source.
- `Tests/Architecture` size ratchets, run serially: the screen ratchet passes. 8 module-ratchet rows fail on files this change does not touch, and those files have the same line counts as on dev.
- `check_markup_interpolation.py` reports OK, and `preflight.sh` passes.
- Review round 1 (`-n 4`): the discovery unit file, `test_settings_default_model_picker.py`, `test_settings_saved_lists_follow_file.py` and `test_local_llm_provider_catalog_service.py` gave 318 passed, 1 failed.
  - The failure was `test_an_id_the_save_refuses_is_named_as_the_cause_not_config_access[saved-model]`. It asserts unsaved changes after a single `pilot.pause()` following an Input edit, which is load-dependent.
  - Further `-n 4` runs of the picker file:
    - at head: 1 failure in 4 runs on that node;
    - at head without the new test: 1 run, passed;
    - at dev: 1 of 3 runs failed, on a sibling focus test (`test_an_invalid_custom_id_is_rolled_back_when_the_field_is_left[focus-moves-away]`).
  - The two head runs that passed ran at load averages of 30 to 38 on 18 CPUs.
  - Private-profile tests each run in their own child process with their own profile, so the new test cannot leak state into it.
- Review round 2:
  - The moved test passed 5 of 5 serial runs.
  - With round 1's hint put back temporarily, it failed on its line ("... Use the server's address, such as http://localhost:5001, ...").
  - The picker file and the discovery unit file together (`-n 4`, load average 36): 230 passed.
  - `preflight.sh` and the markup check pass.

Files: `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`, `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py`, `tldw_chatbook/Chat/provider_endpoint_contract.py`, `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`, `Tests/UI/test_settings_default_model_picker.py`, `Docs/User_Guide/settings.md`.

