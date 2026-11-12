---
id: TASK-38963
title: "Discovery accepts hosts the endpoint contract refuses, and Console display and compare refuse single-label hosts it accepts"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: '2026-11-06 07:47'
labels:
- follow-up
- providers
- console
- settings
priority: medium
references:
- tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py
- tldw_chatbook/Chat/console_provider_endpoints.py
- tldw_chatbook/UI/Console_Modules/model_switcher.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- Tests/Chat/test_local_server_discovery.py
dependencies:
- TASK-36517
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-36516/36517 branch (the state after it merges), 2026-11-05. TASK-36517's notes list both halves as follow-ups for the coordinator to file. Line numbers are from that branch.

**Discovery accepts hosts that the save refuses.**
- Discover models parses the URL itself (`openai_compatible_model_discovery.py:225-259`, then `validate_url`), apart from the contract's host rule (`valid_dns_label`).
- A probe at base tried `-kobold`, `kob*old`, `1.2.3` and a 64-character label, each on `:8000/v1`, for both vllm and custom:
  - `supports_openai_compatible_model_discovery(...)` returns True;
  - `resolve_provider_endpoint(...).errors` is `('Endpoint URL must include a valid host and port.',)`.
- The TASK-36517 branch does not touch discovery and keeps these contract rules, so the split remains after it merges. Discover lists models from an endpoint that Save then refuses.

**Console display and compare reject single-label hosts that the contract accepts.**
- `_parse_http_endpoint` (`tldw_chatbook/Chat/console_provider_endpoints.py:393-420`) calls `_is_allowed_endpoint_host` (`:423-435`), which accepts localhost, an IP or a dotted name (`_is_dotted_dns_name`, `:438-442`, requires a `.`). Otherwise it accepts a host only when `not has_scheme and port is not None`.
- So `http://kobold:5001` and `http://kobold_cpp:5001` both come out as `'invalid endpoint'`, from `safe_endpoint_display` (`:332-362`) and `normalize_generic_endpoint_for_compare` (`:365-379`). Without the scheme, `kobold:5001` displays as itself.
- A probe at base:
  - `generic_endpoint_differs("http://other:5001", {"api_url": "http://kobold:5001"})` returns False;
  - `adoption_endpoints_differ("koboldcpp", "http://kobold:5001", "http://other:5001")` returns False.
- Effects:
  - The gateway's unsaved-endpoint guard (`console_provider_gateway.py:4950-4962`) treats two different single-label servers as the same endpoint, and so does the Settings unsaved check (`settings_screen.py:14543`).
  - The adoption toast (`model_switcher.py:949-955`) interpolates `safe_endpoint_display(configured_endpoint)` and reads "Keeping the saved endpoint invalid endpoint for KoboldCpp; …".
  - Single-label service names are the Docker Compose case TASK-36517 set out to support.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For every host in TASK-36517's two pinned sets, plus single-label hosts with and without a scheme, Discover models eligibility and the provider save make the same accept/refuse decision; a test is red on dev for -kobold, kob*old, 1.2.3 and a 64-character label
- [x] #2 http://kobold:5001 and http://kobold_cpp:5001 display as themselves, and two different single-label hosts compare as different in generic_endpoint_differs and adoption_endpoints_differ; a test is red on dev ('invalid endpoint', and False)
- [x] #3 With a saved single-label endpoint and a different single-label session endpoint, the Console unsaved-endpoint guard blocks the send, and the adoption toast names the saved endpoint rather than "invalid endpoint"; both are shown red on dev
- [x] #4 TASK-36517's refusal tests and the consent-key digest test stay green
<!-- AC:END -->

## Implementation Plan

1. Contract (`Provider_Contract/provider_endpoint_contract.py`): make the host rule public (`canonical_endpoint_host`, today's private `_canonical_host`) and the schemeless-host rule (`is_exact_schemeless_host`), so callers outside the contract decide with the exact rules the save uses.
2. Discovery (`openai_compatible_model_discovery.py`): `_parse_endpoint` refuses a host the contract refuses and a schemeless non-loopback endpoint (the save refuses both); the QwenCloud branch applies the same host rule to its normalized base. The bracketed IPv6 zone id stays as discovery already pins it.
3. Console display and compare (`console_provider_endpoints.py`): `_is_allowed_endpoint_host` uses the contract's host rule. A single-label host other than localhost still needs a port, now with or without a scheme, so a pasted token never echoes (`test_safe_endpoint_display_rejects_single_label_token_with_scheme`). `safe_endpoint_display` also reads koboldcpp's native route through the contract, so the adoption toast names a saved KoboldCpp endpoint.
4. Tests in `Tests/Chat/test_local_server_discovery.py` (dreams-boundary lane, whole file): the discovery/save matrix over TASK-36517's two host sets plus single-label hosts, with and without a scheme; display/compare of single-label hosts; the gateway's unsaved-endpoint guard and the adoption toast. Each shown red on base first.
5. Covering runs, task notes.

## Implementation Notes

Discover models, the provider save, and Console's endpoint display and compare now decide hosts by one rule, the endpoint contract's.
- Contract (`provider_endpoint_contract.py`): the private `_canonical_host` and `_is_exact_schemeless_host` are public as `canonical_endpoint_host` and `is_exact_schemeless_host`, with docstrings. Their rules are unchanged; `valid_dns_label` stays the label rule inside `canonical_endpoint_host`.
- Discovery (`openai_compatible_model_discovery.py`): `_parse_endpoint` refuses a host `canonical_endpoint_host` refuses and an endpoint typed without a scheme unless its host is loopback, the two rules the save applies. Its docstring already said it accepted only "host-only local URLs". Such an endpoint is now "malformed" (not "unsupported"), so Discover's message asks for a full http:// or https:// address. QwenCloud's branch applies the same host rule to its normalized base (it accepted `kob..old`). A bracketed IPv6 zone id stays allowed, as `test_non_qwen_discovery_preserves_scoped_ipv6_zone_id_support` pins.
- Console (`console_provider_endpoints.py`): `_is_allowed_endpoint_host` is `canonical_endpoint_host` plus one display rule. A single-label host other than localhost needs a port, with or without a scheme, so a pasted key is never echoed (`test_safe_endpoint_display_rejects_single_label_token_with_scheme` and its two siblings pin this). `_is_dotted_dns_name` is gone. IP literals are still accepted first, so a zone id still compares as itself.
- AC#3's KoboldCpp toast names a native `/api/v1/generate` endpoint. `safe_endpoint_display` read every endpoint as Custom's, and Custom refuses that route, so even `http://127.0.0.1:5001/api/v1/generate` displayed as "invalid endpoint". The new `_display_resolution` falls back to koboldcpp's reading, which adds only the native route. The unsaved-endpoint copy now names both addresses for KoboldCpp too.
- Tests, all in `Tests/Chat/test_local_server_discovery.py` (the dreams-boundary step runs it whole). TASK-36517's refused-host list became the shared `_HOSTS_REFUSED_FOR_ANOTHER_REASON`, with the same 14 hosts.
  - `test_discover_models_accepts_exactly_the_hosts_the_save_accepts`: 18 hosts × {scheme, no scheme} × the 19 contract providers. The hosts are the two underscore-set spellings, the 14 refused hosts, `kobold` and `kobold_cpp`.
  - `test_a_single_label_host_displays_and_compares_as_itself`: AC#2's display and the two compare helpers, plus the token pin.
  - `test_a_different_single_label_session_endpoint_is_blocked_as_unsaved`: `ConsoleProviderGateway.resolve_for_send`; the same endpoint stays ready.
  - `test_the_adoption_toast_names_a_saved_single_label_endpoint`: KoboldCpp and llama.cpp, through `adopt_detected_local_server` on a stand-in screen with the config writer patched.
- Red at base source (the tests above with the four source files unchanged, run.sh, `-n 4`): 30 failed, 10 passed. Discovery failed for `-kobold`, `kob*old`, `1.2.3`, the 64-character label and every schemeless remote host, e.g. `('aphrodite', 'http://1.2.3:5001/v1', False)`, and qwencloud failed for `kob..old`. Display read `'invalid endpoint' == 'http://kobold:5001'`. The guard: `assert True is False` (ready). Both toasts read "invalid endpoint". Green after: 134 passed, the whole file.
- AC#4: `test_hosts_refused_for_another_reason_are_still_refused` and `test_an_underscore_host_gets_a_project_instruction_key_and_its_origin_label` are in that green run.
- Covering runs (`-n 4`, run.sh), each compared with the same command at `fu-base-r10`:
  - 28 targets across Chat, LLM_Calls, LLM_Provider_Catalog, Backup_Recovery and Wizards (load 13-18): 2656 passed, 11 failed. The same 11 failed at base (2616 passed). The 40 extra passes are the new tests.
  - 21 Console, Settings and wizard UI files (load 25-72; this run also held the TASK-38964..38966 changes): 1285 passed, 19 failed, against 1280 passed and 22 failed at base. One failure is new here, `test_providers_models_state_line_names_its_scope_in_one_row`. It then passed 3 of 3 alone at load 48 and in the whole-file run. Three base failures passed here. All four are load-timing failures.
- Docs: `Docs/User_Guide/settings.md` (Discover models reads the host as **s** does) and `Docs/User_Guide/console.md` (the adoption notice names the address; single-label endpoints compare as distinct servers).
Review r10e-rev1:
- Single-label hosts without a port still merged in compare. `http://kobold/v1` and `http://other/v1` compared as one sentinel, the guard let a vllm or koboldcpp session on another such server send (`ready=True`), and the KoboldCpp toast for `http://kobold_cpp/api/v1/generate` read "invalid endpoint".
  - `normalize_generic_endpoint_for_compare` now asks the contract's identity (`canonical_connection_identity`) before the display rule, so two different endpoints never share the sentinel.
  - Display accepts a single label with a port or a path. A bare `http://kobold` still displays as "invalid endpoint" because it could be a pasted key, but it compares by its identity. The three token pins in `Tests/Chat/test_console_provider_endpoints.py` hold.
  - Tests: port-less pairs (with a path, the native route, bare) in `test_a_single_label_host_displays_and_compares_as_itself`; a koboldcpp and a vllm port-less pair in the now parametrized gateway test; a port-less KoboldCpp toast case. Red first: `'invalid endpoint' == 'http://kobold/v1'`, the guard `assert True is False` twice, and the toast read "invalid endpoint". Green after.
- Discover's copy for a refused host asked for an http:// /v1 endpoint the URL already was. `contract_refusal_reason` now gives the contract's first error for every provider whose key the contract knows, not only koboldcpp. The catalog service shows it, with koboldcpp's route hint for koboldcpp and no hint otherwise.
  - New `test_discover_names_the_contracts_reason_for_a_refused_host` (vllm, custom): red, `'This endpoin...els endpoint.' == 'Endpoint URL...ost and port.'`.
  - `test_a_kobold_endpoint_the_contract_refuses_gives_the_contracts_reason` pinned `reason("vllm", ambiguous) is None`; it is rewritten on purpose to the contract's reason, plus a `None` case for an accepted vllm endpoint.
- `Docs/User_Guide/settings.md` no longer says Discover lists "only" from what **s** saves. It names the zone-id exception: for providers other than KoboldCpp and QwenCloud, a probe listed `http://[fe80::1%25en0]:5001/v1` for all 14 others. It also says Discover now gives the reason **s** gives. `Docs/User_Guide/console.md` says what is shown: a service name with a port or a path names itself, and a bare name reads "invalid endpoint" but compares as itself.
- `test_providers_models_state_line_names_its_scope_in_one_row` (1 in 20 red at its first `content_region` read) waits for the banner's first layout (`banner.size.width > 0`). It then passed 10 of 10 at `-n 4`, load 16-29.
- Runs after the review (run.sh):
  - Changed files: 827 passed.
  - The 28-target set: 2675 passed, 10 failed, all 10 among base's 11 (the 11th was the QwenCloud test TASK-38966 fixed).
  - The 21-file UI set: 1286 passed, 18 failed, all 18 among base's 22.
  - `Tests/CI` with the size ratchets: 1416 passed, 1 skipped. Preflight passed. `bootbudget.sh`: 32 passed.

Review r10e-rev2:
- rev1's path allowance ignored whether a scheme was typed, so a pasted key with a "/" displayed and compared as itself. Examples: `sk-abc123def456/extra`, `unit-test-key/v1`, an AWS-style secret, in the unsaved-endpoint copy and the adoption toast. At base these read "invalid endpoint". The path now counts only after a typed `http://` or `https://`, and `http://kobold/v1` still displays as itself.
- New `test_a_schemeless_token_with_a_slash_is_never_echoed` (`Tests/Chat/test_console_provider_endpoints.py`, 4 keys: display, compare and the unsaved copy) and one such case in the laned display test. Red at ec13dd49ef: 5 failed, e.g. `'sk-abc123def456/extra' == 'invalid endpoint'`. Green after.
- Display helpers that bypass `_is_allowed_endpoint_host`: `local_server_discovery.endpoint_display` delegates to `safe_endpoint_display`. Discovery's `fingerprint_endpoint` echoes such keys (`http://sk-abc123def456/extra`), but it did so identically at base, so it is listed as a follow-up.
- `contract_refusal_reason`'s docstring now says it returns None for a missing or malformed key; any well-formed key is resolved.
- Runs: the endpoint-display, discovery, contract and LLM_Provider_Catalog files with `Tests/CI`, 2029 passed, 1 skipped. Preflight passed.

Review r10e-rev3:
- `test_a_schemeless_token_with_a_slash_is_never_echoed` moved, unchanged, from `Tests/Chat/test_console_provider_endpoints.py` (no PR lane) to `Tests/Chat/test_local_server_discovery.py` (dreams-boundary, whole file). It keeps its display, compare and unsaved-copy checks.
- Fold-in, present at base too: discovery's `fingerprint_endpoint` turned a key pasted into an endpoint field without a scheme into `http://<key>/`. `refresh_stale_configured_providers` logged it at INFO ("auto-refresh failed for … (fingerprint …)"), past the file sink's redaction, on every launch.
  - An endpoint that discovery and the contract (read as Custom) both refuse now fingerprints to `[invalid-endpoint]`, keeping only a typed http, https or ftp scheme.
  - The log line no longer carries the endpoint at all.
  - Three pins were rewritten on purpose to the stricter sentinel: the ftp, schemeless malformed-port and https malformed-port fingerprints. Each still checks that no secret appears.
  - The production diagnostic inventory's row for `local_llm_provider_catalog_service.py` was re-pinned after reading the statement diff (`--statements --since 453cebcc67`). The one removed statement interpolated the fingerprint. The one added statement interpolates only the catalog key and the error kind.
  - Other places: grep found no other log or display of `endpoint_fingerprint` or of a raw configured endpoint in `LLM_Provider_Catalog`. The plugin model reference digests the fingerprint, and the disk cache stores it for successful listings only.
  - Tests in the laned file: `test_a_refused_endpoints_fingerprint_never_carries_a_pasted_key` (10 key shapes, from the reviewer's probes 8 and 9) and `test_a_failed_catalog_refresh_logs_no_part_of_a_pasted_key` (MistralAI, 2 shapes, every 8-character window of the key). Red at abe944955d: 10 fingerprints read `http://<key>/`, and both logs held the key. Green after.

- Proposed follow-ups, not changed here: Discover models still accepts a bracketed IPv6 zone id that the save refuses (pinned on purpose by discovery's test), and QwenCloud's Discover refuses a schemeless loopback endpoint (`localhost:5001/v1`) that the save accepts. Console display (and the Chat settings preview) still show a schemeless key that contains a dot, such as a ZAI `<32 hex>.<16>` key, as a dotted host, as at base.

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`, `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py`, `tldw_chatbook/Chat/console_provider_endpoints.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`, `Tests/UI/test_settings_provider_save_scope.py`, `Docs/User_Guide/settings.md`, `Docs/User_Guide/console.md`.
