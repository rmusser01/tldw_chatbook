---
id: TASK-39042
title: "Console reads an endpoint typed without a scheme by its own rules, so Chat settings refuses localhost:8080 that Settings saves, readies llama.cpp on a remote host the save refuses, and shows a pasted dotted key as a host"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 15:49
labels:
- follow-up
- console
- providers
- settings
priority: medium
references:
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Chat/console_provider_endpoints.py
- tldw_chatbook/Chat/console_settings_defaults.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- Tests/Chat/test_local_server_discovery.py
- Tests/Chat/test_console_settings_defaults.py
- Docs/User_Guide/console.md
dependencies:
- TASK-38963
- TASK-38964
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on endpoint PR #3432 (head 760d2987c7), which lands TASK-38963 and TASK-38964. Every probe result below is the same on both; line numbers are dev's, and ones marked #3432 are from that head. Probes ran without network or config writes; the key is synthetic.

**Readiness has its own scheme rule.**
- Console readiness checks the session endpoint with `_valid_base_url` (`console_session_settings.py:3199-3208`, called at `:1802-1808`). It runs `validate_url`, which needs a typed scheme, except for llama.cpp, whose value `normalize_llamacpp_base_url` rewrites first.
- vllm, ollama and custom with `localhost:8080` read "Invalid URL" (blocker `endpoint_invalid`, "Endpoint must be a valid http(s) URL."), so Chat settings cannot save it. **s** saves `http://localhost:8080/v1/chat/completions` for the same input.
- llama.cpp reads Ready for `gpu-box.lan:8080`, `192.168.1.20:8080` and a pasted dotted key. The contract refuses all three: "Remote endpoint URLs must include an HTTP or HTTPS scheme." That is the rule TASK-16200 AC#4 set ("remote schemeless endpoints remain blocked").
- TASK-38964's new `console_default_endpoint_refusal` (#3432 `console_settings_defaults.py:1119-1140`) does not refuse any of these, so this is not a regression from that PR.

**A pasted key with a dot displays as a host.**
- `safe_endpoint_display`'s fallback for a value without a scheme (`console_provider_endpoints.py:355-361`; #3432 :355-358) returns the typed text whenever `http://<value>` resolves. `_is_allowed_endpoint_host` (`:423`; #3432 :440) accepts any dotted host.
- `parse_console_endpoint_preview` (`console_settings_defaults.py:1396`; #3432 :1425) accepts any valid dotted hostname without a scheme.
- With the synthetic key `0123456789abcdef0123456789abcdef.ABCDEFGHijklmnop` (a ZAI-shaped `<32 hex>.<16>`), the key is shown lower-cased by:
  - `safe_endpoint_display` and `local_server_discovery.endpoint_display`;
  - the unsaved-endpoint copy ("Selected endpoint: 0123…");
  - `format_console_endpoint_preview`;
  - the Chat settings Endpoint row for llama.cpp (`console_session_settings.py:3388-3404`);
  - Settings' Test-result Endpoint row (`settings_screen.py:17184`; found by reading the code, not mounted).
- The same happens with `/v1` appended and with an `sk-` prefix.
- TASK-38963 already stopped discovery's `fingerprint_endpoint` from echoing it (`[invalid-endpoint]`); on dev it reads `http://0123…/`.
- A value with a typed `http://` cannot be told from a real URL and is out of scope.

**Pins a fix must respect.**
- `local-service:9090/v1` displays as itself (`Tests/Chat/test_console_provider_endpoints.py:91`).
- `("printer.local/api", "printer.local", "LAN")` (`Tests/Chat/test_console_settings_defaults.py:1635`) contradicts a rule that a non-loopback value without a scheme needs a port.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 vLLM, Ollama and Custom take `localhost:8080`, `127.0.0.1:8080/v1` and `[::1]:8080/v1` in Chat settings as **s** does: readiness is not `endpoint_invalid`, and Save endpoint & use model writes the contract's persisted endpoint; a test is red on dev (`label == "Invalid URL"`)
- [x] #2 llama.cpp readiness is not Ready for a non-loopback session endpoint typed without a scheme (`gpu-box.lan:8080`, `192.168.1.20:8080`, the synthetic dotted key); a test is red on dev (`label == "Ready"`)
- [x] #3 A non-loopback value typed without a scheme or a port is never shown: for the synthetic dotted key, with and without `/v1` and with an `sk-` prefix, `safe_endpoint_display`, the unsaved-endpoint copy, the Chat settings Endpoint row and discovery's `endpoint_display` read "invalid endpoint", and `parse_console_endpoint_preview` returns None; a test in `Tests/Chat/test_local_server_discovery.py` is red on dev (the key is echoed)
- [x] #4 Values with a typed scheme, loopback values without one, and `local-service:9090/v1` display as before; any pin the new rule contradicts (such as `printer.local/api`) is rewritten on purpose and named in the notes
- [x] #5 `Docs/User_Guide/console.md` describes the behaviour
<!-- AC:END -->

## Implementation Plan

1. Tests first in `Tests/Chat/test_local_server_discovery.py` (laned): Chat settings readiness and validation for vLLM, Ollama and Custom with the three loopback spellings (AC#1); the Console default write compared with what **s** writes (AC#1, extending TASK-38964's writer test); llama.cpp readiness for remote schemeless values (AC#2); the dotted key on every display surface and the preview (AC#3). Show them red at base.
2. Readiness (`_valid_base_url`): the contract's scheme rule, `is_exact_schemeless_host`; such a value is read as `http://`, any other schemeless value is invalid.
3. Console default write: write what **s** writes (`_persisted_endpoint_for_owner`, made public), in the readiness preview and in the config.
4. Display: `safe_endpoint_display` and `parse_console_endpoint_preview` refuse a schemeless value that has no port and is not loopback.
5. Rewrite the `printer.local/api` preview pin on purpose; update `Docs/User_Guide/console.md`.

## Implementation Notes

Console now reads an endpoint typed without a scheme by the endpoint contract's rule, writes what **s** writes, and never shows a schemeless, port-less remote value.
- Readiness (`console_session_settings._valid_base_url`): a value without `://` is valid only when `is_exact_schemeless_host` accepts its netloc (localhost, 127.0.0.1, [::1]); it is then read as `http://`. Any other schemeless value is `endpoint_invalid`, so llama.cpp no longer reaches Ready by way of `normalize_llamacpp_base_url` adding `http://` to a remote host.
- Save endpoint & use model (`console_settings_defaults._build_locked_default_mutation`): it writes `persisted_endpoint_for_key`, a new public wrapper in `provider_setup_persistence.py` around the save's own `_persisted_endpoint_for_owner`. Before, it wrote the typed value, so `localhost:8080` would have reached the config without a scheme. A registry entry (`custom-ep:<slug>`) writes the contract's `normalized_input` (adds the scheme; `http://192.168.1.9:8080` is unchanged), because the registry's `validate_entry` refuses a schemeless `base_url`.
- Display (`console_provider_endpoints.safe_endpoint_display`) and preview (`parse_console_endpoint_preview`): a value typed without a scheme needs a port or a loopback host. The unsaved-endpoint copy, discovery's `endpoint_display` and Settings' Test-result Endpoint row all go through `safe_endpoint_display`. Compare is unchanged.
- AC#3, deviation in one surface: the criterion says the Chat settings Endpoint row reads "invalid endpoint". With AC#2's fix, readiness for these llama.cpp values is `endpoint_invalid`, and `_format_endpoint_summary_row` gives that blocker "Endpoint: Not configured", as for any invalid endpoint (pinned by `test_summary_builder_reports_only_genuine_provider_endpoint_inheritance`). The row never shows the key; the test asserts that copy.
- Pins rewritten on purpose:
  - `test_endpoint_preview_exposes_only_authority_and_network_class` (`Tests/Chat/test_console_settings_defaults.py`): `("printer.local/api", "printer.local", "LAN")` became `("printer.local:631/api", "printer.local:631", "LAN")`, and `printer.local/api` joined the preview's refused list.
  - `test_make_default_never_writes_an_endpoint_the_contract_refuses` (`Tests/Chat/test_local_server_discovery.py`, TASK-38964): its written case asserted the typed value was written. It now asserts the value **s** writes (`_saved_endpoint`), e.g. `http://127.0.0.1:8001/v1/chat/completions` for vLLM, and it gained the three loopback spellings.
- Tests in `Tests/Chat/test_local_server_discovery.py` (dreams-boundary runs it whole): `test_chat_settings_reads_a_schemeless_loopback_endpoint_as_the_save_does` (AC#1, vLLM/Ollama/Custom × 3), `test_llamacpp_readiness_refuses_a_remote_endpoint_typed_without_a_scheme` (AC#2, plus the AC#3 row), `test_a_schemeless_dotted_key_is_never_shown_as_a_host` (AC#3, the key, `/v1` and `sk-`), `test_values_with_a_scheme_a_port_or_a_loopback_host_display_as_before` (AC#4, green at base too, including `local-service:9090/v1`).
- Red at base source (run.sh, `-n 4`, load 9.8): 36 failed. AC#1 `assert 'Invalid URL' != 'Invalid URL'` (9); AC#2 `assert 'Ready' != 'Ready'` (5); AC#3 `'0123456789ab...ghijklmnop/v1' == 'invalid endpoint'` (3); the writer test refused the loopback spellings (`runtime_published` False, 15) and wrote the typed value (4). Green after.
- Covering runs (run.sh, `-n 4`): the 40 files that name the changed helpers (35 by grep, plus the apply-defaults flow, resize reflow, gateway, disclosures and save-scope UI files), at load 13-68: 3830 passed, 174 failed, 17 errors. The 191 failing nodes, rerun at `fu-base-r11` at load 36: 187 of them fail there too. Of the other 4, two passed on a serial rerun here, and two (`test_vllm_console_real_metadata_conflict_blocks_detached_send`, `test_console_model_switch_inherits_selected_model_default_profile`) fail serially at base as well. The core files (discovery, defaults, endpoints, `Tests/LLM_Provider_Catalog/`, Chat session settings, readiness words and codes) at load 23: 1213 passed, 1 failed (`test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh`, red at base too).
- Docs: `Docs/User_Guide/console.md` (schemeless addresses, what Save endpoint & use model writes, when a value reads "invalid endpoint").

Files: `tldw_chatbook/Chat/console_session_settings.py`, `tldw_chatbook/Chat/console_settings_defaults.py`, `tldw_chatbook/Chat/provider_setup_persistence.py`, `tldw_chatbook/Chat/console_provider_endpoints.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/Chat/test_console_settings_defaults.py`, `Docs/User_Guide/console.md`.
