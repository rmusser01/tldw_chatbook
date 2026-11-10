---
id: TASK-39151
title: "Console's llama.cpp resolution still reads a remote endpoint typed without a scheme as http://, so a config value or an old chat the save would refuse is probed and used"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 18:27
labels:
- follow-up
- console
- providers
- settings
priority: medium
references:
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/Chat/console_chat_store.py
- Tests/Chat/test_local_server_discovery.py
- Tests/Chat/test_console_provider_gateway.py
- Docs/User_Guide/console.md
dependencies:
- TASK-39042
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11's settings review: TASK-39042's notes say "Left alone, as the coordinator asked: `normalize_llamacpp_base_url` accepting any host, and old remote llama.cpp chats". Filed in round 12 of the 2026-10-23 burn-down. Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are dev 3a6498ae14's (re-read 2026-11-09).

**The contract's rule.** A remote endpoint needs a typed scheme. TASK-16200 AC#4 says "remote schemeless endpoints remain blocked", and the contract refuses one with "Remote endpoint URLs must include an HTTP or HTTPS scheme." (`Provider_Contract/provider_endpoint_contract.py:281`). TASK-39042 made Chat-settings readiness follow it (`_valid_base_url`, `console_session_settings.py:3202`): only a loopback spelling may leave out the scheme.

**Where the old any-host rule still runs.**
- `normalize_llamacpp_base_url` has two copies (`console_session_settings.py:340` and `console_provider_gateway.py:1865`). Both add `http://` to any value without `://`, whatever its host.
- The gateway's send-time resolution (`resolve_llamacpp`, `console_provider_gateway.py:4190-4276`) normalizes with it, probes `/health`, and returns Ready.
- A probe on dev and on the stack, with a mock transport (no network):
  - `gpu-box.lan:8080` and `192.168.1.20:8080` resolve `ready=True`;
  - their `base_url` is `http://gpu-box.lan:8080` and `http://192.168.1.20:8080`;
  - each got a `/health` request first.
- Readiness for a config-held value reads `Ready` on both trees: `build_console_settings_readiness` with no session endpoint and `api_settings.llama_cpp.api_url = "gpu-box.lan:8080"`. The same value as a session endpoint reads `Invalid URL` / `endpoint_invalid`. The save refuses such a value, but a hand-edited or older config can hold one.
- Other callers of the normalizer:
  - the gateway's model listing (`:5211`, `:5533`) and dispatch (`:6524`, `:6591`, `:6714`);
  - the controller (`console_chat_controller.py:715`);
  - the model switcher (`UI/Console_Modules/model_switcher.py:878-879`);
  - Settings (`settings_screen.py:7456`, `settings_provider_view_model.py:560`);
  - the custom endpoint registry (`custom_endpoint_registry.py:576`) and `chat_screen.py:7725`.

**Old chats.** A saved chat keeps its session endpoint in its `console_session_settings` metadata (`console_chat_store.py:19111-19126`). Some chats were saved with a remote schemeless llama.cpp endpoint before TASK-39042, while Chat settings still accepted it. Such a chat now reads `endpoint_invalid`, and its Endpoint row reads "Endpoint: Not configured" (TASK-39042's notes). The gateway would still resolve the stored value as above. What such a chat does on its next send was not traced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No Console path probes or sends to a non-loopback llama.cpp endpoint typed without a scheme, whether it comes from the session, the config or a custom entry; a test with a mock transport is red on dev, where it shows the `/health` request to `http://gpu-box.lan:8080`
- [x] #2 Readiness for such a config value is not Ready; a test is red on dev
- [x] #3 A saved chat whose stored session endpoint is such a value sends nothing to it, and its copy tells the user what to change instead of "Not configured"; the notes say what it shows
- [x] #4 Loopback values typed without a scheme, and values with a typed scheme, resolve as before
- [x] #5 The tests run on a PR lane (never the fast PR contract), with three timings recorded, and `Docs/User_Guide/console.md` describes the behaviour
<!-- AC:END -->

## Implementation Plan

1. Both `normalize_llamacpp_base_url` copies (`console_session_settings.py`, `console_provider_gateway.py`) read a value typed without a scheme through the contract's `read_schemeless_loopback`, so only a loopback spelling gains `http://`; anything else stays schemeless, and `validate_url` then refuses it before `resolve_llamacpp` probes, and the registry loader refuses such a custom entry.
2. Readiness: with no session endpoint, llama.cpp checks the endpoint the send would use (env override, Console override, config, default) with `_valid_base_url`.
3. The Chat settings Endpoint row says what to change for a value typed without a scheme that is not a loopback spelling, instead of "Not configured".
4. Tests in `Tests/Chat/test_local_server_discovery.py` with a mock transport: session, config and custom-entry sources never probe (red on dev); config readiness is not Ready (red on dev); a stored chat parsed from its metadata sends nothing and its row says what to change (red on dev); loopback schemeless and schemed values resolve as before (red under a temporary edit).
5. Update `Docs/User_Guide/console.md`; three timings of the dreams-boundary step.

## Implementation Notes

Console's llama.cpp path now reads an endpoint typed without a scheme by the contract's rule everywhere, so a remote one (`gpu-box.lan:8080`, `192.168.1.20:8080`) is never probed or sent to.
- **Root cause, one rule in both copies.** `normalize_llamacpp_base_url` (`console_session_settings.py` and `console_provider_gateway.py`) added `http://` to any value without `://`. Both now call the contract's `read_schemeless_loopback` (TASK-39042), so only `localhost`, `127.0.0.1` and `[::1]` gain `http://`; any other value comes back schemeless. Every caller routes through it:
  - send-time resolution (`resolve_llamacpp`): `validate_url` refuses the schemeless value before any `/health` or `/v1/models` request, and the resolution reads `INVALID_LLAMACPP_BASE_URL_COPY` ("Provider blocked: invalid llama.cpp base URL. Use an http(s) URL such as http://127.0.0.1:9099."); the controller's selection core (config, `[console] llama_cpp_base_url_override`, `TLDW_CONSOLE_LLAMA_CPP_BASE_URL`) feeds the same value in;
  - the gateway's stream and completion calls already raise "invalid llama.cpp base URL" on `validate_url`;
  - the custom endpoint registry's `validate_entry` now refuses such a llama.cpp `base_url`, as it already did for the other families, so the loader skips a hand-edited entry ("custom endpoint 'gpu' ignored: Endpoint must be a valid http(s) URL.") and the Settings editor refuses one with that message instead of silently adding `http://`;
  - the compare, display and draft callers (model switcher adoption compare, Settings' reasoning-override key, chat screen, Chat settings modal, endpoint template modal) get the schemeless value back; none of them makes a request.
  - Outside Console, `chat_with_llama` (`LLM_Calls`) normalizes a config value the same way; `requests` then refuses the schemeless URL itself (no connection), not with a `ChatConfigurationError`.
- **AC#2.** `build_console_settings_readiness` checks, when the chat has no endpoint and is not a custom entry, the endpoint llama.cpp would send to (`_console_endpoint_restart_fallback`: environment, Console override, `[api_settings.llama_cpp]`, default) with `_valid_base_url`. It costs about 40 µs more per llama.cpp readiness build without a session endpoint (measured at load ~85, most of it `validate_url`'s metrics), the same check a session endpoint already paid.
- **AC#3, what an old chat shows.** A chat whose stored `console_session_settings` metadata holds `gpu-box.lan:8080` parses unchanged, its send resolves not ready with `INVALID_LLAMACPP_BASE_URL_COPY` and makes no request, its readiness is "Invalid URL" with the same detail, and the Endpoint line of its settings summary reads "Endpoint: Invalid (a remote address needs http:// or https://)" instead of "Endpoint: Not configured". The value itself is never shown (it could be a pasted key). That line applies to any provider whose typed session endpoint is a non-loopback value without a scheme; other invalid endpoints (`ftp://…`) still read "Not configured" (`test_summary_builder_reports_only_genuine_provider_endpoint_inheritance`, unchanged).
- **AC#4.** Loopback spellings typed without a scheme and values with a typed scheme resolve, probe `/health` and read ready as before, from the session, the config and a custom entry. `127.0.0.2:8080` and `LOCALHOST:8080` typed without a scheme are now refused here as the contract, the save and readiness (TASK-39042) already refuse them.
- **Pins rewritten on purpose:**
  - `test_llamacpp_readiness_refuses_a_remote_endpoint_typed_without_a_scheme` (TASK-39042 AC#3) asserted the row "Endpoint: Not configured"; it now asserts the new copy.
  - `test_gateway_attaches_on_device_destination_after_llamacpp_normalization` (`Tests/Chat/test_console_provider_gateway.py`) typed `127.42.7.9:9099/v1/chat/completions` without a scheme and expected Ready; it now types `http://`, and still pins the normalized base and the on-device destination. AC#4's "loopback values typed without a scheme" is read as the contract's loopback spellings (`is_exact_schemeless_host`: localhost, 127.0.0.1, [::1]), the rule TASK-39042 gave readiness; any other `127.x` typed without a scheme was already "Invalid URL" in Chat settings and refused by the save.
- **Tests** (`Tests/Chat/test_local_server_discovery.py`, which the dreams-boundary step runs whole; every request goes to an `httpx.MockTransport` that records URLs):
  - `test_llamacpp_never_probes_a_remote_endpoint_typed_without_a_scheme` (session, config, custom entry × 2 hosts), through `build_console_provider_selection_from_settings` and `ConsoleProviderGateway.resolve_for_send`. Red at base ae9f9e8d21: 6 failed, `assert ['http://gpu-...:8080/health'] == []` and `['http://192....:8080/health']`.
  - `test_llamacpp_readiness_refuses_a_held_endpoint_typed_without_a_scheme` (`[api_settings.llama_cpp]`, Console override, environment × 2). Red at base: 6 failed, `assert 'Ready' != 'Ready'`.
  - `test_a_saved_chat_with_a_remote_schemeless_llamacpp_endpoint_sends_nothing`: stored metadata through `parse_persisted_console_session_settings`. Red at base on the `/health` request.
  - `test_llamacpp_loopback_or_schemed_endpoints_resolve_as_before` (5 values × 3 sources): green at base and after; with `read_schemeless_loopback` temporarily removed from both copies, the 9 loopback cases failed (reverted).
- **Docs:** `Docs/User_Guide/console.md` (the address is never contacted, from any source, and what the summary line and a send show); `Docs/User_Guide/settings.md` (a llama.cpp custom endpoint's base URL needs a scheme unless loopback).
- **Lane.** The new tests are in `Tests/Chat/test_local_server_discovery.py`, which the dreams-boundary job's "Run Dreams and Guardian boundary contracts" step runs whole (never the fast PR contract); no workflow change was needed for them. That step, with this round's nodes, through `jobstep.sh` (its 35 targets in order, its flags, serial) three times: 1126 passed each, in 1742.6 s, 1055.7 s and 967.1 s, at 1-minute load 84, 65 and 29 when each started. In CI the job took 10.9-14.1 min of its 20 over ten PR runs on 2026-11-09/10 (the step 7.7-9.8 min); this task's new tests take about 8 s (28 cases) serial on their own.
- **Runs** (run.sh, `-n 4` unless noted, 1-minute load 30-85 on 18 CPUs with other agents running; shared by TASK-39149, TASK-39150 and TASK-39151 on branch fu/r13-sec, base ae9f9e8d21, 2026-11-10):
  - Final core set (discovery, project instructions, Trace v2, trajectory export and import, the endpoint contract, custom endpoint registry, exchange capture, trace final values, the provider gateway): 1380 passed, 3 failed. The 3 (`test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base`, `test_console_send_leaves_base_url_unpinned_in_default_case[deepseek]`, `TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary`) fail at base with the same command.
  - Wider covering run: the 96 non-UI files that name a changed helper (`normalize_llamacpp_base_url`, the readiness and summary builders, the custom endpoint registry, project instructions, trajectory export, `resolve_llamacpp`) plus `Tests/LLM_Provider_Catalog/`: 5054 passed, 167 failed, 29 errors; 14 UI and performance files (Chat settings, custom endpoints, discovery, switcher, project instructions, trace export and import, keystroke census): 814 passed, 152 failed. Every failing node was rerun with the four changed source files at base: all fail there too, except the 127.x gateway pin rewritten on purpose (TASK-39151), 10 goal and agent-swap nodes and 1 Chat-settings node, which passed in that one base batch. Rerun on their own, alternating head and base twice, the same 4 of the 10 failed on both trees and the other 6 passed on both (base once failed a fifth, a capacity-wait test), and the Chat-settings node failed on both with `RecoveryRequired: raw_source_selection_changed`. Most reds in both runs are that local `RecoveryRequired` trap.
  - `Tests/LLM_Calls` llama files: 175 passed. CI workflow contracts (`test_ci_queue_pressure_contract.py`, `test_derived_artifacts_workflow.py`, `test_console_p0_regression_gate.py`): 365 passed. Serial: `Tests/Architecture/test_module_size_ratchet.py` 201 passed (gateway 8,488 lines under its 8,493 pin); the screen-size ratchet, chat code identity ratchet, Agents/Chat boundary and agent runtime import footprint, 33 passed. `./scripts/preflight.sh`: all checks passed.

Files: `tldw_chatbook/Chat/console_session_settings.py`, `tldw_chatbook/Chat/console_provider_gateway.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/Chat/test_console_provider_gateway.py`, `Docs/User_Guide/console.md`, `Docs/User_Guide/settings.md`.
