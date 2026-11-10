---
id: TASK-39151
title: "Console's llama.cpp resolution still reads a remote endpoint typed without a scheme as http://, so a config value or an old chat the save would refuse is probed and used"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 21:27
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
  - Outside Console, `chat_with_llama` (`LLM_Calls`) normalizes a config value the same way and, since review r13s-rev1 (M5, widened in rev2), refuses a URL without an `http`/`https` scheme with a `ChatConfigurationError` before sending (below).
- **AC#2.** `build_console_settings_readiness` checks, when the chat has no endpoint and is not a custom entry, the endpoint llama.cpp would send to (`_console_endpoint_restart_fallback`: environment, Console override, `[api_settings.llama_cpp]`, default) with `_valid_base_url`. It costs about 40 µs more per llama.cpp readiness build without a session endpoint (measured at load ~85, most of it `validate_url`'s metrics), the same check a session endpoint already paid.
- **AC#3, what an old chat shows.** A chat whose stored `console_session_settings` metadata holds `gpu-box.lan:8080` parses unchanged, its send resolves not ready with `INVALID_LLAMACPP_BASE_URL_COPY` and makes no request, its readiness is "Invalid URL" with the same detail, and the Endpoint line of its settings summary reads "Endpoint: Invalid (add http:// or https://)" instead of "Endpoint: Not configured" (copy as revised in review r13s-rev1, below; a key-shaped value reads "Endpoint: Invalid"). The value itself is never shown (it could be a pasted key).
- **AC#4.** Loopback spellings typed without a scheme and values with a typed scheme resolve, probe `/health` and read ready as before, from the session, the config and a custom entry. `127.0.0.2:8080` and `LOCALHOST:8080` typed without a scheme are now refused here as the contract, the save and readiness (TASK-39042) already refuse them.
- **Pins rewritten on purpose:**
  - `test_llamacpp_readiness_refuses_a_remote_endpoint_typed_without_a_scheme` (TASK-39042 AC#3) asserted the row "Endpoint: Not configured"; it now asserts the new copy for each value (see review r13s-rev1).
  - `test_gateway_attaches_on_device_destination_after_llamacpp_normalization` (`Tests/Chat/test_console_provider_gateway.py`) typed `127.42.7.9:9099/v1/chat/completions` without a scheme and expected Ready; it now types `http://`, and still pins the normalized base and the on-device destination. AC#4's "loopback values typed without a scheme" is read as the contract's loopback spellings (`is_exact_schemeless_host`: localhost, 127.0.0.1, [::1]), the rule TASK-39042 gave readiness; any other `127.x` typed without a scheme was already "Invalid URL" in Chat settings and refused by the save.
- **Tests** (`Tests/Chat/test_local_server_discovery.py`, which the dreams-boundary step runs whole; every request goes to an `httpx.MockTransport` that records URLs):
  - `test_llamacpp_never_probes_a_remote_endpoint_typed_without_a_scheme` (session, config, custom entry × 2 hosts), through `build_console_provider_selection_from_settings` and `ConsoleProviderGateway.resolve_for_send`. Red at base ae9f9e8d21: 6 failed, `assert ['http://gpu-...:8080/health'] == []` and `['http://192....:8080/health']`.
  - `test_llamacpp_readiness_refuses_a_held_endpoint_typed_without_a_scheme` (`[api_settings.llama_cpp]`, Console override, environment × 2). Red at base: 6 failed, `assert 'Ready' != 'Ready'`.
  - `test_a_saved_chat_with_a_remote_schemeless_llamacpp_endpoint_sends_nothing`: stored metadata through `parse_persisted_console_session_settings`. Red at base on the `/health` request.
  - `test_llamacpp_loopback_or_schemed_endpoints_resolve_as_before` (5 values × 3 sources): green at base and after; with `read_schemeless_loopback` temporarily removed from both copies, the 9 loopback cases failed (reverted).
- **Docs:** `Docs/User_Guide/console.md` (the address is never contacted, from any source, and what the summary line and a send show); `Docs/User_Guide/settings.md` (a llama.cpp custom endpoint's base URL needs a scheme unless loopback).
- **Lane.** The new tests are in `Tests/Chat/test_local_server_discovery.py`, which the dreams-boundary job's "Run Dreams and Guardian boundary contracts" step runs whole (never the fast PR contract); no workflow change was needed for them. That step, with this round's nodes (TASK-39149's two Trace export nodes, about 5 s, were still in it; review r13s-rev1 moved them to recovered-suites), through `jobstep.sh` (its 35 targets in order, its flags, serial) three times: 1126 passed each, in 1742.6 s, 1055.7 s and 967.1 s, at 1-minute load 84, 65 and 29 when each started. In CI the job took 10.9-14.1 min of its 20 over ten PR runs on 2026-11-09/10 (the step 7.7-9.8 min); this task's new tests take about 8 s (28 cases) serial on their own.
- **Runs** (run.sh, `-n 4` unless noted, 1-minute load 30-85 on 18 CPUs with other agents running; shared by TASK-39149, TASK-39150 and TASK-39151 on branch fu/r13-sec, base ae9f9e8d21, 2026-11-10):
  - Final core set (discovery, project instructions, Trace v2, trajectory export and import, the endpoint contract, custom endpoint registry, exchange capture, trace final values, the provider gateway): 1380 passed, 3 failed. The 3 (`test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base`, `test_console_send_leaves_base_url_unpinned_in_default_case[deepseek]`, `TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary`) fail at base with the same command.
  - Wider covering run: the 96 non-UI files that name a changed helper (`normalize_llamacpp_base_url`, the readiness and summary builders, the custom endpoint registry, project instructions, trajectory export, `resolve_llamacpp`) plus `Tests/LLM_Provider_Catalog/`: 5054 passed, 167 failed, 29 errors; 14 UI and performance files (Chat settings, custom endpoints, discovery, switcher, project instructions, trace export and import, keystroke census): 814 passed, 152 failed. Every failing node was rerun with the four changed source files at base: all fail there too, except the 127.x gateway pin rewritten on purpose (TASK-39151), 10 goal and agent-swap nodes and 1 Chat-settings node, which passed in that one base batch. Rerun on their own, alternating head and base twice, the same 4 of the 10 failed on both trees and the other 6 passed on both (base once failed a fifth, a capacity-wait test), and the Chat-settings node failed on both with `RecoveryRequired: raw_source_selection_changed`. Most reds in both runs are that local `RecoveryRequired` trap.
  - `Tests/LLM_Calls` llama files: 175 passed. CI workflow contracts (`test_ci_queue_pressure_contract.py`, `test_derived_artifacts_workflow.py`, `test_console_p0_regression_gate.py`): 365 passed. Serial: `Tests/Architecture/test_module_size_ratchet.py` 201 passed (gateway 8,488 lines under its 8,493 pin); the screen-size ratchet, chat code identity ratchet, Agents/Chat boundary and agent runtime import footprint, 33 passed. `./scripts/preflight.sh`: all checks passed.

Files: `tldw_chatbook/Chat/console_session_settings.py`, `tldw_chatbook/Chat/console_provider_gateway.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/Chat/test_console_provider_gateway.py`, `Docs/User_Guide/console.md`, `Docs/User_Guide/settings.md`.

**Review r13s-rev1 (six Minor; M1, M2, M3 and M5 here, M4 and M6 in TASK-39149).**
- M1: the Endpoint line told any value typed without a scheme to add one, a pasted key included (which `http://` would then make Ready and probe as a host). Readiness now records why the endpoint is invalid (`ConsoleSettingsReadiness.invalid_endpoint`, set only when it is): `"needs_scheme"` when the contract refuses the value only for lacking a scheme (`REMOTE_ENDPOINT_SCHEME_REQUIRED`, now a contract constant) and it is a host and port or an IP literal (`_invalid_endpoint_kind`); otherwise `"invalid"`. The line reads "Endpoint: Invalid (add http:// or https://)" or "Endpoint: Invalid". The copy no longer says "remote": the contract also refuses `LOCALHOST:8080` and `127.0.0.2:8080` typed without a scheme (only the exact spellings `localhost`, `127.0.0.1`, `[::1]` may leave it out) and accepts both once `http://` is added, which `test_the_endpoint_line_tells_only_an_address_to_add_a_scheme` asserts for every "add" case.
  - Any other invalid typed endpoint (such as `ftp://…`) now reads "Endpoint: Invalid" too, not "Not configured"; "Not configured" stays for a missing endpoint or provider. Pin rewritten on purpose: `test_summary_builder_reports_only_genuine_provider_endpoint_inheritance` (`Tests/UI/test_console_session_settings.py`, `ftp://invalid-endpoint`).
  - TASK-39042's `test_llamacpp_readiness_refuses_a_remote_endpoint_typed_without_a_scheme` now pairs each value with its line: the two addresses "add", the three key shapes plain "Invalid".
- M2: a held value llama.cpp now refuses (`[api_settings.llama_cpp]`, `[console] llama_cpp_base_url_override`, `TLDW_CONSOLE_LLAMA_CPP_BASE_URL`, such as compose `llama:8080`) read "Endpoint: Not configured" beside "Invalid URL". The readiness field carries the held value's kind, so the line reads the same as for a typed one. `test_llamacpp_readiness_refuses_a_held_endpoint_typed_without_a_scheme` asserts it, and `llama:8080` joins `_REMOTE_SCHEMELESS_LLAMACPP` (so it is also never probed).
- Red at 9a48208343's sources (run.sh): 24 failed. In the discovery file 23: held values (9, `'Endpoint: Not configured' == 'Endpoint: In... or https://)'`), the TASK-39042 pin (5), the new line test (8; `ftp://…` read `'Endpoint: Not configured'`, the others the old "remote address" copy or the add-scheme copy where plain "Invalid" is expected), and the saved chat (1). The rewritten UI pin, run on its own: `'Endpoint: Not configured' == 'Endpoint: Invalid'`. With M5's 3 below, 24 + 3. Green after.
- M3: `console.md` gives a skipped llama.cpp custom endpoint its own sentence with the copy its chat actually shows, "Provider blocked: this custom endpoint is no longer configured. Restore its entry or choose another provider before sending." (probed), and describes both Endpoint lines.
- M5: `chat_with_llama` (`LLM_Calls/LLM_API_Calls_Local.py`) raises `ChatConfigurationError` (status `None`: nothing reached a provider) naming the missing `http://` or `https://` when the normalized URL has no scheme, before anything is sent, instead of `requests` failing it as a network error. `test_chat_with_llama_refuses_a_remote_endpoint_typed_without_a_scheme` (in the discovery file, so on the dreams-boundary lane; the shared sender and the config snapshot are stubbed): red at 9a48208343 for the three remote values (`DID NOT RAISE`, the stub was called), green after; `localhost:8080/v1` and `http://gpu-box.lan:8080/v1` still reach the sender.
- Runs (run.sh, `-n 4`, load 40-57): 53 test files, the 48 that name the Endpoint line, the summary builder, `ConsoleSettingsReadiness`, the contract or `chat_with_llama` (plus project instructions, Trace v2, the custom endpoint registry and the llama `LLM_Calls` files): 3721 passed, 456 failed. The 456 rerun with every changed source file at base ae9f9e8d21: 452 fail there too (mostly the local `RecoveryRequired` trap). The other 4 (`test_create_endpoint_returns_to_responsive_settings_with_settled_evidence[outcome0]`, `test_the_focused_endpoint_guide_says_required_as_its_row_does[llama_cpp-…]`, `test_mounted_back_provider_identity_change_cannot_commit_old_model[credential|endpoint]`) passed at head in two reruns on their own, and the first failed once at base in the same alternation: load flakes. `Tests/CI/` 1240 passed, 1 skipped; the discovery, contract, Chat session settings and Trace v2 files plus the rewritten UI pin: 953 passed, 1 failed (`test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh`, red at base, as TASK-39042 recorded); `LLM_Calls` llama files and `test_encrypted_api_key_never_sent.py`: 227 passed; serial module-size ratchet: 201 passed.

**Review r13s-rev2.**
- 1: the guard that keeps a value the contract refuses for another reason (user info, a query) out of the add-scheme copy (`_invalid_endpoint_kind`'s `errors != (REMOTE_ENDPOINT_SCHEME_REQUIRED,)`) had no test. `test_the_endpoint_line_tells_only_an_address_to_add_a_scheme` adds `user:pw@gpu-box.lan:8080`, `:secretpw@gpu-box.lan:8080` and `gpu-box.lan:8080?key=sk-x`, each "Endpoint: Invalid". With the guard temporarily removed all three failed (they read the add-scheme copy); restored, 11 passed.
- 2: the two note corrections above (the `chat_with_llama` line, and the red count 24 + 3).
- 3 (security): M5's guard checked only for "://", so `ftp://user:hunter2@…` or `htp://…` reached `requests` and failed as a provider error echoing the whole URL, user info included. `chat_with_llama` now refuses any URL whose `urlsplit` scheme is not `http` or `https`, with the same `ChatConfigurationError`, whose message names no part of the value. `test_chat_with_llama_refuses_a_non_http_scheme_without_echoing_it` covers every provider that dispatches to it (`API_CALL_HANDLERS`: llama_cpp, local_llamacpp, local_llamafile, strata, polystrata) × `ftp://user:hunter2@gpu-box.lan:8080` and `htp://gpu-box.lan:8080/v1`: nothing reaches the (stubbed) sender, and the message holds none of `hunter2`, `user:`, `gpu-box`, `ftp`, `htp`. Red before (10 × `DID NOT RAISE`), green after.
- Runs (run.sh, `-n 4`): the 34 non-UI files that reach `chat_with_llama` or the llama-family keys (plus the llama `LLM_Calls` files): 3481 passed, 42 failed, 10 errors; all 44 failing nodes fail the same way with every changed source at base ae9f9e8d21 (`RecoveryRequired`, an `ImportError` of `LocalReasoningReplay` in two files, and agent-call thinking reds). `Tests/CI/` 1240 passed, 1 skipped. `./scripts/preflight.sh` passes. ruff: no new findings.
