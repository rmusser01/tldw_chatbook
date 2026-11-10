---
id: TASK-39151
title: "Console's llama.cpp resolution still reads a remote endpoint typed without a scheme as http://, so a config value or an old chat the save would refuse is probed and used"
status: To Do
created_date: 2026-11-08 06:54
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
- [ ] #1 No Console path probes or sends to a non-loopback llama.cpp endpoint typed without a scheme, whether it comes from the session, the config or a custom entry; a test with a mock transport is red on dev, where it shows the `/health` request to `http://gpu-box.lan:8080`
- [ ] #2 Readiness for such a config value is not Ready; a test is red on dev
- [ ] #3 A saved chat whose stored session endpoint is such a value sends nothing to it, and its copy tells the user what to change instead of "Not configured"; the notes say what it shows
- [ ] #4 Loopback values typed without a scheme, and values with a typed scheme, resolve as before
- [ ] #5 The tests run on a PR lane (never the fast PR contract), with three timings recorded, and `Docs/User_Guide/console.md` describes the behaviour
<!-- AC:END -->
