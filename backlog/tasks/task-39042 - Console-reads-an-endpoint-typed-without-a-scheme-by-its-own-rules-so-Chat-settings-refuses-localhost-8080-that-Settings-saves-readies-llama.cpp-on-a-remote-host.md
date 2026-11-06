---
id: TASK-39042
title: "Console reads an endpoint typed without a scheme by its own rules, so Chat settings refuses localhost:8080 that Settings saves, readies llama.cpp on a remote host the save refuses, and shows a pasted dotted key as a host"
status: To Do
created_date: 2026-11-06 10:09
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
- [ ] #1 vLLM, Ollama and Custom take `localhost:8080`, `127.0.0.1:8080/v1` and `[::1]:8080/v1` in Chat settings as **s** does: readiness is not `endpoint_invalid`, and Save endpoint & use model writes the contract's persisted endpoint; a test is red on dev (`label == "Invalid URL"`)
- [ ] #2 llama.cpp readiness is not Ready for a non-loopback session endpoint typed without a scheme (`gpu-box.lan:8080`, `192.168.1.20:8080`, the synthetic dotted key); a test is red on dev (`label == "Ready"`)
- [ ] #3 A non-loopback value typed without a scheme or a port is never shown: for the synthetic dotted key, with and without `/v1` and with an `sk-` prefix, `safe_endpoint_display`, the unsaved-endpoint copy, the Chat settings Endpoint row and discovery's `endpoint_display` read "invalid endpoint", and `parse_console_endpoint_preview` returns None; a test in `Tests/Chat/test_local_server_discovery.py` is red on dev (the key is echoed)
- [ ] #4 Values with a typed scheme, loopback values without one, and `local-service:9090/v1` display as before; any pin the new rule contradicts (such as `printer.local/api`) is rewritten on purpose and named in the notes
- [ ] #5 `Docs/User_Guide/console.md` describes the behaviour
<!-- AC:END -->
