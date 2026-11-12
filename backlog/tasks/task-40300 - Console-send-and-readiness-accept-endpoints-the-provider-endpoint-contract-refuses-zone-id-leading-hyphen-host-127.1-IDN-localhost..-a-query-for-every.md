---
id: TASK-40300
title: "Console send and readiness accept endpoints the provider endpoint contract refuses (zone id, leading-hyphen host, 127.1, IDN, localhost.., a query) for every provider"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- security
- console
- providers
- endpoints
priority: high
references:
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Chat/console_provider_endpoints.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Chat/console_project_instructions.py
- tldw_chatbook/Chat/custom_endpoint_registry.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
dependencies:
- TASK-39150
- TASK-39151
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (security stream, TASK-39150 and TASK-39151 reviews). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The 2026-11-06 "refuse everywhere" ruling says an endpoint the provider endpoint contract refuses is refused on every path. Save, Discover and the display already refuse these values. The paths that actually probe and send do not, so a hand-edited config, a custom entry or a session endpoint can still send to (or probe) an IPv6 zone-id link-local, a leading-hyphen host such as `-bad.corp`, the `127.1` shorthand, an IDN homograph, `localhost..`, or a URL with a query.

**One root cause.** Console decides endpoints with `Utils/input_validation.validate_url` plus the contract's scheme rule only. It never calls the contract's `resolve_provider_endpoint` or `canonical_endpoint_host` (`Provider_Contract/provider_endpoint_contract.py:235` refuses a query; `:607-638` hold the host rule). The custom registry does the same (`custom_endpoint_registry.py:475`).
- **Send, generic providers** (vLLM, Ollama, Custom, OpenAI, KoboldCpp, Aphrodite): `console_provider_gateway.py:4694/4704` resolves through `effective_provider_endpoint`. `console_provider_endpoints.py:351` returns the raw value for every provider except custom and llama.cpp, and `:354` (`resolution.persisted_endpoint or endpoint`) lets a custom entry fall back to the raw value when the contract refuses it. The resolution is then ready (`:4831-4833`) with no endpoint check.
- **Readiness**: `console_session_settings.py:1990-1996` checks only a session value (or llama.cpp's fallback), so a config-held value for the generic providers is never checked.
- **llama.cpp after TASK-39151**: `_valid_base_url` (`console_session_settings.py:3247-3262`) and `resolve_llamacpp` (gateway `:3971-3972`) use `read_schemeless_loopback` then `validate_url`, which has no rule for `%`, DNS labels, digit-only hosts, IDN or queries. A session zone id, `-bad.corp`, `127.1` and an IDN pass and get the `/health` probe; only `localhost..` is refused. `normalize_llamacpp_base_url` rebuilds the URL with an empty query (`console_session_settings.py:371`, gateway `:1865`), so `http://127.0.0.1:8080?key=...` reads Ready and is probed.
- **Project instructions**: `_parse_endpoint` (`console_project_instructions.py:290`) prefixes `http://` to any schemeless endpoint, remote included, so `gpu-box.lan:8080` gets the identity and consent key of `http://gpu-box.lan:8080`. `_canonical_hostname` (`:315-338`) strips every trailing dot (`rstrip('.')` at `:329`), so `localhost..` and `example.com..` fold to the clean host's identity; the contract strips exactly one dot and refuses `..`.

**Order.** TASK-39150 AC#4 kept project instructions' identity for hosts the send still accepts, so that a send with project instructions on does not become a cancelled run (the TASK-36517 trap). The project-instruction changes therefore land with the send and readiness change, never before it; the pin `test_hosts_the_contract_refuses_but_the_send_accepts_keep_their_identity` is rewritten on purpose.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For vLLM, Ollama, Custom, OpenAI, KoboldCpp, Aphrodite and llama.cpp, an endpoint the contract refuses (from the session, the config or a custom entry) reads not Ready and makes no request at send or probe; a mock-transport test is red before the fix
- [ ] #2 A schemed endpoint with a query or fragment reads Invalid URL in readiness and is never probed
- [ ] #3 Values the contract accepts resolve as before: schemed hosts, schemeless loopback spellings, compose names, and zone-less link-local or ULA literals
- [ ] #4 A custom entry the contract refuses no longer falls back to its raw value
- [ ] #5 Project instructions give a remote schemeless endpoint and a `..` host the refused identity, and fold exactly one trailing dot, in the same PR as the send change
<!-- AC:END -->
