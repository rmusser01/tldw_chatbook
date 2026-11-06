---
id: TASK-39041
title: "Discover models refuses a QwenCloud loopback endpoint typed without a scheme that the save accepts, and lists from a bracketed IPv6 zone id that the save refuses"
status: To Do
created_date: 2026-11-06 10:09
labels:
- follow-up
- providers
- settings
- needs-owner-decision
priority: low
references:
- tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py
- tldw_chatbook/LLM_Calls/qwencloud_url.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/Chat/test_local_server_discovery.py
- Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py
- Docs/User_Guide/settings.md
dependencies:
- TASK-38963
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on endpoint PR #3432 (head 760d2987c7), which lands TASK-38963. TASK-38963's notes list both halves as follow-ups. Line numbers are dev's; ones marked #3432 are from that head.

TASK-38963 made Discover models decide hosts by the endpoint contract's rule, and `Docs/User_Guide/settings.md` (#3432 :326-334) now says Discover "reads the endpoint's host as **s** does". Two splits remain, both present on dev too.

**QwenCloud: loopback without a scheme.**
- Settings hands the typed, unsaved Endpoint field straight to Discover (`_provider_discovery_staged_settings`, `settings_screen.py:16276`).
- QwenCloud's eligibility branch (`openai_compatible_model_discovery.py:485-490`; #3432 :508-513) normalizes the raw value with `normalize_qwencloud_base_url`. `qwencloud_url.py:117-123` reads `localhost:5001/v1` as the scheme `localhost` and raises "must be an absolute HTTP(S) URL".
- The generic branch's `_parse_endpoint` adds `http://` to a value typed without a scheme (`:228`); #3432 limits that to the exact loopback spellings (#3432 :230-252, via `is_exact_schemeless_host`). QwenCloud's branch does neither.
- A probe (no network, no config writes), identical on dev and on #3432:
  - `localhost:5001/v1`, `127.0.0.1:5001/v1` and `[::1]:5001/v1` give discover=False. The catalog service reports `status=unsupported` and never calls the client: "This endpoint is not a valid OpenAI-compatible models endpoint."
  - The save stores `http://localhost:5001/v1` (and the other two the same way).
  - vllm and custom accept all three for Discover.

**Every other provider: a bracketed IPv6 zone id.**
- For vllm, custom, ollama and llama_cpp, Discover lists from `http://[fe80::1%25en0]:5001/v1`. The contract refuses it for every provider: `('Endpoint URL must include a valid host and port.',)`. The save then raises "Endpoint is invalid.".
- QwenCloud's and koboldcpp's Discover refuse it; koboldcpp has done so since TASK-36380.
- Console treats it two ways: `safe_endpoint_display` reads "invalid endpoint", while `normalize_generic_endpoint_for_compare` keeps it as itself.
- Neither rule was chosen by a task or an ADR:
  - Discover's acceptance is from b415972a97 (PR #1562, QwenCloud, 2026-08-12), which added `_is_safe_generic_authority` and the pin `test_non_qwen_discovery_preserves_scoped_ipv6_zone_id_support` (`Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py:267`).
  - The contract's refusal is from d9df84eff7 (PR #1630), which added the `"%"` host refusal (`provider_endpoint_contract.py:547`; #3432 :620, in `canonical_endpoint_host`).
  - The same PR refused zone ids for TTS (`Tests/TTS/test_openai_compatible_endpoint.py:216`).
- TASK-38963 kept the split on purpose and documented it as the exception in `settings.md` (#3432 :331-334).
- Accepting zone ids means relaxing `canonical_endpoint_host`, which the save, Console compare and endpoint identity all use. Refusing them means rewriting the #1562 pin. The owner decides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With QwenCloud and `localhost:5001/v1`, `127.0.0.1:5001/v1` or `[::1]:5001/v1` in the Endpoint field, Discover models reaches the discovery client at the same models URL as the `http://` form; a test in `Tests/Chat/test_local_server_discovery.py` is red on dev (`status == "unsupported"`, client never called)
- [ ] #2 TASK-38963's `test_discover_models_accepts_exactly_the_hosts_the_save_accepts` also covers the three loopback spellings, with and without a scheme, for every contract provider, and is green
- [ ] #3 The owner decides whether a bracketed IPv6 zone id is an endpoint that Discover and the save both accept or both refuse; the decision is recorded in this task's notes
- [ ] #4 After that decision, Discover, the save and Console's display and compare agree on `http://[fe80::1%25en0]:5001/v1` for every contract provider; a test of the decided rule is red on dev, and the pin the decision contradicts is rewritten on purpose and named in the notes
- [ ] #5 `Docs/User_Guide/settings.md` describes the decided behaviour and keeps no stale exception
<!-- AC:END -->
