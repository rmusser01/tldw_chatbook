---
id: TASK-38963
title: "Discovery accepts hosts the endpoint contract refuses, and Console display and compare refuse single-label hosts it accepts"
status: To Do
created_date: 2026-11-06 01:21
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
- [ ] #1 For every host in TASK-36517's two pinned sets, plus single-label hosts with and without a scheme, Discover models eligibility and the provider save make the same accept/refuse decision; a test is red on dev for -kobold, kob*old, 1.2.3 and a 64-character label
- [ ] #2 http://kobold:5001 and http://kobold_cpp:5001 display as themselves, and two different single-label hosts compare as different in generic_endpoint_differs and adoption_endpoints_differ; a test is red on dev ('invalid endpoint', and False)
- [ ] #3 With a saved single-label endpoint and a different single-label session endpoint, the Console unsaved-endpoint guard blocks the send, and the adoption toast names the saved endpoint rather than "invalid endpoint"; both are shown red on dev
- [ ] #4 TASK-36517's refusal tests and the consent-key digest test stay green
<!-- AC:END -->
