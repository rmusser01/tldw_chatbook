---
id: TASK-39150
title: "Console's project-instruction endpoint identity and destination label keep an IPv6 zone id, against the 2026-11-06 refuse-everywhere ruling"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- console
- providers
- privacy
priority: medium
references:
- tldw_chatbook/Chat/console_project_instructions.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Chat/console_trace_final_values.py
- tldw_chatbook/Chat/console_exchange_capture.py
- tldw_chatbook/Chat/console_exchange_export.py
- tldw_chatbook/Chat/console_chat_controller.py
- Tests/Chat/test_console_project_instructions.py
dependencies:
- TASK-36517
- TASK-39041
- TASK-39137
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (settings review of TASK-39137, round-12 branch fu/r12-settings). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are dev's; the module is the same on the stack.

**The ruling.** Owner decision (2026-11-06), quoted from TASK-39041's notes: "Refuse everywhere: Discover refuses it too, matching the save, the contract and TTS; the #1562 pin that preserves zone ids is rewritten on purpose. Zone ids are link-local only and no task or ADR chose to accept them." TASK-39041 applied it to Discover, the save, display and compare. TASK-39137 (round 12) applied it to the Library egress classifier.

**Where it does not hold yet.**
- `_canonical_hostname` (`Chat/console_project_instructions.py:315-333`) accepts any IP literal first, through `ip_address(hostname)`, so a scoped literal passes with its zone.
- `_parse_endpoint` (`:280-303`) calls it. `sanitized_destination_label` (`:232-250`) and `canonical_provider_endpoint_identity` (`:267-269`) both read the result.
- A probe on dev and on the stack (no network):
  - `sanitized_destination_label("vLLM", "http://[fe80::1%25en0]:5001")` gives `vLLM (http://[fe80::1%25en0]:5001)`;
  - `canonical_provider_endpoint_identity` gives `http://[fe80::1%25en0]:5001`;
  - the raw `%en0` form is kept the same way;
  - the contract's `canonical_endpoint_host` refuses both hosts.
- Callers:
  - the provider gateway (`console_provider_gateway.py:1348`);
  - trace final values (`console_trace_final_values.py:1147`);
  - exchange capture (`console_exchange_capture.py:844`) and exchange export (`console_exchange_export.py:63`);
  - the controller's destination label (`console_chat_controller.py:1000`).

**This parser differs from the contract on other hosts too.** The same probe:
- accepts `127.1`, and `bücher.example` as `xn--bcher-kva.example`;
- folds `localhost..` to `localhost`;
- the contract refuses all three. Both refuse `-bad.example` and a 64-character label.

**Take care.** An endpoint this parser refuses makes `canonical_provider_endpoint_identity` raise `ValueError("invalid provider endpoint")`. TASK-36517's notes record what that did when the parser refused a host the contract accepted: the foreground run ended `RUN_CANCELLED`, the background start raised, and traces read "[invalid endpoint]". No PR lane runs `Tests/Chat/test_console_project_instructions.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A bracketed zone-id endpoint, `%25` or raw `%`, reads "invalid endpoint" in the destination label and gets the same identity result as any other refused endpoint; a test is red on dev
- [ ] #2 Each caller listed above handles a zone-id endpoint as it handles any other refused endpoint, with no crash, cancelled run or raised background start that a refused host does not already cause; the notes say what each caller does
- [ ] #3 Link-local and ULA literals without a zone keep their label and identity
- [ ] #4 The other hosts where this parser and the contract differ (`127.1`, an IDN, `localhost..`) are brought in line, or named in the notes as deliberate
- [ ] #5 The tests run on a PR lane (never the fast PR contract), with three timings of that step recorded
<!-- AC:END -->
