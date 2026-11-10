---
id: TASK-39150
title: "Console's project-instruction endpoint identity and destination label keep an IPv6 zone id, against the 2026-11-06 refuse-everywhere ruling"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 18:27
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
- [x] #1 A bracketed zone-id endpoint, `%25` or raw `%`, reads "invalid endpoint" in the destination label and gets the same identity result as any other refused endpoint; a test is red on dev
- [x] #2 Each caller listed above handles a zone-id endpoint as it handles any other refused endpoint, with no crash, cancelled run or raised background start that a refused host does not already cause; the notes say what each caller does
- [x] #3 Link-local and ULA literals without a zone keep their label and identity
- [x] #4 The other hosts where this parser and the contract differ (`127.1`, an IDN, `localhost..`) are brought in line, or named in the notes as deliberate
- [x] #5 The tests run on a PR lane (never the fast PR contract), with three timings of that step recorded
<!-- AC:END -->

## Implementation Plan

1. `_canonical_hostname` (`Chat/console_project_instructions.py`) refuses a host holding `%` (a zone id, `%25` or raw, and any percent in a host), the contract's rule, before the IP-literal branch.
2. Trace each caller (gateway capture, trace final values, exchange capture and export, the controller's destination label, and the notice-key callers) against a zone id and an already-refused host, and record what each does.
3. Decide `127.1`, an IDN and `localhost..` against the send path (probe: the gateway readies config-held endpoints the contract refuses for every provider), and either align them or name them as deliberate.
4. Tests in `Tests/Chat/test_local_server_discovery.py` (dreams-boundary runs it whole): zone ids read "invalid endpoint" and get the refused identity result like `-bad.corp` across the callers (red on dev); zone-less link-local and ULA literals keep label and identity (red under a temporary edit).
5. Three timings of the dreams-boundary step.

## Implementation Notes

Owner ruling (2026-11-06, quoted from TASK-39041): "Refuse everywhere: Discover refuses it too, matching the save, the contract and TTS; the #1562 pin that preserves zone ids is rewritten on purpose. Zone ids are link-local only and no task or ADR chose to accept them."

- **Fix.** `_canonical_hostname` (`Chat/console_project_instructions.py`) returns None for a host holding `%` before its IP-literal branch, the contract's `canonical_endpoint_host` rule. A bracketed zone id, `%25` or raw `%`, now reads "(invalid endpoint)" and raises `ValueError("invalid provider endpoint")` for its identity and consent key, exactly as `-bad.corp` does. A percent-encoded DNS host (which yarl used to decode) is refused the same way; the contract refuses it too.
- **AC#2, what each caller does with a zone id** (all identical to `-bad.corp`, pinned by `_project_instruction_endpoint_callers` in the AC#1 test):
  - provider gateway, `ConsoleProviderCallSignals.begin_exchange`: catches the ValueError and captures `"[invalid endpoint]"`; the call goes on.
  - trace final values, `verify_provider_request_shadow`: returns the content-free `{"available": False, "omission_reason": "sanitizer_failed"}` shadow.
  - exchange capture (`build_request_capture`) and exchange export (`_endpoint`): `"[invalid endpoint]"`.
  - controller, `build_project_instruction_dispatch_notice`: destination label "vLLM (invalid endpoint)"; it never raises.
  - not in the task's list, but the same parser: the consent key (`project_instruction_notice_key`, read by `_native_start_project_required_action` and `confirm_project_dispatch`) raises "invalid provider endpoint". With project instructions on, a ready send to such an endpoint therefore ends as a cancelled run or a raised background start, which `-bad.corp` already causes (TASK-36517's notes). It is reached only when the send readies the endpoint: a probe (mock transport, no network) showed config-held `[fe80::1%25en0]`, `-bad.corp`, `127.1`, `bücher.example` and `localhost..` resolving Ready for vLLM, Ollama, Custom, OpenAI, KoboldCpp and Aphrodite, and llama.cpp readying a session zone id or `-bad.corp` (its `validate_url` check only). A session zone id for the other providers is blocked earlier ("endpoint not saved", "Selected endpoint: invalid endpoint"). Listed as a proposed follow-up in the round report.
- **AC#3.** Zone-less `[fe80::1]`, `[FE80::0001]` and `[fd12:3456::9]` keep their label, identity, trace shadow and consent key (digests measured at ae9f9e8d21).
- **AC#4, named as deliberate.** `127.1`, a Unicode IDN (`bücher.example` → `xn--bcher-kva.example`) and `localhost..` (folded to `localhost`) stay accepted. The save refuses all three, but Console's send still resolves them from a hand-edited config (probe above), so refusing them here would only turn such a send with project instructions into a cancelled run and drop its stored consent; `Tests/Chat/test_console_project_instructions.py` and TASK-36517's `_NOTICE_KEYS_BEFORE_UNDERSCORES` also pin IDN consent keys. None of the three lets one destination's consent stand for another: `127.1` keeps its own identity, and the IDN and `localhost..` fold onto the same host. `test_hosts_the_contract_refuses_but_the_send_accepts_keep_their_identity` pins the choice; with `_canonical_hostname` temporarily made `canonical_endpoint_host`, it failed for all three and TASK-36517's IDN digest pin failed too (reverted). Aligning them belongs after the send refuses contract-refused endpoints (follow-up).
- **Tests** (`Tests/Chat/test_local_server_discovery.py`, which the dreams-boundary step runs whole):
  - `test_a_zone_id_endpoint_is_refused_by_project_instructions_like_a_refused_host` (4 cases). Red at base ae9f9e8d21: 4 failed, e.g. `{'label': 'vLLM (http://[fe80::1%25en0]:5001)'} != {'label': 'vLLM (invalid endpoint)'}` and `{'trace_final_values': 'http://[fe80::1%25en0]:5001'} != {... 'sanitizer_failed'}`; the `-bad.corp` baseline held at base. Green after.
  - `test_zone_less_link_local_and_ula_literals_keep_their_project_identity` (3): green at base and after; under a temporary edit refusing every IPv6 literal all 3 failed (`'ValueError: ...ider endpoint' == 'http://[fe80::1]:5001'`), reverted.
  - `test_hosts_the_contract_refuses_but_the_send_accepts_keep_their_identity` (3): see AC#4.
- **Lane.** The new tests are in `Tests/Chat/test_local_server_discovery.py`, which the dreams-boundary job's "Run Dreams and Guardian boundary contracts" step runs whole (never the fast PR contract); no workflow change was needed for them. That step, with this round's nodes, through `jobstep.sh` (its 35 targets in order, its flags, serial) three times: 1126 passed each, in 1742.6 s, 1055.7 s and 967.1 s, at 1-minute load 84, 65 and 29 when each started. In CI the job took 10.9-14.1 min of its 20 over ten PR runs on 2026-11-09/10 (the step 7.7-9.8 min); this task's new tests take about 3 s (10 cases) serial on their own.
- **Runs** (run.sh, `-n 4` unless noted, 1-minute load 30-85 on 18 CPUs with other agents running; shared by TASK-39149, TASK-39150 and TASK-39151 on branch fu/r13-sec, base ae9f9e8d21, 2026-11-10):
  - Final core set (discovery, project instructions, Trace v2, trajectory export and import, the endpoint contract, custom endpoint registry, exchange capture, trace final values, the provider gateway): 1380 passed, 3 failed. The 3 (`test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base`, `test_console_send_leaves_base_url_unpinned_in_default_case[deepseek]`, `TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary`) fail at base with the same command.
  - Wider covering run: the 96 non-UI files that name a changed helper (`normalize_llamacpp_base_url`, the readiness and summary builders, the custom endpoint registry, project instructions, trajectory export, `resolve_llamacpp`) plus `Tests/LLM_Provider_Catalog/`: 5054 passed, 167 failed, 29 errors; 14 UI and performance files (Chat settings, custom endpoints, discovery, switcher, project instructions, trace export and import, keystroke census): 814 passed, 152 failed. Every failing node was rerun with the four changed source files at base: all fail there too, except the 127.x gateway pin rewritten on purpose (TASK-39151), 10 goal and agent-swap nodes and 1 Chat-settings node, which passed in that one base batch. Rerun on their own, alternating head and base twice, the same 4 of the 10 failed on both trees and the other 6 passed on both (base once failed a fifth, a capacity-wait test), and the Chat-settings node failed on both with `RecoveryRequired: raw_source_selection_changed`. Most reds in both runs are that local `RecoveryRequired` trap.
  - `Tests/LLM_Calls` llama files: 175 passed. CI workflow contracts (`test_ci_queue_pressure_contract.py`, `test_derived_artifacts_workflow.py`, `test_console_p0_regression_gate.py`): 365 passed. Serial: `Tests/Architecture/test_module_size_ratchet.py` 201 passed (gateway 8,488 lines under its 8,493 pin); the screen-size ratchet, chat code identity ratchet, Agents/Chat boundary and agent runtime import footprint, 33 passed. `./scripts/preflight.sh`: all checks passed.

Files: `tldw_chatbook/Chat/console_project_instructions.py`, `Tests/Chat/test_local_server_discovery.py`.
