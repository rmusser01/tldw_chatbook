---
id: TASK-41722
title: "Console refuses a pasted QwenCloud /v1/responses base that the adapter accepts"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 17:24
labels:
- follow-up
- providers
priority: low
references:
- tldw_chatbook/LLM_Calls/qwencloud.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Noted by TASK-41522 (round 16), predates it. The QwenCloud adapter checks its normalised base, so a pasted `.../compatible-mode/v1/responses` works (pinned by `Tests/LLM_Calls/test_qwencloud.py`), while Console's endpoint check reads the raw value and refuses it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console and the adapter reach the same decision for a pasted QwenCloud `/v1/responses` base
- [x] #2 A test pins both surfaces
<!-- AC:END -->

## Implementation Plan

1. Contract: for `qwencloud`, a trailing `/responses` is the adapter's own route, which `normalize_qwencloud_base_url` strips; the contract reads the base before it, so Console readiness, the send, Settings and the adapter decide alike.
2. Test both surfaces (Console readiness + send resolution, and the adapter through `chat_api_call`) for a pasted `.../compatible-mode/v1/responses`; red at base on the Console side.
3. Rewrite on purpose the TASK-41700 gateway-route row that pinned QwenCloud refusing `<gateway>/responses`, which its adapter accepts.

## Implementation Notes

Fixed in the endpoint contract, so every Console surface (readiness, send, Settings save, Make default, probe availability) decides as the adapter does: for `qwencloud`, a trailing `/responses` is the adapter's own route, which `normalize_qwencloud_base_url` strips from a pasted endpoint, so the contract reads the base before it. Before, Console's readiness read the raw value as a route and showed "Invalid URL" for `.../compatible-mode/v1/responses` while the adapter sent it. Settings now saves such a paste as `.../compatible-mode/v1`. Other refusals are unchanged: `/v1/responses/responses` is still refused by both.

**Test:** `Tests/Chat/test_local_server_discovery.py::test_console_and_the_qwencloud_adapter_agree_on_a_pasted_responses_endpoint`: from the config and from the session, Console's readiness is not "Invalid URL", the send resolution is Ready and the adapter is called with `.../compatible-mode/v1`; through `chat_api_call` the adapter posts `.../compatible-mode/v1/responses`; Settings saves `.../compatible-mode/v1`. Red with c176f553b3's sources on the readiness assertion ("Invalid URL", "Endpoint path names a request route; enter the API base."); green after.

Rewritten on purpose: `test_a_gateway_path_keeps_every_host_and_shape_refusal` asserted that QwenCloud refuses `<gateway>/v1/acct/gw/openai/responses`, which its adapter accepts and sends; QwenCloud's row now omits the `/responses` route (its `/chat/completions/v1` row still refuses).

**Covering run** (150 files: every test file that calls the endpoint contract, the Settings setup persistence or card, the Cohere/Google/QwenCloud adapters or names their URLs; `-n 4`, load 19-67): 663 failed / 9832 passed / 1 error. Re-run of the 663 failing ids with this change's two source files reverted to c176f553b3 (same command): 658 fail the same way (the known `RecoveryRequired: raw_source_selection_changed` family: 289 in `test_console_native_chat_flow.py`, 87 in `test_console_session_settings.py`, the Cohere/Google native-tools files, the plain Settings UI files), and the error is `test_sensitive_llm_logging.py::test_auxiliary_pins_default_openai_endpoint_before_config_changes`, red at base too. The 5 that passed at base in that batch, run alone with the same serial command: 2 fail at head and at base alike (`test_google_native_tools.py::test_list_content_with_tool_calls_keeps_text_parts`, RecoveryRequired; `test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh`), 3 pass at both (order-dependent). `Tests/Chat/test_local_server_discovery.py` whole file 1046 passed; save-scope, connect-key-rows and the QwenCloud contract/adapter files 193 passed; `Tests/CI` 1811 passed; architecture size and identity ratchets (serial) 323 passed; `Tests/lint/ruff_gate.py` OK (the 2 ruff findings left in the test file are at base); preflight passed. Neither source module has a size-ratchet row.

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `Tests/Chat/test_local_server_discovery.py`, `Docs/User_Guide/settings.md`, `Docs/Architecture/llm-providers.md`.

**Review r17cu-rev1 round (2026-11-19).**
- M-2: the rewritten TASK-41700 row left the boundary unpinned. `test_a_gateway_path_keeps_every_host_and_shape_refusal` now asserts QwenCloud refuses `http://127.0.0.1:8000/v1/acct/gw/openai/responses/responses` with `ENDPOINT_ROUTE_REFUSAL`: the contract strips one `/responses`, as the adapter does. Red under a temporary edit that stripped every trailing `/responses` (a `while` for the `if`): the refusal was "" (accepted); green after reverting it.
- I-1 (TASK-41704) also covers QwenCloud's own-host `https://dashscope-intl.aliyuncs.com/compatible-mode`, which saves as `.../compatible-mode/v1` again.
