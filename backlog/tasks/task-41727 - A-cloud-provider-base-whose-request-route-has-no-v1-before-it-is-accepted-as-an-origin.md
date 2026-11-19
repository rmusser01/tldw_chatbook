---
id: TASK-41727
title: "A cloud provider base whose request route has no v1 before it is accepted as an origin"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 16:57
labels:
- follow-up
- providers
priority: low
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
dependencies:
- TASK-41704
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41700's review (round 16), predates it. The generic rule reads a cloud path with no `v1` segment as an origin, so `https://api.openai.com/responses` is accepted and the adapter appends its own route after `/responses`. Close to TASK-41704's scope (the generic rule's `/v1` handling for native cloud bases).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A cloud base ending in a request route with no `v1` before it is refused with the route refusal
- [x] #2 A test pins one such base per adapter family, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Same contract change as TASK-41704: a cloud path with no `v1` goes through the relaxed reading, which refuses a route (`/responses`, `/chat`, `/embeddings`, a legacy `/completions`) with `ENDPOINT_ROUTE_REFUSAL`.
2. Test per adapter case in `Tests/Chat/test_local_server_discovery.py`: the contract refuses `<host>/<route>` with the route refusal and the adapter sends nothing; red at base.

## Implementation Notes

Fixed by TASK-41704's contract change: a first-party cloud key's path with no `v1` now goes through the relaxed reading, whose `_names_a_route` refuses a trailing `/responses`, `/chat`, `/embeddings` or legacy `/completions` with `ENDPOINT_ROUTE_REFUSAL` ("Endpoint path names a request route; enter the API base."). Before, such a path was an origin (`https://api.openai.com/responses` was accepted and the adapter posted `.../responses/chat/completions`). Z.ai is unchanged (its preset reading takes any path as its base, TASK-34100.6), and QwenCloud's own trailing `/responses` is its adapter's route, read by TASK-41722.

**Test:** `Tests/Chat/test_local_server_discovery.py::test_a_cloud_base_ending_in_a_route_without_v1_is_refused`, one node per cloud dispatch case (13: OpenAI, Anthropic, Cohere, Google, Hugging Face standard and router, Groq, OpenRouter, DeepSeek, Mistral and `mistralai`, Moonshot, QwenCloud): the contract gives the route refusal, and through `chat_api_call` nothing is sent and the adapter raises "<Name> API URL: <route refusal>". Moonshot uses `/embeddings`, because its own resolver already refuses `/responses` with its own copy. Red with c176f553b3's sources: 13 of 13 on the contract assertion (the base was accepted); green after.

**Covering run** (150 files: every test file that calls the endpoint contract, the Settings setup persistence or card, the Cohere/Google/QwenCloud adapters or names their URLs; `-n 4`, load 19-67): 663 failed / 9832 passed / 1 error. Re-run of the 663 failing ids with this change's two source files reverted to c176f553b3 (same command): 658 fail the same way (the known `RecoveryRequired: raw_source_selection_changed` family: 289 in `test_console_native_chat_flow.py`, 87 in `test_console_session_settings.py`, the Cohere/Google native-tools files, the plain Settings UI files), and the error is `test_sensitive_llm_logging.py::test_auxiliary_pins_default_openai_endpoint_before_config_changes`, red at base too. The 5 that passed at base in that batch, run alone with the same serial command: 2 fail at head and at base alike (`test_google_native_tools.py::test_list_content_with_tool_calls_keeps_text_parts`, RecoveryRequired; `test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh`), 3 pass at both (order-dependent). `Tests/Chat/test_local_server_discovery.py` whole file 1046 passed; save-scope, connect-key-rows and the QwenCloud contract/adapter files 193 passed; `Tests/CI` 1811 passed; architecture size and identity ratchets (serial) 323 passed; `Tests/lint/ruff_gate.py` OK (the 2 ruff findings left in the test file are at base); preflight passed. Neither source module has a size-ratchet row.

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `Tests/Chat/test_local_server_discovery.py`, `Docs/User_Guide/settings.md`, `Docs/Architecture/llm-providers.md`.
