---
id: TASK-41726
title: "The Settings provider card hides the endpoint contract's reason behind a generic save error"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 16:57
labels:
- follow-up
- settings
- providers
priority: low
references:
- tldw_chatbook/UI/Settings_Modules/category_panels.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41700's review (round 16). Pressing s on a Settings provider card whose endpoint the contract refuses shows "Provider settings are invalid; review the endpoint and model." for every refusal (category_panels.py ~941), and the card's Endpoint row help checks only the KoboldCpp and MLX-LM rules (`typed_endpoint_refusal`), so a route or host refusal is not named there. Console, Discover models and the custom-endpoint editor already show the contract's reason.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The provider card's save error and Endpoint row help name the contract's reason for a refused endpoint
- [x] #2 A pilot test pins the copy for a route refusal and a host refusal
<!-- AC:END -->

## Implementation Plan

1. `typed_endpoint_refusal` (the one function the card's Endpoint row help and its save both call) also returns the contract's own reason for a non-blank endpoint, after KoboldCpp's and MLX-LM's rules, so the save error and the row help name it instead of the generic save error.
2. Pilot test in `Tests/UI/test_settings_provider_save_scope.py` (UI lane): a route refusal and a host refusal, in the row help as typed and as the save result; red at base.
3. User Guide `settings.md`: say the row help and the save name the contract's reason.

## Implementation Notes

`provider_setup_persistence.typed_endpoint_refusal`, the one function both the card's Endpoint row help (`providers_models_card.endpoint_row_copy`) and its save (`category_panels`, before `build_provider_setup_mutation`) call, now ends with the contract's own reason (`endpoint_refusal`) for a non-blank endpoint, after KoboldCpp's native-route rule and MLX-LM's host-and-port rule (whose own copy still wins). So a refused route or host is named in the save result, its toast (already `markup=False`) and the row help as it is typed, instead of "Provider settings are invalid; review the endpoint and model.", which other save failures keep. A blank endpoint (the provider default) is never refused here. No shipped default endpoint is refused by the new check (probe over `_BUILTIN_PROVIDER_ENDPOINTS`, every registry default and the shipped `api_settings` tables).

**Test:** `Tests/UI/test_settings_provider_save_scope.py::test_the_provider_card_names_the_contracts_reason_for_a_refused_endpoint` (pilot, private profile, UI lane via the file's existing census entry, no census change): for OpenAI, typing `https://api.openai.com/v1/responses` and `http://127.1:8000/v1` shows the route refusal and "Endpoint URL must include a valid host and port." in the row help, **s** gives the same copy, and nothing is written. Red with the base `typed_endpoint_refusal`: the row help read "blank uses the provider default", and, with the row-help assertions temporarily removed, the save read "Provider settings are invalid; review the endpoint and model."; green after. Serial cost: 25 s at load 34-37 (`-p no:xdist --timeout=180`).

**Covering run** (150 files: every test file that calls the endpoint contract, the Settings setup persistence or card, the Cohere/Google/QwenCloud adapters or names their URLs; `-n 4`, load 19-67): 663 failed / 9832 passed / 1 error. Re-run of the 663 failing ids with this change's two source files reverted to c176f553b3 (same command): 658 fail the same way (the known `RecoveryRequired: raw_source_selection_changed` family: 289 in `test_console_native_chat_flow.py`, 87 in `test_console_session_settings.py`, the Cohere/Google native-tools files, the plain Settings UI files), and the error is `test_sensitive_llm_logging.py::test_auxiliary_pins_default_openai_endpoint_before_config_changes`, red at base too. The 5 that passed at base in that batch, run alone with the same serial command: 2 fail at head and at base alike (`test_google_native_tools.py::test_list_content_with_tool_calls_keeps_text_parts`, RecoveryRequired; `test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh`), 3 pass at both (order-dependent). `Tests/Chat/test_local_server_discovery.py` whole file 1046 passed; save-scope, connect-key-rows and the QwenCloud contract/adapter files 193 passed; `Tests/CI` 1811 passed; architecture size and identity ratchets (serial) 323 passed; `Tests/lint/ruff_gate.py` OK (the 2 ruff findings left in the test file are at base); preflight passed. Neither source module has a size-ratchet row.

Docs: User Guide `settings.md` says the save result and the Endpoint row's help give the contract's reason, the row as you type.

Files: `tldw_chatbook/Chat/provider_setup_persistence.py`, `Tests/UI/test_settings_provider_save_scope.py`, `Docs/User_Guide/settings.md`.
