---
id: TASK-41941
title: "Remaining Console code-identity checks: compact model bar setup, mounted capture admission and the Preparing readers"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- console
- adr-356
priority: medium
references:
- tldw_chatbook/Widgets/compact_model_bar.py
- tldw_chatbook/UI/Console_Modules/turn_admission.py
- tldw_chatbook/UI/Console_Modules/composer_run_controls.py
- Tests/Performance/test_compact_setup_ownership.py
dependencies:
- TASK-41642
- TASK-41643
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41642/41643 moved skill discovery, trust preparation and the received route to registrations (ADR-356 rule 4). Left on code identity:
- the compact model bar's own setup: `_source_current`, `_function_current`, `_WIDGET_SOURCE`, `_PROOF_HELPERS` and `config._COMPACT_MODEL_CONFIG_SOURCE` (its Performance tests pin the refusals: `test_compact_setup_ownership.py` cases custom_body/body/queued_body/malformed);
- the mounted capture path in `UI/Console_Modules/turn_admission.py`: `stock_mounted_capture`, `precapture`, `stock_rag_depth_current`, `is_stock_session_provider` (it still refuses a wrapped scratch provider, so TASK-41643's wrapped-scratch case only proves admission);
- the Preparing screen and controller readers and `stock_native_methods`' other callers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed site decides from registrations or runtime-owned state, not code identity
- [ ] #2 Wrapped collaborators behave as stock; foreign, per-instance and subclass overrides are refused (tests)
- [ ] #3 Tests that pin a code-identity refusal are retargeted to a state change, and the ratchet pins drop by the removed reads
<!-- AC:END -->
