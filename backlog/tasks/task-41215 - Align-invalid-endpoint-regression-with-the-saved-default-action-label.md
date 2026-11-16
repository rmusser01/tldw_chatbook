---
id: TASK-41215
title: Preserve provider contract refusal reasons in Console settings
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15 22:35'
updated_date: '2026-11-16 00:05'
labels:
  - ci
  - console
  - tests
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/43750308318/job/132962026419
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console settings validator discards the provider endpoint contract reason and substitutes a generic URL error before the modal can show its precise refusal. Preserve the existing contract decisions and expose their safe actionable reasons, including query and ambiguous suffix refusal, while retaining secret redaction and corrected endpoint saves.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both invalid endpoint cases reach and verify the contract refusal reason without exposing the query secret.
- [x] #2 Correcting each endpoint permits the existing save-default submission with the corrected value.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A (implements the existing provider endpoint contract and TASK-40300 first-safe-error policy)
Reason: propagate the existing contract refusal through Console settings validation without changing accepted endpoints, storage or interfaces.
1. Retain the two current CI failures and the deeper local refusal-copy failures; trace validation before the modal default-save check and read TASK-40300.
2. Keep the current action label expectation and return the existing endpoint_refusal reason from the shared settings validator, removing its redundant boolean wrapper.
3. Intentionally update older generic-error copy pins to the precise contract reason; preserve query redaction, endpoint refusal and corrected-submission assertions.
4. Run targeted validator, field-label and endpoint-save regressions, independently review and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The shared Console session validator now appends the existing endpoint_refusal safe reason directly instead of discarding it behind a boolean and generic URL error. Removed the unused wrapper, updated five older generic-copy pins, and corrected the invalid-draft action label expectation. Exact query/suffix refusal reasons, secret redaction and corrected endpoint submission assertions remain intact.

Validation: eight targeted validator, field-label and full endpoint refusal/correction UI cases pass. Scoped Ruff and independent review pass. Files: Chat/console_session_settings.py, Tests/Chat/test_console_session_settings.py, Tests/UI/test_model_config_field_labels.py and test_console_settings_saved_defaults.py.

ADR required: no; implements TASK-40300 first-safe-contract-error policy without changing accepted endpoints or interfaces.
<!-- SECTION:NOTES:END -->
