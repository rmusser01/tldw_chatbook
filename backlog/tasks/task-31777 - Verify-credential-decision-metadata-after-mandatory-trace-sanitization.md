---
id: TASK-31777
title: Verify credential decision metadata after mandatory trace sanitization
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 23:00'
updated_date: '2026-09-05 23:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The custom OpenAI trace matrix expects a credential-named field in sanitized handler values even though the mandatory recursive filter removes it and exposes the safe decision separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The matrix independently proves real dispatch preserves resolved keyless intent while sanitized trace values omit credential fields
- [x] #2 Safe credential category and overlay metadata remain asserted and complete trace tests pass without sanitizer changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced missing-key failure and inspect dispatcher projection, mandatory credential filtering and bounded decision metadata.
2. Correct the test to distinguish raw handler dispatch from sanitized trace projection, preserving keyless decision and overlay assertions. This exposed a genuine missing-overlay defect: the flag is removed before overlay construction. Preserve only its strict boolean meaning at the sole private overlay call; never restore the credential-named field to stored projections. Add false/absent decision coverage.
3. Run complete trace final-value/redaction tests and scoped static checks; record evidence.
ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Restore existing bounded decision metadata while preserving the mandatory credential filter; no security or provider contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The original test failed on the intentionally filtered api_key_resolved field. Correcting that expectation exposed the genuinely missing credential_decision overlay: its flag had been filtered before annotation. The sole private overlay call now receives only the strict boolean decision from the already-verified values; stored boundary/handler projections remain sanitized. Tests preserve real dispatch keyless intent, sanitized omission, redaction disclosure and bounded credential source, and reject false/absent/nonboolean annotations. The original overlay assertion failed before the runtime repair. Independent scoped review has no remaining findings. Final seven complete Chat files passed 205 tests; whole-file Ruff and changed-function formatting passed. ADR-097 unchanged. XML: /private/tmp/tldw-current-chat-repair-final.xml. Aggregate descriptor warning 209 belongs to the existing summary selection and remains open.
<!-- SECTION:NOTES:END -->
