---
id: TASK-82
title: Functionalize Settings Privacy and Security posture
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Settings > Privacy & Security a useful read-only privacy posture and recovery panel that exposes credential, encryption, redaction, and local/server boundary status without leaking raw secrets or adding unsafe credential mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Privacy & Security renders current encryption, redaction, sensitive-config, provider credential source, and data-boundary posture as structured rows.
- [x] #2 Users can run the existing privacy check and navigate to relevant recovery surfaces without raw secret exposure.
- [x] #3 Inspector copy explains the selected privacy controls and recovery paths instead of generic Settings guidance.
- [x] #4 Unavailable encryption or credential mutation paths are explicitly labeled as deferred/password-gated rather than appearing broken.
- [x] #5 Focused tests cover posture calculation, redaction, recovery navigation, and mounted Privacy & Security rendering.
- [x] #6 Actual CDP/Textual-web screenshot QA is captured and user-approved before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This slice presents existing privacy/config state and adds recovery navigation while preserving the current credential/encryption service boundary. It does not introduce a storage schema, sync/conflict policy, data ownership change, provider/runtime boundary, security policy, dependency, or long-lived application structure.

1. Add pure helper tests for redacted Privacy & Security posture calculation.
2. Implement the smallest helper module for encryption, redaction, provider credential-source, secret-count, and data-boundary status.
3. Add mounted Settings regressions for structured Privacy rendering, no raw secret exposure, and existing Check Privacy behavior.
4. Wire SettingsScreen Privacy & Security to the helper while preserving read-only Save/Revert behavior.
5. Add mounted recovery-navigation and inspector-guidance regressions.
6. Implement Open Providers & Models, Open Advanced Config, and category-specific inspector copy.
7. Run focused pytest verification and git diff hygiene.
8. Capture actual Textual-web/CDP screenshot evidence and get user approval before PR.
9. Update TASK-82 notes and create a small PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `settings_privacy_security.py` as a pure, redacted posture helper for encryption status, sensitive config counts, provider env-var status, provider config secret counts, and local/server boundary copy.
- Reworked Settings > Privacy & Security into a structured read-only pane with posture rows, credential-source rows, visible recovery actions, and explicit password-gated/deferred mutation messaging.
- Added recovery actions that route users to Providers & Models and Advanced Config without exposing raw credential values.
- Replaced generic inspector guidance for Privacy & Security with category-specific guidance about credential source, recovery, and redaction boundaries.
- Added helper and mounted Settings regressions covering posture calculation, redaction, visible rendering, and recovery navigation.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_privacy_security.py Tests/UI/test_settings_configuration_hub.py --tb=short` reported `196 passed, 1 warning`.
- Verification: `git diff --check` returned clean.
- QA screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/settings/privacy-security-guided-2026-06-08.png` (`2400x1522` PNG) was captured via Textual-web/CDP and approved by the user.
- ADR required: no. This presents existing privacy/config state and recovery navigation only; no storage/schema/sync/security policy/runtime boundary changes were introduced.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settings > Privacy & Security now functions as a redacted privacy posture and recovery surface instead of static placeholder copy. It reports credential-source posture, encryption status, secret-count posture, and data-boundary status, with recovery navigation to the existing provider and advanced config surfaces.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
