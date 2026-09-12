---
id: TASK-23108
title: Settings surfaces raw exception text in user-facing status and toasts
status: Done
assignee: []
created_date: '2026-08-28 14:06'
updated_date: '2026-08-29 02:24'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three paths hand exception reprs straight to the user: settings_screen.py:10871 ('Model discovery failed: {exc}'), :11006 ('Could not save discovered models: {exc}'), :14281 ('Backfill failed: {e}' toast). Users need a plain-language summary with a next step; technical detail belongs in logs/Diagnostics. Existing redact_secret_text wrapping must be preserved. P2 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tldw-chatbook-ui-screens-settings-screen-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three named paths present a plain-language failure summary with a suggested next step; raw exception text is not the primary user-facing message
- [ ] #2 Technical detail remains reachable (log or Diagnostics) and secret redaction is preserved
- [ ] #3 A sweep of settings_screen.py finds no other user-facing f-string interpolating a bare exception
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The three named failure paths present a plain-language summary with a next step via failure_status_text(), and the backfill partial-failure toast no longer pastes the raw last error. The review round caught a security regression introduced by the first cut: the new copy tells users 'Details are in Logs (F8)', and the same change had started writing exception message text to the rotating file sink after only redact_secret_text -- whose regex matches 'X = value' assignments and therefore passes '?key=<token>' URLs and 'Authorization: Bearer ...' straight through, where the pre-existing code had logged the exception type name only. All these paths are back to type-name-only, confirmed at statement level by the diagnostic-inventory drift report. The sink itself still has no redaction; that gap is filed separately. PR #2170.
<!-- SECTION:NOTES:END -->
