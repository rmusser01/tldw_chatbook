---
id: TASK-32047
title: Preserve diagnostic log text while redacting only credentials and PII
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 16:58'
updated_date: '2026-09-08 19:29'
labels:
  - logging
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The reported Console failure log replaced ordinary startup and completion messages with whole-line redaction markers. Restore useful diagnostic messages across the Logs view, copy actions, and application log file while protecting credentials and personally identifying values.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary diagnostic messages, exception details and stack frames remain readable in the Logs view, both copy actions and the application log file.
- [x] #2 Credentials and recognizable PII are masked without suppressing surrounding diagnostic fields, timestamps, versions, correlation IDs or non-secret keys.
- [x] #3 Log retention bounds and private file permissions remain enforced.
- [x] #4 Existing versioned trace credential projections are unchanged by the logging correction.
- [x] #5 Real collector, clipboard, file-sink and sanitizer regressions verify both retained diagnostic content and removed sensitive values.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: The owner explicitly replaces the metadata-only application logging policy with credential-and-PII redaction across application log surfaces.

1. Reproduce whole-message loss through the real buffered collector and copy action, and characterize the file admission filter.
2. Amend ADR-029 for the owner's logging policy; retain private storage permissions and the separate trace-storage contract.
3. Use one descriptive, redacted log representation across the view, copy actions and rotating file sink. Narrow credential matching and add PII masking without discarding unrelated diagnostic fields.
4. Preserve the existing trace credential projection while changing logging behavior, and update user-facing log guidance.
5. Run focused sanitizer, collector, file-handler and trace-projection regressions; review the diagnostic inventory delta and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Application log files, the Logs view and both copy actions now preserve ordinary diagnostic text while masking credentials and recognizable PII. Existing private file ownership/permissions, rotation and bounded buffers remain enforced. Previously installed file handlers have the legacy metadata-only filter removed and retain the redacting formatter.

The sanitizer preserves provider/model names, versions, correlation IDs, phases and non-secret keys. Regression cases cover credential aliases, authentication cookies and Digest fields, URL/ODBC credentials, labelled PII and long no-match inputs. The original credentials-v1 string projection remains explicit and unchanged for persisted Console traces.

Updated app.py, Logging_Config.py, UI/Logs_Window.py, Utils/log_sanitizer.py, the frozen trace-redactor import, real collector/file tests and Docs/User_Guide/logs.md. ADR required: yes; amended backlog/decisions/029-local-private-data-boundary.md for the owner-approved credential/PII-only application logging policy. No trace disclosure or schema changes are part of this logging correction.

Validation: 166 targeted collector, clipboard, file-sink, sanitizer and Textual Logs-screen tests passed after applying the output patch. The prior adjacent sanitizer/provider-log run passed 167 tests, and the trace repair passed 295 targeted tests. Review findings were reproduced before fixing them; final output changes were self-reviewed. Modified Python ranges were formatted, git diff --check is clean, and Ruff comparison against HEAD found no introduced findings (existing lint debt remains). Full suite not run.

Final scripts/preflight.sh verification passed all six derived-artifact checks, including the reviewed production diagnostic inventory; no inventory regeneration was needed.

Follow-up verification after updating to dev 7a7493e529: 177 targeted sanitizer, private-file, clipboard, collector and Textual Logs tests passed. Updated the Logs empty-state assertion to describe the approved credentials/PII-only policy. The combined trace fixes pass 407 targeted trace tests; preflight passes and no new Ruff findings were introduced.
<!-- SECTION:NOTES:END -->
