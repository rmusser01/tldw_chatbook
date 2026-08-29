---
id: TASK-23190
title: >-
  Settings log sink has no redaction and redact_secret_text misses header and
  query-string secrets
status: To Do
assignee: []
created_date: '2026-08-29 02:25'
labels:
  - security
  - settings
  - logging
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-23108 sends users to the log for failure detail ('Details are in Logs (F8)') and its user-facing paths were reduced to logging exception TYPE names only, precisely because message text is not safe to write there. The residual gap is the sink: the rotating file handler in Logging_Config.py applies no redaction (only the in-app Logs buffer runs redact_log_line), and redact_secret_text's _SECRET_ASSIGNMENT_PATTERN matches 'X = value' assignments only -- so an 'Authorization: Bearer sk-...' header or a '?key=<token>' URL reaches disk unchanged from any OTHER caller that logs exception text. Standing project rule (loguru diagnose incident) is to fix this at the sink rather than at each call site. Raised by the Qodo review of PR #2170 and by the TASK-23108 review round.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Secrets in header form ('Authorization: Bearer <token>') and query-string form ('?key=<token>', '?api_key=<token>') are redacted before reaching the rotating file sink
- [ ] #2 Redaction is applied at the sink so it covers callers that do not opt in, not only paths that call redact_secret_text explicitly
- [ ] #3 A test writes each secret shape through the real logging configuration and asserts the on-disk record contains no secret material
- [ ] #4 Existing type-name-only call sites are unaffected and no diagnostic gains new interpolation
<!-- AC:END -->
