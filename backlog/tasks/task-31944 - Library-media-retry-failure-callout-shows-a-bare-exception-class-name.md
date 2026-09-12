---
id: TASK-31944
title: Library media - retry failure callout shows a bare exception class name
status: Done
assignee: []
created_date: '2026-09-07 08:25'
updated_date: '2026-09-07 20:34'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR E final review M-4 (2026-09-05): the media browse controller's _retry_failure_reason falls back to the exception's class name when the exception carries no message, so the user reads 'Couldn't retry - RuntimeError'. A short map for the classes that actually occur (timeout, connection, database) would give a reason a user can act on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A retry failure caused by a timeout, a connection error or a database error shows a human-readable reason instead of the exception class name
- [x] #2 An unmapped exception still produces a callout with a usable fallback reason
- [x] #3 The mapping and the fallback are pinned
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the mapped reasons (timeout, connection, sqlite3, the media layer's own DatabaseError) and a privacy pin that non-OS exception text never surfaces. 2. Replace the class-name leg of `_retry_failure_reason` with a small kind → reason map and an `an unexpected error` fallback; keep PR G's path redaction and OS-error message leg byte-identical.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_retry_failure_reason` keeps PR G's two legs (TimeoutError → `timed out`; an OS or sqlite3.OperationalError message, path-redacted) and replaces the bare class name with `_mapped_failure_reason`: connection failures → `the connection failed`; sqlite3 errors AND the media layer's own `DatabaseError` (Client_Media_DB_v2 wraps every sqlite fault in it, so the sqlite3 leg alone was unreachable on the real path — found at review) → `the database could not be read`; anything else → `an unexpected error`. `ConflictError` deliberately stays on the fallback. The DatabaseError import is function-local so the UI controller does not import a DB module at module scope. Privacy: only literals or the OS message can reach the screen (pinned: `Media search failed.` never surfaces; `private-media-failure` kept); both fences still log `exception_type=` for diagnostics. The class name was removed from the screen only. User Guide rows corrected (library.md still claimed a type name; fixed at the final review).
<!-- SECTION:NOTES:END -->
