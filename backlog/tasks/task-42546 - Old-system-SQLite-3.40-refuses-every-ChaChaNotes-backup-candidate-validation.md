---
id: TASK-42546
title: "Old system SQLite (3.40) refuses every ChaChaNotes backup candidate validation"
status: Done
assignee: []
created_date: 2026-11-24 23:44
updated_date: 2026-11-27 17:10
labels:
- follow-up
- backup-recovery
- linux
priority: medium
references:
- tldw_chatbook/Backup_Recovery/sqlite_validation.py
- tldw_chatbook/DB/private_sqlite.py
- backlog/docs/lessons-testing-evidence.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while verifying TASK-42513/42517 on Linux (2026-11-24). On Python 3.12 linked against SQLite 3.40.1 (the `python:3.12-bookworm` image, which is Debian 12's system library), every ChaChaNotes backup or restore candidate validation is refused with `sqlite_security_unavailable`.

The refusal comes from `open_recovery_validation` -> `_restrict_connection`. That function sets `PRAGMA trusted_schema=OFF`, and its next statement fails with `sqlite3.DatabaseError: malformed database schema (console_trace_request_headers) - unsafe use of json_valid()`: on that SQLite version, the `json_valid()` in a v56 CHECK constraint is not accepted with an untrusted schema. `_validate_candidate` maps the error to `sqlite_security_unavailable`, so the user sees a security refusal, not a version problem.

Evidence:
- The Dreams tracking riders step in that container had 94 failures in files TASK-42513/42517 did not change: native-76 compatibility (43), v78 (17), v79 (13) and durable fleet (6), all on this refusal, plus 18 `test_finite_db_retirement.py` drain checks.
- The same tree on Ubuntu 24.04 (Python 3.12.3, SQLite 3.45.1, as on CI's ubuntu-latest) passes all but those 18 drain checks, which fail there at base too.
- macOS (3.49.1) passes.

Real users whose Python links an older system SQLite (Debian 12, older distribution Pythons) would be unable to back up or restore ChaChaNotes. The normal app keeps working, because ordinary connections keep `trusted_schema` on.

This is a product decision, so nothing is implemented here. Options:
- **Recommended:** a minimum SQLite version check for backup/restore validation that names the version found and the version needed, in place of the security refusal. The exact minimum (somewhere in 3.41-3.45) needs a bisect.
- A validation path that tolerates the older SQLite without weakening `trusted_schema=OFF`, for example by validating the JSON-checking constraints separately.
- Declaring the older SQLite unsupported in the install docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner chooses the behaviour for an unsupported SQLite (refuse with clear copy, handle, or document as unsupported), recorded in the task
- [x] #2 The minimum SQLite version that validates a current ChaChaNotes candidate under `trusted_schema=OFF` is measured, not assumed
- [x] #3 On an older SQLite the user sees the chosen outcome, with copy that names the SQLite version, not `sqlite_security_unavailable`; a test covers it (a stubbed version check is acceptable)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decision (2026-11-27):** option 1, refuse with clear copy. On an SQLite too old for backup and restore validation, backup and restore refuse with copy that names the SQLite version found and the version needed, in place of `sqlite_security_unavailable` (which today reaches the user as the generic `backup_operation_failed`).

**What fails on 3.40.1 (measured 2026-11-27, `python:3.12-bookworm`, offline probe of each construct in its own file database, reopened with `PRAGMA trusted_schema=OFF`):** every JSON scalar function used in the schema is refused as unsafe: `json_valid`, `json_type`, `json_extract`, `json_array_length` and `json` in CHECK constraints, `json_extract` in an index expression, and `json_object` and `json_set` in trigger bodies. The table-valued `json_each` in a trigger or view is accepted. SQLite 3.49.1 (macOS, uv's CPython 3.12.11) accepts all ten. So the refusal is not specific to `console_trace_request_headers`: any owner whose schema embeds a JSON scalar function is affected.

**Minimum: SQLite 3.41.0 (AC #2, measured 2026-11-27).** The same probe against SQLite built from sqlite.org's source tarballs in that container: 3.40.1 refused the same eight (so it is the release, not Debian's build flags); 3.41.0, 3.41.2, 3.42.0, 3.43.0, 3.43.2, 3.44.0, 3.44.2, 3.45.0, 3.45.1 and 3.45.3 accepted all ten. With the real code (the committed tree installed with the architecture-guards job's install line, HOME an empty scratch folder): a fresh current ChaChaNotes database validated the way backup and restore validate it returns `()` on 3.41.0 and `('sqlite_version_unsupported',)` on 3.40.1; `test_sqlite_validation.py`, `test_chachanotes_native76_compatibility.py`, the v78 and v79 migration files and `test_durable_fleet_messages.py` give 180 passed on 3.41.0 and 100 failed / 80 passed on 3.40.1.

**What changed (AC #3).**
- `Backup_Recovery/qualification.py`: `MINIMUM_SQLITE_VERSION = (3, 41, 0)` with the measurement, `require_supported_sqlite()`, and the shared capability check `_release_for_roots` returns `(False, "sqlite_version_unsupported")` first. That one check makes Create backup unavailable with its reason beside the button (`create_unavailable_reason`), and Replace restore and later rollback unavailable, at review and again at start.
- `Backup_Recovery/recovery_service.py`: `start_restore` calls `require_supported_sqlite()` before either mode, because an isolated restore has no capability check; `issue_code` passes the code through, and `issue_message` names both versions: "Backup and restore need SQLite 3.41.0 or newer, and this Python uses SQLite 3.40.1. Run Chatbook with a Python built with a newer SQLite, then try again. The rest of Chatbook works as before."
- `Backup_Recovery/sqlite_validation.py`: `_validate_candidate` reports a `sqlite_security_unavailable` refusal as `sqlite_version_unsupported` when SQLite is older than 3.41.0. It only relabels a refusal that already happened, so nothing that validated before is refused now.
- Ordinary app connections are untouched (they keep `trusted_schema` on), so the rest of the app runs on 3.40 as before.
- Tests (`Tests/Backup_Recovery/test_sqlite_validation.py`, architecture-guards lane, stubbed version): capability refused on 3.40.1 and not on 3.41.0, isolated restore refused before an operation starts, the relabel (with 3.41.0 keeping `sqlite_security_unavailable`), and the copy naming both versions. Without the source change the four refusal tests fail and the two 3.41.0 controls pass.
- Docs: `Docs/Backup-and-Recovery.md` (availability paragraph) and the README requirements list.
<!-- SECTION:NOTES:END -->
