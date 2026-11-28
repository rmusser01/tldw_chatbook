---
id: TASK-42546
title: "Old system SQLite (3.40) refuses every ChaChaNotes backup candidate validation"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 23:44
updated_date: 2026-11-27 18:38
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

**What changed (AC #3).** On an SQLite older than 3.41.0, every backup and restore review stops at once with "Backup and restore need SQLite 3.41.0 or newer, and this Python uses SQLite 3.40.1. Run Chatbook with a Python that has a newer SQLite (the "Local backup and recovery" guide names some), then try again. The rest of Chatbook works as before." (code `sqlite_version_unsupported`).
- `Backup_Recovery/qualification.py`: `MINIMUM_SQLITE_VERSION = (3, 41, 0)` with the measurement, the one predicate `sqlite_supported()`, `require_supported_sqlite()`, and the constant `SQLITE_VERSION_UNSUPPORTED`. The capability check shared by Complete capture and new replacement (`_release_for_roots`) refuses first, and `replacement_capability` refuses before its safety-copy dependency check, so an old SQLite is the one reason shown.
- `Backup_Recovery/recovery_service.py`: `preview_backup_details` (Create backup's review), `restore_group_requirements` (the restore review's first step), `preview_restore` (both modes) and `start_rollback_preview` refuse before reading anything, because each reads the live databases first and on 3.40 that read already fails (the review then listed valid stores as unavailable, or raised `target_owner_unclassified`, shown as `backup_operation_failed`). `start_restore` and `start_backup` recheck at start. `issue_code` passes the code through and `issue_message` names both versions.
- `Backup_Recovery/sqlite_validation.py`: `_validate_candidate` reports a `sqlite_security_unavailable` refusal as `sqlite_version_unsupported` on such an SQLite. It only relabels a refusal that already happened, so nothing that validated before is refused now.
- `UI/Screens/backup_restore_screen.py`: the rollback review's Availability line adds the reason's message when replacement is unavailable, as the Replace restore review already did.
- `Backup_Recovery/launcher.py` (command-line recovery): "Replacement unavailable: <code>" is followed by its message, and "Recovery refused: sqlite_version_unsupported" by the version message (only that code: the plugin-removal path prints its own help before it raises there). tldw-chatbook-30, whose earlier work held this file, confirmed on 2026-11-27 it has none open on it.
- Inspecting a backup and extracting its files without restoring read only the archive, so they still work on an old SQLite. Ordinary app connections keep `trusted_schema` on, so the rest of the app runs on 3.40 as before (the one other runtime `trusted_schema=OFF` read, the agent-tools upgrade worker, reads `agent_runs.db`, whose schema has no JSON scalar functions).
- Tests (`Tests/Backup_Recovery/test_sqlite_validation.py`, architecture-guards lane, version stubbed with monkeypatch): the capability check refused on 3.40.1 and not on 3.41.0; isolated restore refused before an operation starts; the four review steps refused before any read; replacement naming SQLite before the safety-copy selection (3.41.0 control: the dependency reason); the relabel (3.41.0 keeps `sqlite_security_unavailable`); the copy naming both versions; the command line's restore refusal; the rollback review's Availability line. Each fails without its source change, and each 3.41.0 control passes either way.
- Docs: `Docs/Backup-and-Recovery.md` (availability paragraph: what an older SQLite shows, what still works, how to check a Python, and measured versions: Debian 12's system SQLite 3.40.1, Ubuntu 24.04's Python 3.12 3.45.1, uv-managed macOS builds 3.49.1 and 3.50.4) and the README requirements list. The Linux uv-managed builds were not checked (the container download was refused in this session), so the docs give the check command rather than a Linux version for them.

**Review round 1 (2026-11-27).** Important: Replace restore and later rollback on an old SQLite still ended in `backup_operation_failed` (the review read the target before any capability check); fixed by refusing at the start of `preview_restore` and `start_rollback_preview`. Minor, all fixed: the command line did not name the versions; Create backup's review listed valid stores as unavailable; replacement checked safety copies before SQLite; the version comparison and code were repeated (one predicate and constant now); the copy and docs named no way to a newer SQLite; task hygiene (assignee, notes). The rollback review's Availability line now says why, as the Replace review does.

**Review round 2 (2026-11-27).** All round-1 fixes confirmed. Minor, all fixed: the restore review's linked-group step (`restore_group_requirements`) ran before the gate, costing one extra Review on an old SQLite; the backup case of the review test used an invalid backup suffix, so without its gate it failed on `invalid_backup_suffix` instead of the read trap; the copy named a repository path that installs do not ship, and now names the guide by title. TASK-42650 (filed with this change) had the wrong cause and was rewritten.
<!-- SECTION:NOTES:END -->
