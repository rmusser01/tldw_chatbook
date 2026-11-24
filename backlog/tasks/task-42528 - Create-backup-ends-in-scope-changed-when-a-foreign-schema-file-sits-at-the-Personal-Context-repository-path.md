---
id: TASK-42528
title: "Create backup ends in scope_changed when a foreign-schema file sits at the Personal Context repository path"
status: To Do
assignee: []
created_date: 2026-11-24 01:38
labels:
- follow-up
- backup-recovery
- personal-context
priority: low
references:
- tldw_chatbook/Personal_Context/recovery.py
- tldw_chatbook/Backup_Recovery/capture_service.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/DB/private_sqlite.py
dependencies:
- TASK-40134
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found at TASK-34100.16.1's rebase onto dev a6df4d8d48 (2026-11-19). Filed from its round-24 port onto dev b31e3e9241 (2026-11-23), where it was re-checked and traced.

**What happens.** A file at the Personal Context repository path (`tldw_chatbook_personal_context.db` in the user data folder) that is SQLite but not the repository's schema previews as one `unsupported` row. With "allow partial", Review succeeds and Create is enabled. Create then ends in `scope_changed`, although nothing changed between Review and Create, and Review again gives the same result every time. So any profile with such a file can never make a backup, and the message tells the user to re-review, which cannot help.

**Cause (traced on the port tree with a probe that wrapped `_Repository.discover`).**
1. Review discovers under `_preview_reads`. `_Repository.discover` opens the file through `connect_private_sqlite("recovery.personal_context.repository", read_only=True)`, validation reports a schema issue, and the row is `unsupported`.
2. Create runs the same discovery again under the maintenance session (`MaintenanceSession._discover_capture_inventory`, `session._discovery_reads()`). `_capture_names` holds no `unsupported` row, so the repository file is not in the held roots, and the open is refused with `RecoveryRequired("capture_source_outside_scope")` (`storage_admission._acquire_capture_storage`).
3. `_Repository.validate` catches only `OSError`, `ValueError` and `sqlite3.Error`, so the refusal escapes. The profile's whole discovery fails (`config_discovery_failure`, `dependency_unavailable`; every profile row is gone). The maintained digest differs from the reviewed one, and `MaintenanceSession._discover_capture_inventory` raises `ValueError("scope_changed")`.

A real repository is not affected: it is `included`, `_capture_names` holds it, and Create succeeds (verified with a real `PersonalContextRepository`, in its own `delete` mode and forced to WAL, in TASK-34100.16.1.1's notes). Untested but likely affected: any other repository row that is present but not `included`, such as `unavailable` while a first link is in progress (TASK-40134 decision 7).

**Reproductions.**
- Dev a6df4d8d48, 2026-11-19 (TASK-34100.16.1's rebase notes): `test_local_sqlite_wal_lifecycle_does_not_void_a_reviewed_scope[personal_context.repository]` ended in `scope_changed` on pure dev, with or without WAL activity.
- Port of TASK-34100.16.1 onto dev b31e3e9241, 2026-11-23:
  - that case, restored temporarily: `ValueError: scope_changed`, raised by `MaintenanceSession._discover_capture_inventory`;
  - a probe with a foreign file (`CREATE TABLE t (x)`) in `delete` and in WAL mode, with no held connection: two Reviews give the same digest, and Create ends in `scope_changed` both times;
  - the traced probe above: Review's two discoveries show one `unsupported` repository row and the same digest; the maintained discovery shows `config_discovery_failure` and no profile rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a foreign-schema SQLite file at the repository path and "allow partial" on, Review then Create backup succeeds; the archive does not hold the file, and its preview row stays `unsupported`
- [ ] #2 The discovery Create runs under maintenance gives the same scope digest as Review for any present repository file that is not `included`, so Create never ends in `scope_changed` when nothing changed
- [ ] #3 A valid repository is still captured, and a repository mid first link is still refused by name, as before
- [ ] #4 Regression tests on a real temp HOME cover the foreign-schema file (red before the fix) and the mid-first-link repository
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Recommendation (from the trace, not implemented).** Make the discovery Create runs under maintenance classify the repository file the same way Review does, without needing the file held. The smallest change is in `Personal_Context/recovery.py`: `_Repository.validate` treats the admission refusal (`RecoveryRequired`) like its other open failures, returning `personal_context_validation_unavailable`. The row then stays `unsupported` in both discoveries, and the file is never read during capture. Check whether any other owner's `discover` opens a file it does not hold under maintenance (grep for `connect_private_sqlite` inside a `discover` or `validate`) and give them the same rule. The alternative, holding present non-`included` rows in `_capture_names`, would enroll a file that is never captured, which ADR-126's narrow-scope rule argues against.
<!-- SECTION:NOTES:END -->
