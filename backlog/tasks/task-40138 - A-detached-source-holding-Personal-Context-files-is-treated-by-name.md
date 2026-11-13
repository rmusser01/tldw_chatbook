---
id: TASK-40138
title: "A detached source holding Personal Context files is treated by name"
status: Done
assignee:
- '@claude'
created_date: 2026-11-10 17:13
updated_date: 2026-11-13 04:48
labels:
- follow-up
- backup
- recovery
- personal-context
priority: medium
references:
- tldw_chatbook/Backup_Recovery/detached_layout.py
- tldw_chatbook/Backup_Recovery/config_adapter.py
- backlog/docs/backup-recovery-owner-inventory.md
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-39148
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** The detached-layout preview (ADR-256 portable imports) refuses any file it does not know as `detached_unknown_content` (`Backup_Recovery/detached_layout.py:1146-1153`). Neither Personal Context database is known (`:1037-1058`).
- So every portable import from a profile that ever started an interview is refused, for a reason the user cannot see.
- The fixed owner list (`:929-956`) already gives `notes.sync_state` an excluded treatment, with its paths in `known` (`:1046-1049`).

**Interview drafts.** They are local-only by policy, and an excluded `_LocalSQLite` yields `intentionally_excluded` rows (`config_adapter.py:693-706`, `:1163-1170`). Their key is keyring-only (`Personal_Context/interview_launch.py:97-113`), so the bytes would be unreadable anywhere else.

**The repository.** Its key cannot come along: detached import is credential-excluded (`detached_layout.py:1`, `:492-493`), and the relocation host runs with a null keyring (`portable_relocation.py:250`). Decision 10 refuses it by name instead.
- If TASK-40134 has landed, its captured adapter is picked up by the detached layout automatically. The refusal must hold either way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A detached source holding the interview drafts database and its sidecars is imported with the drafts as an excluded row and a review line saying they stay on the original device, instead of `detached_unknown_content` (red first)
- [x] #2 A detached source holding the Personal Context repository is refused with `detached_owner_unavailable:personal_context.repository`, and the user sees copy saying its key stays on the original device and to make an encrypted backup with credentials there and restore it with Replace
- [x] #3 The owner inventory doc's detached-layout notes name both treatments
<!-- AC:END -->

## Implementation Plan

1. Re-locate the cited lines on base 26b2e677dd: the fixed owner list, `known` and the refusal now sit in `_owner_items` / `_declared_layout` of `Backup_Recovery/detached_layout.py`; the user-facing surface for a refusal is `portable_relocation.command_main` (Windows portable entry, `Packaging/windows/portable/app_entry.py`).
2. Red tests in `Tests/Backup_Recovery/test_detached_layout.py`: drafts DB and sidecars are imported as `intentionally_excluded` producer rows with a review line; the repository (file or any sidecar) is refused as `detached_owner_unavailable:personal_context.repository`, both with the installed `_LocalSQLite` row and with a stand-in captured `_SQLiteDeclaration` adapter (TASK-40134's shape). Red test in `Tests/Backup_Recovery/test_portable_relocation.py` that `command_main` prints the spec's copy for that code.
3. Drafts: append `personal_context.interview_drafts` at the tail of the fixed owner list, add its four paths to `known`, and add the review line when the drafts file is present.
4. Repository: add its four paths to `known` and refuse by name in `_declared_layout` before any owner opens data, so the refusal does not depend on which adapter declares it (a fixed-list `unsupported` row would only hold for the `_LocalSQLite` row; a captured adapter would be normalized and included by the SQLite loop).
5. Copy: map the code to the spec's sentence in `command_main`.
6. Owner inventory doc: name both detached treatments.

## Implementation Notes

Detached (portable) imports now treat both Personal Context files by name. Base 26b2e677dd; the filing's lines had moved (fixed owner list `detached_layout.py:913-929`, `known` `:1020-1058`, refusal `:1136`).

- **Drafts (AC#1).** `personal_context.interview_drafts` is appended at the tail of the fixed owner list, and its four paths are in `known` beside `notes.sync_state`'s. Its installed `_LocalSQLite` yields `intentionally_excluded` rows, so the import succeeds without the drafts. `produce_detached_archive` adds the review line "Personal Context interview drafts stay on the original device." only when the drafts file is present.
- **Repository (AC#2).** `_declared_layout` refuses it by name, `detached_owner_unavailable:personal_context.repository`, when the file or any sidecar exists, before any owner opens data. Deviation from the spec's mechanism (Q5: a fixed-list row that is `unsupported`): that only holds for today's `_LocalSQLite(excluded=False)` row. A captured `_SQLiteDeclaration` adapter (TASK-40134's shape) would be picked up by `_owner_items`' SQLite loop, normalized and imported, and a fixed-list row would then duplicate its logical id. The check names the path, so it holds with either adapter. The portable terminal (`portable_relocation.command_main`, the Windows portable entry) prints the spec's copy under the code: "Personal Context can't be moved this way because its key stays on the original device. Make an encrypted backup with credentials there, then restore it with Replace."
- **Copy caveat (coordinator ruling, round 14).** The copy's advice is true only once TASK-40134 (capture) and TASK-40135 (readable replace restore) are on dev; before that, an encrypted backup with credentials does not carry Personal Context. This branch merges only together with them.
- **Docs (AC#3).** `backlog/docs/backup-recovery-owner-inventory.md` gets detached-layout notes under the TASK-39110 exclusions naming both treatments. The portable bundle's user doc (`Packaging/windows/portable/README.md`; there is no User Guide page for portable relocation) says the same in two sentences.

**Tests.** `Tests/Backup_Recovery/test_detached_layout.py`: `test_detached_import_keeps_interview_drafts_on_the_original_device[True|False]` and `test_detached_import_refuses_the_personal_context_repository_by_name[installed-|installed--wal|captured-]`. The captured case swaps in an installed stand-in (`Sync_Interop`'s `_SyncAdapter` under the repository's owner id and leaf, resolved by `owner_registry.install_adapters` and `sqlite_validation._installed_owner`) whose validator accepts the file. `Tests/Backup_Recovery/test_portable_relocation.py::test_relocation_names_why_personal_context_cannot_move` runs `command_main` in a child (the real worker subprocess included) and reads the printed copy.
- Red before the fix: drafts[True] and both installed cases raised `detached_unknown_content`; drafts[False] read `set() == {'intentionally_excluded'}`; the captured case read `DID NOT RAISE` (the stand-in imported the repository); the relocation test read `Relocation could not complete: detached_unknown_content`. All six green after.
- Covering files at head, `-n 4`, load 50-77: `test_detached_layout.py`, `test_portable_relocation.py`, `test_detached_source.py`, `test_detached_evidence.py` 143 passed, 5 skipped (Windows-only). `Tests/Architecture/test_backup_owner_inventory.py` + `test_recovery_entrypoints.py` serial: 57 passed.
- Review r14pcb-rev1: the installed refusal case now covers all four suffixes (`""`, `-wal`, `-shm`, `-journal`); a temporary narrowing of the check to `("", "-wal")` turned `[installed--shm]` and `[installed--journal]` red (2 failed, 3 passed), then reverted. The portable README wording is plainer ("Relocation stops if the profile holds Personal Context itself").
- Lane: neither test file is in a PR lane (only one `test_detached_layout.py` node is, under the Architecture guard job). The new nodes ran 12-15 s each at `-n 4`, load 70; laning them is listed as a follow-up rather than edited here (no criterion asks for it).

Files: `tldw_chatbook/Backup_Recovery/detached_layout.py`, `tldw_chatbook/Backup_Recovery/portable_relocation.py`, `Tests/Backup_Recovery/test_detached_layout.py`, `Tests/Backup_Recovery/test_portable_relocation.py`, `backlog/docs/backup-recovery-owner-inventory.md`, `Packaging/windows/portable/README.md`.

