---
id: TASK-41302
title: "Restored profiles' Needs setup list shows raw owner ids"
status: Done
assignee:
- '@claude'
created_date: 2026-11-16 23:33
updated_date: 2026-11-17 00:02
labels:
- follow-up
- backup
- ux
priority: low
references:
- tldw_chatbook/UI/Screens/backup_restore_screen.py
- tldw_chatbook/Backup_Recovery/recovery_service.py
dependencies:
- TASK-41111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 15 of the 2026-10-23 burn-down from TASK-41111's notes.

**Why.** Backup & Restore > Restored profiles lists pending owners under **Needs setup** by their internal ids (`owner.replace(".", " / ")`, so `tts / profile store`, `models / artifacts`), while every other review names them through `owner_label` since TASK-41111.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current profile and each restored profile list pending owners by their `owner_label` names
- [x] #2 A test pins the names, red before the change
<!-- AC:END -->

## Implementation Plan

1. Render both Needs setup lists through `recovery_service.owner_label`.
2. Update the tests and docs that pin the id form; extend the mounted Restored profiles test.

## Implementation Notes

**Summary.** Both **Needs setup** lists in Backup & Restore ▸ Restored profiles (Current profile, and each restored profile) render pending owners through `recovery_service.owner_label`. They read, for example, "Model files", "Saved voices" and "Server sync state" instead of `models / artifacts`, `tts / voices` and `runtime / sync state`: the names every other review has used since TASK-41111. The import is local to the list render: the service module is already loaded there, and the screen's import time is unchanged.

**Tests and red/green evidence** (load 85-100 on 18 CPUs):
- New `Tests/UI/test_backup_restore_setup_entry.py::test_restored_profiles_needs_setup_names_owners_by_their_labels` (whole-file UI census entry). Current profile pending `models.artifacts` and `tts.voices` lists "• Model files" then "• Saved voices". A restored profile pending `ui.themes` lists "• Themes". No id form appears.
- TASK-40711's `test_restored_profiles_settings_review_approves_config_alone[approved]` pinned the id form (`• runtime / sync state`). It now pins "• Server sync state".
- Both were red with the id rendering temporarily restored ("AssertionError: Current profile ..." and "assert ('• Server sync state' in ..."), and green after.

**Docs.** `Docs/Backup-and-Recovery.md`: the settings, speech and model-files paragraphs name the listed entries by these labels.

**Files.** `tldw_chatbook/UI/Screens/backup_restore_screen.py`, `Tests/UI/test_backup_restore_setup_entry.py`, `Docs/Backup-and-Recovery.md`.
