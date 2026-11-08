---
id: TASK-39172
title: "Three test_config_participant_lifetimes.py nodes are red on dev, two because a pause or a re-entrant publication does not raise RecoveryRequired, and the file runs on no PR lane"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- backup
- config
priority: low
references:
- Tests/Backup_Recovery/test_config_participant_lifetimes.py
- tldw_chatbook/Backup_Recovery/bootstrap.py
- tldw_chatbook/config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (settings stream, on base fu-base-r12). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08 (run.sh, `-n 4`, plain, load ~21-27). The same three of the file's nodes fail on both trees:
- `test_two_config_processes_preserve_each_others_keys`: `assert 'serializing' == 'saved'`.
- `test_reentrant_publication_from_derived_scope_refuses_all_effects`: `Failed: DID NOT RAISE <class 'tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired'>`.
- `test_pause_refuses_cached_bootstrap_without_erasing_cache`: `Failed: DID NOT RAISE <class 'tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired'>`.

The file carries no profile marker, and its other nodes pass plain. No PR lane runs it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node passes on dev, or is retired with the commit that changed the behaviour it pinned named; each red is shown first with the same command
- [ ] #2 Each is classified as test drift or a product defect; a pause or re-entrant publication that no longer refuses as designed is a product defect and gets a test that is red before its fix
- [ ] #3 The file runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
