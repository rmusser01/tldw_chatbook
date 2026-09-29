---
id: TASK-33370
title: 135 DB and UI tests fail with RecoveryRequired('raw_source_selection_changed')
  under the per-test environment redirect
status: To Do
created_date: 2026-09-28 20:12
dependencies:
- TASK-33260
labels:
- testing
- backup-recovery
- tech-debt
priority: high
updated_date: 2026-09-29 02:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A local run of Tests/DB, Tests/ChaChaNotesDB and the files calling search_conversations_page produced 187 distinct failure tracebacks on dev. 135 of them raise tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired('raw_source_selection_changed').

The affected files:
- mostly the Subscriptions/Watchlists DB suites (test_subscriptions_db_watchlists*.py, test_subscriptions_db*.py)
- Tests/UI/test_personas_lore.py (46 setup errors)
- several ChaChaNotes migration suites

These tests are blind: they report the recovery bootstrap, not the behaviour they pin. PERF-01 (TASK-33260) hit the same class in the perf guards, which failed when build_test_app_config -> load_settings ran under the per-test env redirect. It fixed them with the conftest's existing @pytest.mark.bootstrap_profile. PR CI lanes do not run most of these files, so nothing red surfaced. Found 2026-09-28 while verifying PERF-02 (PR #2887) and PERF-01 (PR #2888): the failures reproduce on the unchanged base commit 9cd9aad65f (dev 48019b1914 plus the audit docs commit), so they are pre-existing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The shared cause is identified and fixed at the common seam (conftest or fixture), not by marking tests one by one, unless per-test marking is shown to be the correct contract
- [ ] #2 The previously RecoveryRequired-blinded tests run their real assertions; any that then fail for another reason are fixed or listed with a named owner
- [ ] #3 backlog/docs/lessons-testing-evidence.md records how to recognise this failure class
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Scope addition (2026-09-28, found while verifying PERF-06): Tests/test_config_*.py fails 186 tests identically at base c174e30f6b and on the PERF-06 branch. The TASK-32804.1 warm-read tests skip with 'raw_source_selection_changed' for the same reason. Under @pytest.mark.bootstrap_profile the same config reads work, so the marker, or the shared conftest seam it uses, is the likely fix for this set too.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
