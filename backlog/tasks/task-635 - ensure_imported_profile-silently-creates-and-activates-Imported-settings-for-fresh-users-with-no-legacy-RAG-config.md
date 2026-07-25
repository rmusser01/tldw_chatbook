---
id: TASK-635
title: >-
  ensure_imported_profile silently creates and activates Imported settings for
  fresh users with no legacy RAG config
status: Done
assignee: []
created_date: '2026-07-25 20:55'
updated_date: '2026-07-25 21:29'
labels:
  - followup
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT 2026-07-25 (scratchpad/uat-refix, refix-report.md finding 2): a brand-new user with no [AppRAGSearchConfig.rag.*] material in config.toml ran Backfill and config.toml silently gained [rag.service] profile = "imported_settings" without ever touching Set active -- the UI's Active:/Editing: rows flipped to a profile the user never chose. ensure_imported_profile() (RAG_Search/simplified/active_config.py) unconditionally snapshots resolve_active_rag_config() into a new writable 'Imported settings' profile and activates it on the very first call to get_shared_rag_service() in the process, with no check for whether the user actually has any pre-existing legacy RAG config worth preserving continuity for. The behavior is only meaningful for upgraders who ran the app before the profile system existed (SP1 fingerprint continuity); a truly fresh install has no legacy collection to protect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh user with no hand-set [AppRAGSearchConfig.rag.*] material never gets an auto-created/auto-activated Imported settings profile; the active pointer stays on the default builtin (hybrid_basic) and config.toml is never written to as a side effect
- [x] #2 A legacy upgrader with hand-set [AppRAGSearchConfig.rag.*] material still gets the first-run Imported settings profile created and activated, with the SP1 fingerprint-continuity invariant preserved
- [x] #3 All existing Tests/RAG/test_first_run_import.py legacy-present cases still pass
- [x] #4 New tests cover the fresh-user no-legacy-sections case (no profile created, pointer untouched)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Root-cause: ensure_imported_profile() (active_config.py) unconditionally snapshots resolve_active_rag_config() into a new "Imported settings" profile and activates it on the first-ever get_shared_rag_service() call in the process, with no check for whether the user has any genuine pre-profile-system legacy RAG config -- so a fresh install gets a silent pointer flip the first time anything (e.g. Backfill) touches the shared service.
2. Study Tests/RAG/test_first_run_import.py's existing SP1 fingerprint-continuity cases first (per task instructions) to understand which currently encode "legacy upgrader" scenarios vs which incidentally rely on the always-import bug.
3. Add _has_legacy_rag_config_material() (task-495-adjacent helper) that returns True only when [AppRAGSearchConfig.rag] resolves to a non-empty dict; gate ensure_imported_profile()'s creation branch on it (the healing branch for an already-existing profile is unaffected).
4. Update existing tests: default the _wire() fixture to the no-legacy baseline: adjust the tests that implicitly relied on unconditional import to explicitly wire a legacy-upgrader scenario via _wire_legacy_rag_config; add new fresh-user tests (no legacy sections -> no profile, pointer untouched; malformed legacy section -> same fail-safe behavior).
5. Verify GREEN: full Tests/RAG/test_first_run_import.py + Tests/RAG/ + Tests/UI/test_settings_rag_profile_region.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: ensure_imported_profile() unconditionally snapshotted resolve_active_rag_config() into a new "Imported settings" profile and activated it on the first-ever get_shared_rag_service() call in the process -- with no check for whether the user had any genuine pre-profile-system legacy config. A brand-new install running Backfill for the first time silently got [rag.service] profile = "imported_settings" written to config.toml, flipping Active:/Editing: to a profile never chosen.

Fix: added _has_legacy_rag_config_material() -- True only when [AppRAGSearchConfig.rag] resolves to a non-empty dict -- and gated ensure_imported_profile()'s creation branch on it. The existing-profile healing branch (for a half-done prior first run) is unaffected: once the profile exists, healing its activation is not a fresh-install concern.

Studied Tests/RAG/test_first_run_import.py first per the task instructions: several existing tests (test_first_run_creates_imported_profile_and_sets_active, both fingerprint-continuity tests, test_ensure_imported_profile_swallows_save_failure, and test_imported_profile_unchanged_when_no_legacy_keys_set) incidentally relied on the always-import bug rather than genuinely exercising a legacy-upgrader scenario. Updated the shared _wire() fixture to default to an explicit no-legacy baseline, then updated those tests to explicitly wire minimal legacy content via _wire_legacy_rag_config so they now represent real upgraders (test_imported_profile_unchanged_when_no_legacy_keys_set renamed to test_imported_profile_unchanged_when_no_legacy_query_time_keys_set, using a non-query-time legacy key as the presence signal so its "nothing merged" assertion is unaffected). Added two new tests: a fresh-user no-legacy-material case and a malformed/non-dict legacy-section case, both asserting no profile is created and the pointer is untouched.

Verified: Tests/RAG/test_first_run_import.py (16 passed), full Tests/RAG/ (567 passed, 8 skipped), Tests/UI/test_settings_rag_profile_region.py (120 passed) -- all green.

Modified: tldw_chatbook/RAG_Search/simplified/active_config.py (_has_legacy_rag_config_material + ensure_imported_profile gate + docstring update); Tests/RAG/test_first_run_import.py (_wire() default, updated legacy-upgrader tests, two new fresh-user tests).
<!-- SECTION:NOTES:END -->
