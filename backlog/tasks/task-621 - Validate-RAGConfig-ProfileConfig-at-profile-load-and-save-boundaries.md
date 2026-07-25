---
id: TASK-621
title: Validate RAGConfig/ProfileConfig at profile load and save boundaries
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 17:30'
updated_date: '2026-07-25 17:55'
labels:
  - rag
  - validation
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
RAG profile JSON files are untrusted input: they can be hand-edited, migrated from an older version, or written by a future CLI/import path that never goes through the Settings screen's own pre-save validation (hard_config_errors). RAGConfig.validate() already exists and classifies configuration problems, but it is only wired up for the Settings screen's live editing flow. The profile manager's load path (config_profiles.py) and its save_profile boundary have no validation at all, so a corrupted or hand-edited profile can silently carry broken values, and a non-screen caller can persist a profile that is structurally invalid.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hand-corrupted profile JSON (e.g. a bad enum value or a negative top_k) logs a warning and the profile still loads and degrades gracefully instead of raising or silently disappearing
- [x] #2 A profile file that is genuinely unparseable (structurally broken) still logs an error and is skipped, preserving today's per-file isolation behavior
- [x] #3 Calling save_profile directly with a hard-invalid RAGConfig raises a clear ValueError instead of persisting the invalid config
- [x] #4 The Settings screen's existing save path (which already surfaces validate() errors via hard_config_errors before calling save_profile) is not double-reported
- [x] #5 All existing tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add validate()-logging to ConfigProfileManager._load_custom_profiles (config_profiles.py): after a per-file JSON successfully parses into a ProfileConfig (existing try/except already isolates structural/parse failures per file -- unchanged), run profile.rag_config.validate() defensively (its own inner try/except so a validate() bug can never remove an otherwise-successfully-parsed profile) and log each returned message as a warning. The profile still loads and is registered exactly as before -- this is additive visibility only, so a hand-corrupted value (bad enum, negative top_k) degrades gracefully instead of silently persisting unnoticed.
2. Add a hard-invalid guard to ConfigProfileManager.save_profile: after the existing read-only/collision guards and before _save_one/registration (preserving the transactional OSError-safety contract), call profile.rag_config.validate() and raise ValueError with the joined messages if non-empty. Verified all builtins + every existing test-created profile already pass validate() (confirmed via a throwaway script), so this cannot break clone_profile/create_custom_profile/existing tests. The Settings screen's own pre-save gate (hard_config_errors, called before save_rag_defaults_to_active_profile) already surfaces validate() errors to the user, so this addition only fires for callers that bypass that screen path (CLI/import, e.g. active_config.ensure_imported_profile, which already wraps save_profile in a catch-all try/except).
3. RED: add tests to Tests/RAG/test_config_profiles.py -- (a) test_save_profile_rejects_hard_invalid_rag_config (chunk_overlap >= chunk_size) expects ValueError and confirms no registration/no file written; (b) test_load_degrades_gracefully_on_hand_corrupted_profile_json writes a profile JSON with a bad vector_store.type enum and a negative default_top_k directly to disk, constructs a manager, asserts it does NOT raise, the profile IS loaded with its (corrupted) values intact, and a warning was logged (loguru sink capture, per Tests/Agents/test_agent_runtime_review_hook.py's pattern -- caplog does not capture loguru in this repo). Run to confirm RED.
4. Implement the two validate() call sites; rerun the new tests to confirm GREEN.
5. Run the full Tests/RAG/ suite to confirm no regressions, plus Tests/UI/test_settings_rag_profile_adapter.py (the screen's own save path) to confirm no double-reporting/behavior change there.
6. Commit as feat(rag): validate profiles at load/save boundaries (task-621).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wired RAGConfig.validate() (already existed; the settings adapter's hard_config_errors was its only prior caller) into ConfigProfileManager's two untrusted-input boundaries in config_profiles.py:

(a) Load (_load_custom_profiles): after a per-file JSON successfully parses into a ProfileConfig (the pre-existing outer try/except still isolates genuinely unparseable files -- unchanged), rag_config.validate() runs defensively (its own inner try/except) and any issues are logged as warnings. The profile still loads and registers normally either way -- a hand-corrupted value (bad vector_store.type enum, negative default_top_k) degrades gracefully and visibly in the logs instead of silently persisting or crashing the manager.

(b) Save (save_profile): after the existing read-only/builtin-collision guards and before _save_one/registration (preserving the prior transactional OSError-safety contract), a non-empty validate() result now raises ValueError with the joined messages. save_profile is the single choke point every non-screen caller goes through (clone_profile, create_custom_profile, active_config.ensure_imported_profile, and any future CLI/import path), so guarding there protects all of them without touching each call site. The Settings screen's own pre-save gate (hard_config_errors) already runs RAGConfig.validate() and surfaces field-scoped errors to the user before ever reaching save_rag_defaults_to_active_profile -> save_profile, so in the normal screen flow this is a silent backstop, not a second report of the same problem.

Verified (throwaway script) that every builtin profile's rag_config already passes validate() cleanly, so clone_profile/create_custom_profile of any builtin cannot regress.

Found and fixed one real fallout during verification: Tests/RAG/test_config_profiles.py::test_clone_builtin_creates_writable_copy mutated a clone's chunk_size to 111 without adjusting chunk_overlap (high_accuracy's builtin overlap is 128), which is genuinely hard-invalid under validate() -- changed to 300 (a value still above the overlap) to preserve the test's actual intent (clone/builtin independence). Also found two Tests/UI/test_settings_rag_profile_adapter.py tests (test_hard_config_errors_filters_out_unexposed_field_violations, test_validate_library_rag_defaults_does_not_gate_on_an_unexposed_field_violation) whose fixture setup deliberately round-tripped an already-invalid vector_store.type through save_profile to simulate a pre-existing violation on a field the Library/RAG UI doesn't expose; since the mutated ProfileConfig is the same live object _active_profile() reads (per the wired fixture), the redundant save_profile() call was simply removed -- the in-place mutation is sufficient and no longer trips the new guard, while the tests still exercise exactly what they were meant to (hard_config_errors/validate_library_rag_defaults not gating on unexposed-field violations).

Modified files:
- tldw_chatbook/RAG_Search/config_profiles.py (load-boundary warning logging in _load_custom_profiles; save-boundary ValueError guard + docstring update in save_profile)
- Tests/RAG/test_config_profiles.py (4 new tests; one existing test's chunk_size fixed to a value valid alongside its overlap)
- Tests/UI/test_settings_rag_profile_adapter.py (2 existing tests' fixture setup adjusted to mutate the live profile in place instead of re-saving)

Verification: Tests/RAG/ -> 558 passed, 8 skipped (baseline 552/8 + 2 seam-lock tests from task-620 + 4 new task-621 tests). Tests/UI/test_settings_rag_profile_adapter.py + test_settings_rag_profile_region.py -> 181 passed.
<!-- SECTION:NOTES:END -->
