---
id: TASK-41116
title: "Eight Backup_Recovery owner-suite nodes are deselected in the push-to-dev job: two are cheap test fixes, one needs a private_sqlite boundary decision"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- tests
- dev-red
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/Backup_Recovery/test_complete_roundtrip.py
- Tests/Backup_Recovery/test_inventory.py
- Tests/Backup_Recovery/test_core_owners.py
- Tests/Backup_Recovery/test_later_rollback_credential_ui.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40307's dev-push job leaves out 8 nodes (`BACKUP_OWNER_SUITES_DESELECTED`, `derived-artifacts.yml` the `--deselect` lines of "Run the Backup_Recovery owner suites"). Until each is fixed or retired, those behaviours are guarded by nothing. TASK-40307's notes give each cause:
- `test_complete_roundtrip.py::test_two_known_profiles_complete_capture_preserves_native_state_and_resumes` and `::test_two_captured_profiles_plan_shared_concrete_roots_for_isolated_restore`: fixture. `(profile / "custom").mkdir(parents=True, mode=0o700)` leaves the parent at the umask mode, and `config_participants.companion_guard` refuses `config_companion_parent_unsafe`. Cheap.
- `test_inventory.py::test_fresh_process_discovery_imports_no_bootstrap_or_services_and_changes_nothing`: drift. The child forbids every `tldw_chatbook.Utils*` import, and since 8bbac5bd20 `tldw_chatbook/__init__.py` imports `Utils.tiktoken_runtime`. Cheap (allow that one module).
- `test_core_owners.py::test_factory_discovery_and_schema_policy_are_runtime_independent`: since 79a46b3b4e `recovery_workflow_schema.open_installed_schema` reaches `DB.private_sqlite`. A test pin or a product boundary decision.
- `test_inventory.py::test_normal_config_and_pure_resolvers_agree`: reads config in a plain test (`RecoveryRequired: raw_source_selection_changed`).
- `test_inventory.py::test_custom_db_and_disabled_features_are_not_external_or_unused`: intermittent (`'included' == 'unsupported'`), cause not investigated.
- `test_complete_roundtrip.py::test_two_captured_profiles_restore_and_open_with_native_content` and `test_later_rollback_credential_ui.py::test_f9_later_rollback_requires_explicit_credential_review`: the shared `native_package` fixture's fixed 60 s `pip install --target`; past it, f9 failed `AssertionError: real exec expected`, cause not investigated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 8 nodes passes and its --deselect line and BACKUP_OWNER_SUITES_DESELECTED entry are removed, or it is retired with the commit that changed what it pinned named
- [ ] #2 The private_sqlite boundary in schema_policy() is decided (test pin or product change) and recorded in the notes
- [ ] #3 The cause of 'real exec expected' in the f9 node is recorded
<!-- AC:END -->
