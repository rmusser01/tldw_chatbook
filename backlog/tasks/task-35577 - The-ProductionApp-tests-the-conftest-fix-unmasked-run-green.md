---
id: TASK-35577
title: The ProductionApp tests the conftest fix unmasked run green
status: To Do
created_date: 2026-10-19 15:02
dependencies: []
labels:
- follow-up
- tests
priority: medium
references:
- Tests/ProductionApp/conftest.py
- Tests/ProductionApp/test_chat_composition_retirement.py
- Tests/ProductionApp/test_chat_root_state_removal.py
- Tests/ProductionApp/test_workflows_session_lifecycle.py
- Tests/ProductionApp/test_backup_restore_composition.py
- Tests/ProductionApp/test_service_composition_lifecycle.py
- Tests/ProductionApp/test_reactive_ownership_maturity.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by Phase 9's base-vs-head comparison (2026-10-19). On dev, Tests/ProductionApp/conftest.py rewrote HOME, XDG and TLDW_CONFIG_PATH at import, so most of the directory failed at collection and its assertions never ran. Phase 9's TASK-33001.11 fixed that conftest, and the files now collect and run. Some fail, in exactly the same way on dev with only that conftest copied over, so none of this is a Phase 9 regression:
- test_chat_composition_retirement.py (10) and test_chat_root_state_removal.py (2): setup's save_values('chat_defaults', ...) returns False, because the config write raises RecoveryRequired('raw_source_selection_changed'). The participant registered at collection names the test sandbox's config, while the per-test isolation points TLDW_CONFIG_PATH elsewhere. TASK-33001.11 fixed the same mechanism for test_provider_selection_ownership.py with @private_profile_test. Two more nodes fail on the same refusal, run alone: test_service_composition_lifecycle.py::test_runtime_backend_transition_detaches_and_rebinds_notes_organization and test_reactive_ownership_maturity.py::test_registered_routes_use_fresh_production_owners_and_safe_snapshots.
- test_workflows_session_lifecycle.py (7 setup errors): RunHooksEngine.__init__ has required notification_targets and launch_guard since 087fc5693f (2026-09-27), and the fixture predates that.
- test_backup_restore_composition.py (2): its fixture concatenates result.stdout and result.stderr, which are None (the child logs to a file). Behind that, the isolated-restore child fills no 'profile_base' slot (destination_slots never emits 'data_root').
- test_service_composition_lifecycle.py::test_production_app_composes_one_stable_dependency_graph: it asserts notes_organization_sync_service on an unmounted app, but that attribute has been wired after mount since 23a3843dc3.
- test_reactive_ownership_maturity.py: the surrogate-pattern guard flags SimpleNamespace substitutes in test_native_credential_recovery.py and test_workflows_session_lifecycle.py.
These files run in no PR lane (only the manually dispatched backup-recovery qualification workflow lists one of them, test_service_composition_lifecycle.py), so nothing gates on them yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed test either passes under the scratch wrapper, or has a recorded reason it is quarantined, and none fails at setup
- [ ] #2 The tests that save config run on a fresh per-test profile, as test_provider_selection_ownership.py does, so they pass in a shared pytest process
- [ ] #3 The surrogate-pattern guard passes with no allowlist entry added without a recorded review
<!-- AC:END -->
