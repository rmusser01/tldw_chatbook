---
id: TASK-69
title: Surface Chatbook Sync v2 profile status
status: Done
labels:
- sync
- sync-v2
- ui-status
priority: medium
modified_files:
- tldw_chatbook/Sync_Interop/sync_profile_status_state.py
- tldw_chatbook/Sync_Interop/sync_state_repository.py
- tldw_chatbook/Library/library_collections_state.py
- tldw_chatbook/Widgets/Library/library_collections_panel.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/Sync_Interop/test_sync_profile_status_state.py
- Tests/Library/test_library_collections_state.py
- Tests/Widgets/test_library_collections_panel.py
- Tests/UI/test_product_maturity_phase39_library_collections.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use the merged Sync v2 profile summary contract in Chatbook running-app workflows so users can see configured mode, pending work, conflicts, cursor identity, and last error without enabling new write-sync controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 profile status is exposed through a small display-state/service seam that maps repository summary states to user-facing labels and recovery severity.
- [x] #2 A running-app surface can render local-only, server-frontend, local-first pending, attention-required, and not-configured summaries without triggering sync mutations.
- [x] #3 Focused tests cover status mapping and prove rendering the summary does not enqueue or push/pull sync work.
- [x] #4 Verification, touched files, and final outcome are recorded in the Backlog task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `SyncProfileStatusDisplay` as a pure Sync v2 profile summary display-state adapter. It maps repository/service summaries into stable labels, severity, counts, dataset/device labels, safe last-error copy, and an explicit read-only notice.

Extended `LibraryCollectionsPanelState` and `LibraryCollectionsPanel` with an optional Sync profile status banner. The banner renders above the Collection workbench and does not introduce sync mutation controls.

Updated `LibraryScreen` to load `sync_scope_service.get_sync_v2_profile_summary` for the active server profile when the Collections mode snapshot refreshes. The lookup uses the same worker-isolated service-call helper as existing sync dry-run reads and returns no banner when no active server profile is available.

Verification:
- Red test confirmed `Tests/Sync_Interop/test_sync_profile_status_state.py` failed on the missing display-state module.
- Red widget test confirmed `LibraryCollectionsPanelState.from_values` rejected `sync_profile_summary` before implementation.
- Red mounted UI test confirmed the Library Collections screen did not render `#library-sync-profile-status` before the screen-level summary load.
- `../../.venv/bin/python -m pytest Tests/Sync_Interop/test_sync_profile_status_state.py Tests/Widgets/test_library_collections_panel.py Tests/Library/test_library_collections_state.py Tests/UI/test_product_maturity_phase39_library_collections.py -q` passed with 21 tests.
- `../../.venv/bin/python -m pytest Tests/Sync_Interop Tests/Library Tests/Widgets/test_library_collections_panel.py Tests/UI/test_product_maturity_phase39_library_collections.py -q` passed with 187 tests.
- `/usr/bin/env PYTHONPYCACHEPREFIX=/tmp/tldw_chatbook_sync_profile_status_pycache ../../.venv/bin/python -m compileall tldw_chatbook/Sync_Interop/sync_profile_status_state.py tldw_chatbook/Library/library_collections_state.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/UI/Screens/library_screen.py` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q -r tldw_chatbook/Sync_Interop/sync_profile_status_state.py tldw_chatbook/Library/library_collections_state.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/UI/Screens/library_screen.py -f json -o /tmp/bandit_chatbook_sync_profile_status.json` passed.
- `git diff --check` passed.

PR review follow-up:
- Validated Sync profile scope values before summary-service calls and gated Library summary loading to server-authoritative runtime state.
- Disabled Rich markup for Sync profile banner text.
- Removed duplicated read-only copy from status detail strings because the banner already renders a dedicated read-only notice.
- Added Google-style docstrings for the public `SyncProfileStatusDisplay` adapter.
- Removed the duplicate manual dangerous-fragment blacklist and relies on shared input validation with `allow_html=False`.

Review verification:
- `../../.venv/bin/python -m pytest Tests/Sync_Interop/test_sync_profile_status_state.py Tests/Widgets/test_library_collections_panel.py Tests/UI/test_product_maturity_phase39_library_collections.py::test_library_collections_surfaces_sync_profile_summary_without_write_sync Tests/UI/test_product_maturity_phase39_library_collections.py::test_library_collections_does_not_load_sync_profile_summary_in_local_mode Tests/UI/test_product_maturity_phase39_library_collections.py::test_library_collections_validates_sync_profile_scope_before_summary_load -q` passed with 10 tests.
- `../../.venv/bin/python -m pytest Tests/Sync_Interop Tests/Library Tests/Widgets/test_library_collections_panel.py Tests/UI/test_product_maturity_phase39_library_collections.py -q` passed with 190 tests.
- `/usr/bin/env PYTHONPYCACHEPREFIX=/tmp/tldw_chatbook_sync_profile_status_pycache ../../.venv/bin/python -m compileall tldw_chatbook/Sync_Interop/sync_profile_status_state.py tldw_chatbook/Library/library_collections_state.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/UI/Screens/library_screen.py` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q -r tldw_chatbook/Sync_Interop/sync_profile_status_state.py tldw_chatbook/Library/library_collections_state.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/UI/Screens/library_screen.py -f json -o /tmp/bandit_chatbook_sync_profile_status_review.json` passed.
- `git diff --check` passed.

Rebase follow-up:
- Rebased `codex/sync-profile-status` onto latest `origin/dev`.
- Preserved newer write-sync promotion labels/tests from `dev` while keeping Sync profile status tests.
- Updated Sync v2 profile summary identity/conflict aggregation for the domain-qualified `source_scope_key` shape now present on `dev`.

Rebase verification:
- `../../.venv/bin/python -m pytest Tests/Sync_Interop/test_sync_profile_status_state.py Tests/Widgets/test_library_collections_panel.py Tests/UI/test_product_maturity_phase39_library_collections.py::test_library_collections_surfaces_sync_profile_summary_without_write_sync Tests/UI/test_product_maturity_phase39_library_collections.py::test_library_collections_does_not_load_sync_profile_summary_in_local_mode Tests/UI/test_product_maturity_phase39_library_collections.py::test_library_collections_validates_sync_profile_scope_before_summary_load -q` passed with 11 tests.
- `../../.venv/bin/python -m pytest Tests/Sync_Interop/test_sync_scope_service.py::test_sync_scope_service_returns_sync_v2_profile_summary Tests/Sync_Interop/test_sync_state_repository.py::test_sync_v2_profile_summary_aggregates_state_counts_and_status Tests/Sync_Interop/test_sync_state_repository.py::test_sync_v2_profile_summary_scopes_none_principal_and_workspace_exactly -q` passed with 3 tests.
- `../../.venv/bin/python -m pytest Tests/Sync_Interop Tests/Library Tests/Widgets/test_library_collections_panel.py Tests/UI/test_product_maturity_phase39_library_collections.py -q` passed with 205 tests.
- `/usr/bin/env PYTHONPYCACHEPREFIX=/tmp/tldw_chatbook_sync_profile_status_pycache ../../.venv/bin/python -m compileall tldw_chatbook/Sync_Interop/sync_profile_status_state.py tldw_chatbook/Sync_Interop/sync_state_repository.py tldw_chatbook/Library/library_collections_state.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/UI/Screens/library_screen.py` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q -r tldw_chatbook/Sync_Interop/sync_profile_status_state.py tldw_chatbook/Sync_Interop/sync_state_repository.py tldw_chatbook/Library/library_collections_state.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/UI/Screens/library_screen.py -f json -o /tmp/bandit_chatbook_sync_profile_status_rebase.json` reported three pre-existing B608 findings in `sync_state_repository.py` outside this branch's diff; no new Bandit finding is on changed lines.
- `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Surfaced Sync v2 profile summary status in the Library Collections running-app workflow and addressed PR review feedback. Users can see read-only profile state, pending work, conflict/error attention, and dataset/device identity context without triggering sync writes. Review follow-up validates sync scope inputs before summary loads, gates status loading to server runtime state, disables Rich markup for banner text, removes duplicated read-only detail copy, and documents the public display adapter. Rebase follow-up preserved newer write-sync promotion coverage from dev and updated Sync v2 profile summary aggregation to remain compatible with domain-qualified sync scope keys.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests and verification recorded
- [x] #3 Documentation updated in task record
- [x] #4 Final summary added
- [x] #5 No known blockers
<!-- DOD:END -->
