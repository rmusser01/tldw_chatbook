---
id: TASK-69
title: Surface Chatbook Sync v2 profile status
status: Done
labels:
- sync
- sync-v2
- ui-status
priority: medium
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Surfaced Sync v2 profile summary status in the Library Collections running-app workflow. Users can now see read-only profile state, pending work, conflict/error attention, and dataset/device identity context without triggering sync writes.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests and verification recorded
- [x] #3 Documentation updated in task record
- [x] #4 Final summary added
- [x] #5 No known blockers
<!-- DOD:END -->
