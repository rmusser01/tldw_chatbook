---
id: TASK-31651
title: >-
  Fold the ingest bulk-Analyze outcomes field into LibraryIngestState and census
  dev's two new ingest methods
status: To Do
assignee: []
created_date: '2026-09-05 15:31'
labels:
  - library
  - refactor
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dev's task-28007 landed after the wave-5 ingest decomposition: it added a flat `_library_ingest_analyze_outcomes` attribute to `LibraryScreen.__init__` and two new screen-resident ingest methods. The wave-5 reconciliation merge bridged the field with a one-off accessor binding on `LibraryIngestController` so the moved `handle_library_ingest_clear_finished` body could keep dev's edit byte-for-byte, but that leaves the ingest state object one field short of its subsystem and leaves two ingest-named methods outside the cluster's own census and decomposition tables. Close both so the ingest subsystem's state boundary and method census are true again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `analyze_outcomes` is a field on `LibraryIngestState`, not a flat `LibraryScreen.__init__` attribute
- [ ] #2 Every screen-side read and write of that field goes through `self._ingest_state.<field>`, matching the ingest cleanup PR's retarget pattern
- [ ] #3 The interim `library_ingest_analyze_outcomes_accessor` constructor binding is removed from `LibraryIngestController` and the moved body reaches the field through the generated state shim like the other 20 fields
- [ ] #4 `handle_library_ingest_analyze_skipped` and `_record_library_ingest_analyze_outcome` each appear in the ingest cluster census as either a mover or an explicitly-reasoned exclusion
- [ ] #5 `Tests/Architecture/test_library_ingest_wiring.py` pins the new field count and the updated mover/exclusion counts, and passes
- [ ] #6 The recipe's ingest rows in sections 8 and 20 state the same numbers the wiring test pins
- [ ] #7 Both size ratchet rows are re-measured and re-pinned in the same commit
- [ ] #8 Dev's task-28007 test additions pass unmodified
<!-- AC:END -->
