---
id: TASK-31204
title: Export destination path normalization bypasses shared path validation
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-03 19:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qodo finding on PR #2344 (library_export_controller.py ~:1275, relocated verbatim from LibraryScreen._apply_library_export_destination): the normalized destination path is not run through path_validation.py's shared checks. Pre-existing behavior surfaced by the extraction; fix in the isolated controller, not in the move PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Destination path flows through the shared path-validation seam
- [x] #2 Covering test exercises a hostile path input
<!-- AC:END -->

## Implementation Plan

1. Verify the finding shape on current dev: the raw FileSave pick IS validated, but the .zip-normalized path (the one actually written) is not re-checked.
2. RED: focused tests against a minimally-wired LibraryExportController (state accessor + screen stub) asserting the normalized path flows through validate_path_simple and a rejected normalized path leaves the form untouched.
3. Re-validate the normalized path inside the existing try/except so both gates share the reject-and-notify path.

ADR required: no
ADR path: N/A
Reason: Closes a validation-gap finding inside one method using the module's existing shared seam; no boundary design change.

## Implementation Notes

``_apply_library_export_destination`` now validates the ``.zip``-normalized destination with ``validate_path_simple`` inside the same try/except as the raw pick, so a suffix-rewritten path that fails validation is rejected with the existing warning notify and the form stays untouched. Verified first that the finding was a real ordering gap (raw validated at :1375, normalized stored at :1385 unchecked) rather than a missing raw gate.

TDD evidence: ``Tests/UI/test_library_export_destination_validation.py`` (new) -- the normalized-seam and normalized-rejection tests failed on unmodified dev; the hostile-raw-pick test passed pre-fix (pinning the existing first gate) and still passes. Post-fix: 3 new + characterization suite green (8 passed). Ruff clean on the new test file; controller file delta zero fixables.

Modified: ``tldw_chatbook/UI/Library_Modules/library_export_controller.py``, ``Tests/UI/test_library_export_destination_validation.py`` (new).
