---
id: TASK-31205
title: Collections quick-capture fields bypass shared input validation
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-03 19:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qodo finding on PR #2344 (library_collections_controller.py ~:871, relocated verbatim from LibraryScreen._submit_library_collection_quick_capture): quick-capture URL/title/tags are not run through input_validation.py's shared checks. Pre-existing behavior surfaced by the extraction; fix in the isolated controller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quick-capture fields flow through the shared input-validation seam
- [x] #2 Covering test exercises hostile field input
<!-- AC:END -->

## Implementation Plan

1. Verify the finding on current dev: quick-capture URL is validated, title/tags/note reach CaptureSaveRequest raw.
2. Add the shared-seam checks: validate_navigation_context_text for the single-line title (<=300) and each tag (<=64); validate_text_input (dangerous-pattern check, <=4000) for the freeform note -- prose with angle brackets stays acceptable. Rejection mirrors the existing URL-rejection shape (status + warning notify + reader refresh + return, draft preserved).
3. RED: one mounted integration test exercising hostile title (script pattern), oversized tag, dangerous-pattern note in turn -- each rejected, draft preserved -- then a fully valid save succeeding. Re-query widgets per phase (reader recomposes after each rejection).

ADR required: no
ADR path: N/A
Reason: Closes a validation-gap finding inside one handler via the existing shared seams; no boundary design change.

## Implementation Notes

``_submit_library_collection_quick_capture`` now runs title/tags/note through ``Utils/input_validation.py`` before constructing ``CaptureSaveRequest``: ``validate_navigation_context_text`` for the single-line fields (rejects blank/padded/non-printable/dangerous/oversized; caps 300/64) and ``validate_text_input`` with allow_html=False for the freeform note (cap 4000; only genuinely dangerous patterns are rejected, so ordinary markup-shaped prose survives). Rejection reuses the URL-rejection UX exactly: action-status line, warning notify, reader refresh, draft preserved for retry.

TDD evidence: ``test_quick_capture_hostile_fields_are_rejected_and_valid_still_saves`` (Tests/UI/test_library_collections_capture_reader.py) failed on the unmodified controller (hostile title was accepted and saved) and passes with the fix; the same mounted session then proves a fully valid form still saves with identical assertions to the existing happy-path characterization test. Harness note recorded in the test docstring: widgets must be re-queried after each rejection because the reader recomposes. Suites: capture-reader file 16 passed; characterization + wiring show one failure that reproduces identically on the stashed (unmodified) controller -- pre-existing, unrelated. Ruff: test file formatted clean.

Modified: ``tldw_chatbook/UI/Library_Modules/library_collections_controller.py``, ``Tests/UI/test_library_collections_capture_reader.py``.
