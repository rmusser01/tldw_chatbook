---
id: TASK-678
title: Clear the ingest pre-flight summary when the form is submitted
status: Done
assignee: []
created_date: '2026-07-26 03:26'
updated_date: '2026-07-26 04:09'
labels:
  - ingest
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a successful submit the path field empties and the form says a path is needed, while the pre-flight block still describes the file that was just submitted. The screen shows two contradictory states at once and the stale summary suggests a file is still staged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Submitting clears the pre-flight summary along with the path
- [x] #2 The gate line and the summary never describe different states at the same time
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The path and title cleared on submit but the pre-flight result did not, so the canvas asserted two things at once: 'Enter a file path to start.' beside a summary of the file just submitted, reading as though something was still staged.

Submit now clears the pre-flight result and its checking flag, and cancels any in-flight analysis worker first so a late result cannot repopulate what was just cleared. The cancel logic moved into its own helper shared with the trigger path.

Changed: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_ingest_guardrail_modal.py, Tests/UI/test_library_screen.py
<!-- SECTION:NOTES:END -->
