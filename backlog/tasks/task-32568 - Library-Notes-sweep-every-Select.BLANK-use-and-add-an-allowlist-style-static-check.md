---
id: TASK-32568
title: >-
  Library Notes: sweep every Select.BLANK use and add an allowlist-style static
  check
status: To Do
assignee: []
created_date: '2026-09-14 22:43'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave-4's P0 (task-32533) was a misspelled blank: LibraryNoteFolderTargetDialog wrote Select.BLANK, which is not a Textual 8.x Select attribute — it resolves to Widget.BLANK (ClassVar[bool] = False), an illegal Select value — so Add to folder / Move note raised InvalidSelectValueError at mount and exited the app, and _submit dismissed with the literal folder id 'Select.NULL'. The class appears roughly 90 more times in this repo and MOST of them are deliberate: False is a real option value in those Selects. That is exactly why a reviewer cannot tell the two apart by reading, and why four in-code comments warning about the trap did not prevent it. The answer is enforcement in this repo's own idiom — the EXPECTED_CHACHANOTES_INDEXES / REVIEWED_METADATA_ONLY_DIAGNOSTICS pattern — so every use is a recorded decision rather than a silent one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every current Select.BLANK use in tldw_chatbook/ is classified as deliberate (False is a real option value) or as a blank-sentinel mistake
- [ ] #2 Each mistake found is fixed with a RED-to-GREEN pin that drives the real mount or update path
- [ ] #3 A static check enumerates the reviewed uses in an allowlist table and fails on any use not in it, in the style of EXPECTED_CHACHANOTES_INDEXES
- [ ] #4 The check runs in preflight and in the required CI job
<!-- AC:END -->
