---
id: TASK-31901
title: >-
  Console note follow-ups: save-as-Note owner fix + settings surface for
  console.summarize_note
status: In Progress
assignee:
  - '@robert'
created_date: '2026-09-07 06:04'
updated_date: '2026-09-07 06:06'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2467 follow-ups: (1) the pre-existing save-as-Note path saves under current_user/default_user instead of the configured notes_user_id, making such notes invisible in library views; (2) the new console.summarize_note internal prompt should be surfaced alongside console.rewind_summarize in the context-memory settings screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Save-as Note flow persists under app.notes_user_id (default_user only as unavailable fallback), asserted by a test,Context-memory settings screen exposes the console.summarize_note prompt for viewing/editing alongside console.rewind_summarize,Targeted tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix user_id in _save_console_message_as_note; assert owner in tests\n2. Add summarize_note prompt surface to settings_context_memory\n3. Run targeted tests
<!-- SECTION:PLAN:END -->
