---
id: TASK-31759
title: >-
  Console More-menu: summarize up to here as note + save transcript up to here
  as note
status: Done
assignee:
  - '@robert'
created_date: '2026-09-06 02:45'
updated_date: '2026-09-06 15:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add two per-message actions to the Console message More-menu: (1) summarize the active-path conversation span up to and including the selected message into a note, and (2) save the formatted transcript of that span as a note. Both write to the notes library via notes_scope_service and are independent of the /rewind compaction state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 More-menu on a completed USER/ASSISTANT message offers 'Summarize up to here as note' and 'Save transcript up to here as note'
- [x] #2 Summarize action uses a dedicated internal prompt (console.summarize_note) and a stateless provider call that does NOT write context_summary or move the rewind boundary
- [x] #3 Save action writes a role-prefixed Markdown transcript with provenance header, inclusive of the selected message, active-path only
- [x] #4 Both notes are created via notes_scope_service.save_note with keywords=['console']
- [x] #5 Oversized summarize spans are blocked with a user-visible notice (no silent truncation)
- [x] #6 Actions are blocked with a notice while a run is active; summarize requires a configured provider
- [x] #7 Unit tests cover action availability, dispatch, gates, note content, and no-compaction-side-effects
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Register console.summarize_note internal prompt
2. Add public stateless summarize_span_to_text on the console_context_compaction service
3. Controller: span slicing helpers + transcript-note builder + summarize_span_as_note
4. Action service: two overflow entries + dispatch branches in the message controller
5. UI wiring: button-id parser entries, exclusive workers, transient notices
6. Tests mirroring existing console action/summarize suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on branch feat/console-more-note-actions (worktree ../tldw-chatbook-note-actions, based on origin/dev). See Implementation Notes in the task file; targeted tests green.
<!-- SECTION:NOTES:END -->
