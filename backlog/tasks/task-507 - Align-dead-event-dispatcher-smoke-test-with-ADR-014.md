---
id: TASK-507
title: Align dead event-dispatcher smoke test with ADR-014
status: Done
assignee: []
created_date: '2026-07-24 17:17'
updated_date: '2026-07-24 17:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the dead dictionary-attach removal tests to treat the intentionally retired legacy event dispatcher as absent while continuing to import-check live event and app modules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy `event_dispatcher` import is asserted absent.
- [x] #2 Live `chat_events`, `conv_char_events`, and `app` modules still import.
- [x] #3 Dead dictionary-attach removal checks remain intact.
- [x] #4 Focused Character Chat removal tests pass.
- [x] #5 No production event wiring changes are made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the missing-module failure on branch and merge base.
2. Assert `event_dispatcher` remains absent under ADR-014 and keep live module import checks.
3. Run focused Character Chat tests, lint, format, and diff checks.
4. Link ADR-014 in notes and complete only after verification.

ADR required: no
ADR path: `backlog/decisions/014-retire-legacy-navigation-chrome.md`
Reason: This test-only correction implements ADR-014's existing decision that the retired legacy dispatcher stays deleted.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-014-aligned test-only correction. Added an explicit absence assertion for the retired tldw_chatbook.Event_Handlers.event_dispatcher module, removed its stale live-import expectation, and retained the dictionary-removal checks plus live imports for chat_events, conv_char_events, and app. No production files changed. Pre-change RED was reproduced on the branch; the merge-base RED was supplied in the task handoff. Verification: affected file 4 passed; full Tests/Character_Chat suite 423 passed; Ruff check passed; Ruff format check passed; git diff --check passed. ADR required: no. ADR path: backlog/decisions/014-retire-legacy-navigation-chrome.md. Reason: this implements ADR-014's existing decision that event_dispatcher stays deleted.
<!-- SECTION:NOTES:END -->
