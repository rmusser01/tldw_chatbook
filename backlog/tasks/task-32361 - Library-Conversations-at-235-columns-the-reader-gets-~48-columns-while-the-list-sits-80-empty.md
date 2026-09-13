---
id: TASK-32361
title: >-
  Library Conversations at 235 columns: the reader gets ~48 columns while the
  list sits 80% empty
status: Done
assignee: []
created_date: '2026-09-11 06:18'
updated_date: '2026-09-11 07:45'
labels:
  - library
  - layout
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The list takes ~140 columns and is mostly empty; the reader wraps at ~48 (B D11 caps 42/43). The density rule from task-32217 gave an empty pane's columns to its sibling; the split with something open is still the pre-wave ratio. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a conversation open the reader takes the majority of the width at 200 columns and wider
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the conversations split at 235/100/60 with a conversation open.
2. Give the reader the majority from the measurement, not a guessed ratio.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause, traced rather than assumed. The adaptive layout asks `reader_has_item` (`selected_id is not None`), but the Conversations layout is resolved ONLY from `AdaptiveReaderShellResized`, which the shell posts on mount -- one refresh BEFORE `_ensure_library_conversation_reader_selection` settles the selection. Nothing re-resolved after that, so the empty-Reader rule (task-31979, which hands the Reader's columns to the list) stayed applied with a conversation open. Measured at 235x52 before: list 137, Reader 44; a single re-resolve produced list 50, Reader 132 -- so the ratio was never wrong, the layout was stale.

Fix: `LibraryConversationReader.sync_state` -- the one place every selection change reaches this pane -- posts `AdaptiveReaderShellResized` when the open/empty answer flips, and the screen's existing `@on` handler re-resolves. No new resolver rule and no invented ratio.

Measured after, live and in tests: 235x52 reader 132 / list 49; 100x30 reader 44 / list 45 (the rail steps aside; AC#1 is bounded at 200 columns); 60x24 reader 50, no list (single stage). Pinned both ways -- the majority with a conversation open, and task-32217's empty-pane widening still intact.

Files: tldw_chatbook/Widgets/Library/library_conversation_reader.py, Tests/UI/test_library_crit10_layout.py, Docs/User_Guide/library/media-and-conversations.md.

### Fix round 1 (review nit 7)

The AC states "200 columns and wider" but the pin ran only at 235. It is now
parametrized over (200, 52) and (235, 52) with the measured floor for each --
reader 97 / list 49 at 200, reader 132 / list 49 at 235 -- so the boundary the
AC names is the boundary that is pinned.
<!-- SECTION:NOTES:END -->
