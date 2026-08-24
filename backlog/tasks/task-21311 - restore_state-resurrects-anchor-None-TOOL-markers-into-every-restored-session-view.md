---
id: TASK-21311
title: >-
  restore_state resurrects anchor-None TOOL markers into every restored session
  view
status: To Do
assignee: []
created_date: '2026-08-24 00:12'
labels:
  - performance
  - console
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discovered while implementing TASK-21121. `ConsoleChatStore.restore_state` clears `_messages_by_session`, `_message_session_index` and ~25 other per-session caches, but never clears `_tool_markers_by_session`. `_with_tool_markers` leads EVERY rebuilt active-path view with the markers whose anchor is `None` (a marker appended while the session had no active leaf), so a display-only agent tool marker from the pre-restore state reappears at the head of an unrelated restored session's transcript -- including change-summary markers, which the rail's changed-files guard then reads as a live run. Anchored markers are dropped correctly; only the anchor-None ones leak. Verified at base fb0a9601e with the pre-21121 reverse scan, so this is pre-existing and not a 21121 regression.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 restore_state leaves no TOOL marker from the replaced state reachable in any restored session view, whatever the marker's anchor
- [ ] #2 A regression test covers the anchor-None marker case specifically (the anchored case already passes today)
<!-- AC:END -->
