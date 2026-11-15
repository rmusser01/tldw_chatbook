---
id: TASK-41180
title: Coordinate resumable article listening sessions
status: To Do
created_date: 2026-11-15
assignee: []
labels:
  - watchlists
  - tts
priority: high
references:
  - https://github.com/rmusser01/tldw_chatbook/issues/2864
dependencies:
  - TASK-41179
documentation:
  - backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a cited queue and an interrupting article preserve their places and return behavior through navigation, interruption and restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Validated cited items play once in first-citation order, preserving unavailable placeholders and the old queue when replacement cannot succeed.
- [ ] #2 Single article interruption, replacement, completion, Skip, Stop and failure preserve independent bookmarks and the agreed automatic-return behavior.
- [ ] #3 Startup restores unfinished sessions paused without synthesis; completed sessions replay only explicitly, and checkpoints bound crash repetition.
- [ ] #4 Profile or runtime changes, obsolete callbacks, competing instances and surviving native work cannot overlap output or mutate newer sessions.
<!-- AC:END -->
