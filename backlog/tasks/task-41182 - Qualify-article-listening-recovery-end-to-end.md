---
id: TASK-41182
title: Qualify article listening recovery end to end
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
  - TASK-41181
documentation:
  - backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish release evidence that the complete listening workflow meets its saved-state and playback promises.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A real configured-provider and native-output workflow proves queue order, mid-article interruption, restoration and manual resume against retained audio.
- [ ] #2 Crash, source deletion, backup restore with and without audio, corruption and competing-process scenarios preserve privacy and recoverable state.
- [ ] #3 Cross-surface playback, runtime and profile changes and stale completion handling pass the supported native OS matrix with measured timing limits.
- [ ] #4 Focused regression and static checks, independent review and updated user/architecture documentation support an honest issue-resolution decision.
<!-- AC:END -->
