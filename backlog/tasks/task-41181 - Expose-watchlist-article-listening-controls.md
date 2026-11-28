---
id: TASK-41181
title: Expose watchlist article listening controls
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
  - TASK-41180
documentation:
  - backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make saved-article and cited-queue playback discoverable and usable without losing the current reading or briefing context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Reader Listen, selected-citation Listen and Play cited excerpts invoke listening without marking items read or leaving the briefing.
- [ ] #2 Compact transport and global palette expose truthful preparation, timing, suspension, recovery and Pause, Resume, Stop, Skip, Clear and Play again actions.
- [ ] #3 Unsupported bodies, server scope, missing capabilities and serve-mode host output have clear explanations before relevant actions.
- [ ] #4 Mounted controls remain reachable and retain focus at 80x24 and wide sizes using existing tokens, patterns and keybinding rules.
<!-- AC:END -->
