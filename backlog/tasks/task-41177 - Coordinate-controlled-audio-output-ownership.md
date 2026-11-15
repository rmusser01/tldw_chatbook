---
id: TASK-41177
title: Coordinate controlled audio output ownership
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
  - TASK-41176
documentation:
  - backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make article listening coexist predictably with existing speech and media output, with reliable stop, position and completion control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Controlled WAV playback restores a validated retained file at its saved media position using a qualified backend.
- [ ] #2 All existing file, streaming PCM and media start paths serialize output ownership and wait for confirmed retirement before replacement output.
- [ ] #3 Late starts and old-surface cleanup cannot affect the new owner; failed termination retains a visible blocked owner.
- [ ] #4 Existing ordinary file playback and external-handoff behavior remain supported, with targeted regressions and native transfer evidence.
<!-- AC:END -->
