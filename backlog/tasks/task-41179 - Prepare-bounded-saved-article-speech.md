---
id: TASK-41179
title: Prepare bounded saved article speech
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
  - TASK-41178
documentation:
  - backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Read the stored article body faithfully using a stable voice while bounding generation cost and memory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Text, HTML and Markdown prepare inert complete speech text with link labels, excerpt identification and useful empty or over-limit failures.
- [ ] #2 Long admitted bodies synthesize completely through existing TTS paths within explicit text, chunk and aggregate buffer bounds; partial output is never playable.
- [ ] #3 Saved nonsecret voice settings and authorized dependency identities survive restart or default changes; changed dependencies pause without silent substitution.
- [ ] #4 Valid retained audio requires no provider request and late or cancelled results cannot publish after source or authority retirement.
<!-- AC:END -->
