---
id: TASK-25912
title: 'Context: stale-image retirement from tool results'
status: To Do
assignee: []
created_date: '2026-08-31 15:10'
updated_date: '2026-08-31 15:11'
labels:
  - console
  - context
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Images sent into the conversation are charged against the context budget forever. Verified on origin/dev: Chat/console_history_budget.py:119,184 charges per_image_tokens through the budget and the prepared request, and a named grep for retire image across Chat/ returns zero - nothing ever strips an image payload. Hermes replaces image payloads in older tool results with text placeholders, reclaiming roughly 1600 tokens each. Independent of the other two compaction items and the smallest of the three.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Image payloads in tool results older than a configurable recency threshold are replaced with a text placeholder naming what was there
- [ ] #2 The most recent N turns retain their images, so an in-progress visual task is never degraded
- [ ] #3 Reclaimed tokens are reflected in the context accounting
- [ ] #4 The stored conversation is unchanged - retirement affects only what is sent to the provider, and reopening the conversation still shows the image
- [ ] #5 Disabled by config reproduces today's behavior exactly
<!-- AC:END -->
