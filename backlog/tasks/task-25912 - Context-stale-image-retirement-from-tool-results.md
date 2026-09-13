---
id: TASK-25912
title: 'Context: stale-image retirement from tool results'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:10'
updated_date: '2026-09-01 16:43'
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
- [x] #1 Image payloads in tool results older than a configurable recency threshold are replaced with a text placeholder naming what was there
- [x] #2 The most recent N turns retain their images, so an in-progress visual task is never degraded
- [x] #3 Reclaimed tokens are reflected in the context accounting
- [x] #4 The stored conversation is unchanged - retirement affects only what is sent to the provider, and reopening the conversation still shows the image
- [x] #5 Disabled by config reproduces today's behavior exactly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: 4 pure tests (placeholder naming, recency fence, token reduction, identity)\n2. retire_stale_images + StaleImageSettings/Stats in console_history_budget.py\n3. Same agent seam as 25911 (_prune_send_payload), own [agents] retire_stale_images gate (default OFF)\n4. Seam test RED-first; config sample
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
retire_stale_images in console_history_budget.py: rows in turn groups older than keep_recent_turns get their non-text content parts replaced by a text placeholder naming the mime ('[image (image/png) retired from older context to save tokens; the stored conversation still has it]'); text parts and recent turns untouched; identity object on no-op; input rows never mutated (AC#4 — send payload only, store untouched, pinned). AC#3 automatic: count_console_messages_tokens on the retired payload charges per_image_tokens for fewer parts (pinned by a before/after count test). Wired into the SAME AgentService._prune_send_payload seam as 25911, behind its own [agents] retire_stale_images gate (default OFF = byte-identical, pinned by wire-payload test), retire_images_keep_recent_turns threshold, ctor override for tests. 6 new tests (4 pure + 2 seam, seam RED-first this time). Scope note: applied on the agent send seam; the non-agent Console send path (prepare_chat_request) is untouched — its images live in prepared-request semantics owned by the gateway, a follow-up if wanted.
<!-- SECTION:NOTES:END -->
