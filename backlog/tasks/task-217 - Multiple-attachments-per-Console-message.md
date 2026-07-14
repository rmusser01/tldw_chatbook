---
id: TASK-217
title: Multiple attachments per Console message
status: In Progress
assignee: ['@claude']
created_date: '2026-07-13 09:30'
labels:
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 (PR #621) supports a single pending attachment per session (replace-on-reattach). Extend the store, composer indicator, provider payload builder (multiple image_url parts within the max_images cap), persistence, and transcript chips to support multiple attachments per message. Requires a DB decision: messages.image_data holds one image per row today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can stage more than one attachment and see each in the composer before sending
- [ ] #2 Vision payloads carry all staged images within the model's max_images cap
- [ ] #3 Persistence/resume round-trips all attachments of a message
- [ ] #4 DB schema decision documented (per-message columns vs attachment table) with migration if changed
<!-- AC:END -->
