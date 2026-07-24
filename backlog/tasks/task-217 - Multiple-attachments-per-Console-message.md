---
id: TASK-217
title: Multiple attachments per Console message
status: Done
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
- [x] #1 User can stage more than one attachment and see each in the composer before sending (append-wise, cap 5, 📎 N files indicator with clear-all)
- [x] #2 Vision payloads carry all staged images within the model's max_images cap (image-counting budget: newest-message-first reservation, chronological emission)
- [x] #3 Persistence/resume round-trips all attachments of a message (positions ≥1 with real filenames; position 0 keeps the pre-existing mime·size resume label — legacy columns carry no filename; live-QA verified)
- [x] #4 DB schema decision documented (per-message columns vs attachment table) with migration if changed (v18→v19 message_attachments, positions ≥1 only, zero duplication; migration boot proven against a real v18 fixture)
<!-- AC:END -->

## Implementation Plan

1. DB v19 migration + batch accessors. 2. List-first model with mirrored scalars + capped pending list. 3. Atomic split-addressed persistence. 4. Image-counting payload budget + multi-send. 5. Screen staging/save-all/serialization/resume. 6. Chip per attachment. 7. Verification + QA + gate. (Docs/superpowers/plans/2026-07-13-console-multi-attachment.md)

## Implementation Notes

Schema: message_attachments holds positions ≥1 only (composite PK, CHECK ≥1, FK cascade, no sync triggers — TASK-220); position 0 stays in the legacy messages columns, so single-attachment messages never touch the table and legacy readers are untouched by construction. Model is list-first (ConsoleChatMessage.attachments tuple) with the Phase-1 scalars auto-mirrored from attachments[0] through a single store helper. Persistence writes row+table atomically (nested-transaction wrap, rollback-tested); reads batch-fetch. Payload budget counts images. Live QA found a P0 on-branch (multi-send crashed: store omitted required service kwargs, masked by **kwargs fakes) — fixed with explicit scalars + optional service defaults + real-service integration tests closing that gap class. Spec honestly amended: filename-on-resume applies to positions ≥1 only. Known riding minors in the final review record. Verification: 1142-test gate, 9 live captures incl. v18-fixture migration boot and DB-shape-verified multi-send. Key files: DB/ChaChaNotes_DB.py (+migration), Chat/console_chat_models.py, Chat/console_chat_store.py, Chat/chat_persistence_service.py, Chat/console_chat_controller.py, Chat/attachment_core.py, UI/Screens/chat_screen.py, Widgets/Console/console_transcript.py.
