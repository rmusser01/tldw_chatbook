---
id: TASK-219
title: Remove latent broken FileOpen context call in legacy attach handler
status: To Do
assignee: []
created_date: '2026-07-13 09:30'
labels:
  - chat
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing dead code surfaced during Console attachments Phase 1 review (PR #621): tldw_chatbook/UI/Chat_Modules/chat_attachment_handler.py calls FileOpen(..., context="chat_images"), but the FileOpen re-exported from Third_Party.textual_fspicker accepts no context kwarg — the call raises TypeError if that path is ever exercised. Every other picker surface uses EnhancedFileOpen (which supports context + recents/bookmarks). Align the legacy handler with EnhancedFileOpen or drop the kwarg, and add a regression test that actually constructs the picker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Legacy attach button opens a picker without TypeError (test constructs the real dialog class)
- [ ] #2 Picker choice (EnhancedFileOpen vs plain) documented and consistent with repo convention
<!-- AC:END -->
