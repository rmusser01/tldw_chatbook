---
id: TASK-219
title: Remove latent broken FileOpen context call in legacy attach handler
status: Done
assignee:
  - '@claude'
created_date: '2026-07-13 09:30'
updated_date: '2026-07-16 15:07'
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
- [x] #1 Legacy attach button opens a picker without TypeError (test constructs the real dialog class)
- [x] #2 Picker choice (EnhancedFileOpen vs plain) documented and consistent with repo convention
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Switched the legacy handler's picker branch to EnhancedFileOpen with the exact kwargs the Console surface uses (context=chat_images → recents/bookmarks now work here too). RED-first: test constructing the picker through the handler's real branch failed with the exact TypeError pre-fix (stub chat_window forces the branch: no _file_path_input, is_attached False, query_one raising); second test pins the kwargs against picker-signature drift. Reachability confirmed: Chat_Window_Enhanced:468/673 → handle_attach_image_button → picker branch whenever the legacy input is absent. Sweep 795/69/0 (new + comprehensive attachment + image gate + Tests/Chat). Files: UI/Chat_Modules/chat_attachment_handler.py, Tests/UI/test_legacy_attach_picker.py.
<!-- SECTION:NOTES:END -->
