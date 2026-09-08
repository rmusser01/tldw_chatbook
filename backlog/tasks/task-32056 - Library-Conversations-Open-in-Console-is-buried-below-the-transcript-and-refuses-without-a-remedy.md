---
id: TASK-32056
title: >-
  Library Conversations: 'Open in Console' is buried below the transcript and
  refuses without a remedy
status: Done
assignee: []
created_date: '2026-09-08 18:23'
updated_date: '2026-09-08 19:28'
labels:
  - library
  - conversations
  - console
  - ux
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The action sits at the bottom of the reader (row 49 of 52 under a 30-message thread, beside a clipped pager) and pressing it toasts 'Copy or link this conversation into workspace workspace-default before using it in Console.' with nothing on screen that performs the copy or link. Seeded (unlinked) rows hit this; imported or restored conversations will too. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 'Open in Console' lives in the reader header beside Read/Info, where Media places 'Use in Console', with a keyboard route
- [x] #2 An ineligible conversation shows a disabled action with the reason inline and an adjacent action that links or copies it into the active workspace, after which the hand-off works
- [x] #3 No refusal is delivered only as a toast
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move the reader actions toolbar into the header beside Read/Info.
2. Render the workspace refusal inline: '○ Open in Console · not in this workspace' plus a 'Link to workspace' action.
3. Link to workspace calls WorkspaceRegistryService.link_membership, invalidates depth state, re-syncs; handoff then works.
4. Add a 'c' keyboard route mirroring library_media_use_in_console.
5. Docs + live verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The reader's actions toolbar moved from the bottom of the pane into the header, directly under Read/Info (where Media puts 'Use in Console'), and 'c' is a keyboard route to it -- a second Binding on the same key, separated from library_media_use_in_console by check_action's selected-row gate, advertised in the Conversations footer only while it works.

A conversation outside the active workspace now states that on the control: '○ Open in Console · not in this workspace', disabled, with 'Link to workspace' stacked beneath it; pressing it writes the membership through WorkspaceRegistryService.link_membership, invalidates the depth-state cache and re-syncs, after which the hand-off runs. The reason is derived from the SAME row model the press gates on (library_item_context_handoff), so the header and the action cannot disagree -- that disagreement is what made the refusal a toast. LibraryWorkspaceSourceRow now carries the rule's reason_code and Workspaces/eligibility.py maps the two link-resolvable codes to their short inline label.

The block reason reaches the pure widget through the reader's computed-metadata channel (_workspace_block, alongside the existing _list_status/_list_summary), recomputed on every sync so it clears the moment the link lands.

Trade-off: the actions container stacks rather than sitting in a row. Live verification at 100x30 and 60x24 showed a row clipping BOTH the reason and the remedy -- the exact failure this task closes. Residual: at 100x30 the three-pane reader is 43 cells wide and the label's last three characters still clip ('not in this worksp'); the meaning survives and the remedy below is fully legible. Fixing that would mean either shortening the specified copy or adding a second, redundant wrapping line.

Files: tldw_chatbook/Widgets/Library/library_conversation_reader.py, tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py, tldw_chatbook/Workspaces/eligibility.py, tldw_chatbook/Workspaces/display_state.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_library_crit8_conversation_handoff.py (new), Tests/UI/test_library_conversation_reader.py, Tests/UI/test_library_multiselect_conversations.py, Docs/User_Guide/library/media-and-conversations.md
<!-- SECTION:NOTES:END -->
