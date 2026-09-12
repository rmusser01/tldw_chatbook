---
id: TASK-32056
title: >-
  Library Conversations: 'Open in Console' is buried below the transcript and
  refuses without a remedy
status: Done
assignee: []
created_date: '2026-09-08 18:23'
updated_date: '2026-09-08 19:44'
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
- [x] #3 No refusal is delivered only as a toast (exception recorded in the notes: 'no active workspace', which linking cannot resolve)
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

A conversation outside the active workspace now states that on the control: '○ Open in Console · not in this workspace', disabled, with 'Link to workspace' beneath it; pressing it writes the membership through WorkspaceRegistryService.link_membership, invalidates the depth-state cache and re-syncs, after which the hand-off runs. The reason is derived from the SAME decision the press makes, so the header and the action cannot disagree -- that disagreement is what made the refusal a toast. LibraryWorkspaceSourceRow now carries the rule's reason_code and Workspaces/eligibility.py maps the two link-resolvable codes to their short inline label.

The block reason reaches the pure widget through the reader's computed-metadata channel (_workspace_block, alongside the existing _list_status/_list_summary), recomputed on every sync so it clears the moment the link lands.

Trade-off: the actions container stacks rather than sitting in a row. Live verification at 100x30 and 60x24 showed a row clipping BOTH the reason and the remedy -- the exact failure this task closes.

Files: tldw_chatbook/Widgets/Library/library_conversation_reader.py, tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py, tldw_chatbook/Workspaces/eligibility.py, tldw_chatbook/Workspaces/display_state.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_library_crit8_conversation_handoff.py (new), Tests/UI/test_library_conversation_reader.py, Tests/UI/test_library_multiselect_conversations.py, Docs/User_Guide/library/media-and-conversations.md

FIX ROUND 1 (task review, five Important findings):

1+2. The header action and the 'c' accelerator now share ONE predicate,
_library_conversation_handoff_ready(), and the block itself is derived from
the very call the press makes -- library_item_context_handoff -- not
re-derived from source_rows. Before: check_action gated 'c' on the load
fence alone, so the key reached the press and raised the toast on a
conversation whose own button was disabled and said why; and a row absent
from the row model returned 'eligible' from the reader while the press fell
through to the aggregate gate and refused. A block whose reason code has no
short label now still blocks, under LIBRARY_GENERIC_WORKSPACE_BLOCK
('blocked for this workspace'), and withholds the link (a second metadata
key, _workspace_block_linkable, says whether linking would fix it) rather
than offering a remedy that would not work.

3. The reason moved out of the Button label into a wrapping Static
(#library-conversation-open-console-blocked, text-wrap: wrap, width 100%) in
the same actions container. A Textual Button label is single-line and
truncates: at 100x30 the reader pane is ~43 cells and it clipped to '...not
in this worksp'. The Static wraps, so the specified copy survives at every
width without being shortened.

4. AC#3 exceptions, recorded explicitly: after 1 and 2 the only
workspace refusal still delivered solely as a message is
'no_active_workspace' -- linking cannot resolve it (there is no target
workspace), so no inline remedy exists to offer and the user must select a
workspace first; this is documented on the user-guide page. The
_open_selected_conversation_handoff toast also survives as an unreachable
backstop for #library-conversation-use-source and for a state that changes
between render and press.

5. The rewritten settle path was run against its four existing pins in
Tests/UI/test_library_shell.py (import_verb_pair_agrees_across_rail_canvas_and_toast,
ingest_batch_completion_posts_summary_toast, completion_toast_survives_mid_batch_clear,
completion_toast_reports_dedup_as_already_in_library): 4 passed. The parts
order deliberately leads with failures ('N failed · M skipped', matching the
task's own copy spec) and is now pinned by
test_completion_toast_orders_failed_before_skipped.
REVIEW ROUND 2 (PR #2523, Qodo — two Correctness findings):

1. "Link to workspace" now sits behind the SAME loaded_actions_eligible
fence as the hand-off, and _link_selected_conversation_to_workspace
re-checks it before writing. The remedy persists membership for the
RETAINED loaded_id, so while a newly selected conversation was still
loading the link would have written the conversation the user had just
navigated away from.

2. The disabled hand-off no longer names a control it is not showing. The
tooltip answers the load fence first, and for a block linking cannot
resolve it repeats the eligibility rule's own recovery sentence (with no
active workspace: "Select an active workspace before using this item in
Console.") instead of "Press 'Link to workspace'". That sentence reaches
the pure widget as a third computed-metadata key, _workspace_block_detail;
_library_conversation_workspace_block returns (reason, link_resolves_it,
detail) and had been discarding the copy it already computed. AC#3's
recorded exception is therefore narrower than shipped: the no-workspace
case now carries its remedy on the control, not only in a message.
<!-- SECTION:NOTES:END -->
