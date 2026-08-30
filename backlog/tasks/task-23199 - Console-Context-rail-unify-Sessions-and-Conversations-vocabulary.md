---
id: TASK-23199
title: 'Console Context rail: unify Sessions and Conversations vocabulary'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-29 21:56'
updated_date: '2026-08-30 06:38'
labels:
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail presents Sessions, Workspaces, Conversations and Chats simultaneously, and the user's single chat appears in two of them. The Sessions section reports 'Conversation: None' while naming that same chat on the next line and while the Inspector names it correctly. Merge Sessions into Conversations under one vocabulary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Open chats and saved chats are presented under one section with one vocabulary - NOT DONE, deferred, see Implementation Notes
- [x] #2 The rail no longer reports the active conversation as None (REVISED premise - see Implementation Notes)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIALLY DONE. AC #2 delivered; AC #1 (the section merge) deliberately NOT done and left for a decision - it is a structural change and the finding that justified it turned out to be partly wrong.

The audit called this a self-contradiction: Sessions showed 'Conversation  None' with the chat's own name 'Chat 1' directly beneath. Checked before changing anything, and 'None' was ACCURATE: scope_label derives from current_conversation, a PERSISTED conversation id, so an unsaved native session genuinely has none while 'Chat 1' is the unsaved tab's name. Two true statements, worded so they looked like a disagreement.

The row was still not worth its space, and the decisive fact is that it was useless in BOTH states, not just the one the audit saw: scope_label = 'This conversation' if current_conversation else '', so a SAVED chat rendered 'Conversation  This conversation', a tautology. Removed it; the durable conversation id moved to the surviving row's tooltip so no information was lost. Sessions now reads simply 'Chat 1'.

AC #2's original wording ('never reported as None while a chat is open') was wrong in premise - with nothing saved, reporting no saved conversation is correct. It was rewritten to the user-facing outcome actually delivered.

AC #1 deferred, with the evidence now stronger than before: with the scope row gone, Sessions is a header plus ONE row naming the active chat, which the Conversations section already shows as 'Chat 1 / active session'. The two are genuinely redundant and merging would reclaim about four rows. But deleting a rail section touches CONSOLE_RAIL_SECTION_IDS, ConsoleRailPreferences.session_open, persisted layout payloads, the seven-entry descriptor table and a large number of tests that pin section arity - and this session has already found several places where this codebase deliberately pins behaviour the audit misread. That is a change worth doing on its own, with its own review, not appended to a copy fix.

Test fallout: three tests referenced the retired pair. test_conversation_row_shows_placeholder_when_no_active_conversation guarded RAG-45 ('no bare Conversation label with an empty value body') - that is now structurally impossible rather than merely correct, and the test asserts that. Its sibling in TASK-23201, test_conversation_status_row_label_and_value_are_separate_visual_runs, was DELETED: it asserted painted separation of a pair that no longer exists.

preflight green. Files: Widgets/Console/console_workspace_context.py; Tests/UI/test_console_context_rail_vocabulary.py (new); 2 test files updated, 1 obsolete test removed.
<!-- SECTION:NOTES:END -->
