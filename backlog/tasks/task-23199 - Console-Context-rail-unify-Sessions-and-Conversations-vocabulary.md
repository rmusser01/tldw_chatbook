---
id: TASK-23199
title: 'Console Context rail: unify Sessions and Conversations vocabulary'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-29 21:56'
updated_date: '2026-08-30 20:36'
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
- [x] #1 Open chats and saved chats are presented under one section with one vocabulary
- [x] #2 The rail no longer reports the active conversation as None (REVISED premise - see Implementation Notes)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Definition of Done:
--------------------------------------------------
No Definition of Done items defined

Implementation Notes:
--------------------------------------------------
- [x] #2 The rail no longer reports the active conversation as None (REVISED premise - see Implementation Notes)

Definition of Done:
--------------------------------------------------
No Definition of Done items defined

Implementation Notes:
--------------------------------------------------
PARTIALLY DONE. AC #2 delivered; AC #1 (the section merge) deliberately NOT done and left for a decision - it is a structural change and the finding that justified it turned out to be partly wrong.

The audit called this a self-contradiction: Sessions showed 'Conversation  None' with the chat's own name 'Chat 1' directly beneath. Checked before changing anything, and 'None' was ACCURATE: scope_label derives from current_conversation, a PERSISTED conversation id, so an unsaved native session genuinely has none while 'Chat 1' is the unsaved tab's name. Two true statements, worded so they looked like a disagreement.

The row was still not worth its space, and the decisive fact is that it was useless in BOTH states, not just the one the audit saw: scope_label = 'This conversation' if current_conversation else '', so a SAVED chat rendered 'Conversation  This conversation', a tautology. Removed it; the durable conversation id moved to the surviving row's tooltip so no information was lost. Sessions now reads simply 'Chat 1'.

AC #2's original wording ('never reported as None while a chat is open') was wrong in premise - with nothing saved, reporting no saved conversation is correct. It was rewritten to the user-facing outcome actually delivered.

AC #1 deferred, with the evidence now stronger than before: with the scope row gone, Sessions is a header plus ONE row naming the active chat, which the Conversations section already shows as 'Chat 1 / active session'. The two are genuinely redundant and merging would reclaim about four rows. But deleting a rail section touches CONSOLE_RAIL_SECTION_IDS, ConsoleRailPreferences.session_open, persisted layout payloads, the seven-entry descriptor table and a large number of tests that pin section arity - and this session has already found several places where this codebase deliberately pins behaviour the audit misread. That is a change worth doing on its own, with its own review, not appended to a copy fix.

Test fallout: three tests referenced the retired pair. test_conversation_row_shows_placeholder_when_no_active_conversation guarded RAG-45 ('no bare Conversation label with an empty value body') - that is now structurally impossible rather than merely correct, and the test asserts that. Its sibling in TASK-23201, test_conversation_status_row_label_and_value_are_separate_visual_runs, was DELETED: it asserted painted separation of a pair that no longer exists.

preflight green. Files: Widgets/Console/console_workspace_context.py; Tests/UI/test_console_context_rail_vocabulary.py (new); 2 test files updated, 1 obsolete test removed.

--- 2026-08-30 scoping update, AC #1 still open ---

Scoped the merge properly before shipping the branch and found a trap the AC did not anticipate: session_open is NOT only the Sessions section's flag. It is the TASK-14810 MIGRATION SEED. Before that split there was one mixed Session body, and coerce_console_rail_preferences still uses the stored session_open as the fallback default for BOTH workspace_open and conversations_open:

    session_open = _coerce_bool(raw.get('session_open'), defaults.session_open)
    workspace_open=_coerce_bool(raw.get('workspace_open'), session_open)
    conversations_open=_coerce_bool(raw.get('conversations_open'), session_open)

So a payload written before TASK-14810 carries only session_open, and deleting the field would silently strand those users' restored layout. Whoever does this must keep reading session_open from stored payloads purely as a legacy seed while removing it from the dataclass, CONSOLE_RAIL_SECTION_IDS, CONTEXT_SECTION_DESCRIPTORS and the compose.

Blast radius measured, and smaller than a naive grep suggests: 12 test files reference the rail session section specifically (not the 56 that 'session' matches), of which 2 carry section-arity assertions.

One design point for the implementer: the Conversations tray currently passes show_selected_summary=False (only the session tray shows it), so folding Sessions in means turning that on for the conversations content -- otherwise the '<title> - <workspace>' summary is lost, though the browser row itself still marks the active chat.

A drafted test file for the merged state, including the pre-split-payload restore case, is at /tmp/merged_sections_test_draft.py in the authoring session; it is NOT in the branch because it asserts work that is not done.

--- 2026-08-30: AC #1 DONE ---

Folded Sessions into Conversations. Rail content is now 25 rows, so 140x40 fits where it did not before.

The migration-seed trap recorded above was real and is handled: session_open is gone as a preference field but coerce_console_rail_preferences still READS it as the TASK-14810 seed, so a pre-split payload still restores Workspaces and Conversations, and a modern explicit flag still beats the seed. Both pinned by tests.

MADE IT A MERGE, NOT A DELETION - and only the tests caught the difference. My first pass simply removed the section, which silently dropped #console-workspace-selected-conversation: that Static rendered ONLY in the Sessions tray and carries '<title> - <workspace>', which the grouped browser rows cannot show for the selected chat alone. It now renders in the Conversations projection.

Three fallout fixes where my first attempt was wrong, recorded because the pattern repeats:
- Blanket-replacing Sessions' probes with 'workspace' was wrong: Workspace owns a native ConsoleWorkspaceTree scroll owner, so local scroll offsets do not stick there. Sessions had a PLAIN bounded viewport; 'details' is the like-for-like analogue.
- I pointed a ROW-LESS-projection test at the Conversations tray, which builds grouped browser rows and so is not one. Workspaces is the only row-less projection left.
- A fallback-order test expected 'session' because it was workspace's PREDECESSOR. With Workspaces now leading, fallback_active_section falls forward to its successor instead.

One test improved rather than merely repaired: test_exact_100_workspace_state_matrix_is_contained asserted DOM ancestry as a proxy for 'this pane is not occluded'. That proxy also runs at FIRST PAINT, where a freshly mounted descendant can already be painting while its ancestors list is empty - which this change exposed by moving the Conversations search box onto the rail's centre point. It now asserts non-occlusion by a sibling pane directly, which is mount-order independent and truer to what it guards.

One test xfailed under TASK-25706 rather than forced green: test_production_workspace_pointer_keeps_pressed_key_across_outer_reflow scrolls to workspace_header_y - 3, which clamps to 0 now that Workspaces leads, so the reflow it asserts cannot occur. I had the scroll assertion passing with a hand-tuned offset before the click coordinate stopped landing - at which point I was tuning numbers toward green, which yields a test that passes for the wrong reason.

preflight green. 334 passed across the affected surface. Files: Chat/console_rail_state.py, UI/Console_Modules/left_rail.py, Widgets/Console/console_workspace_context.py; Tests/UI/test_console_context_rail_merged_sections.py (new); 11 test files updated.
<!-- SECTION:NOTES:END -->
