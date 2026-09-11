---
id: TASK-32107
title: >-
  Library hand-off consistency: Conversations 'Open in Console' is disabled when
  ineligible while the rail's Use-in-Console stays blocked-but-pressable
  (TASK-716)
status: Done
assignee: []
created_date: '2026-09-08 22:43'
updated_date: '2026-09-11 07:45'
labels:
  - library
  - console
  - ux
  - design-decision
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32056 (PR #2523) disables the conversation reader's 'Open in Console' for an unlinked conversation and offers 'Link to workspace' beside it, which its AC required; the rail's `#library-use-in-console` deliberately stays pressable-with-reason (TASK-716). Two blocked-action grammars now coexist in Library; the critique-8 improvement list asked for one 'Send to Console' verb and placement. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A recorded decision picks one blocked-action grammar for Console hand-offs in Library
- [x] #2 The other surface is aligned to it, or the difference is documented with its reason
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests for link-on-use, Undo, and a non-linkable refusal.\n2. Reader: a link-resolvable block no longer disables Use as source; receipt + Undo ride the metadata seam.\n3. Screen: link first then proceed; Undo mirrors the link.\n4. Record the user decision under ## Decision; update the 32101 pins to the new grammar.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implements the user's recorded decision (`## Decision` in this file): LINK-ON-USE. Pressing 'Use as source' on a conversation whose only block is one a link can resolve (`not_in_active_workspace`, `cross_workspace`) links it into the active workspace and proceeds, in one step, with a '✓ linked · <workspace> · this conversation can now be used in Console' receipt and an 'Undo link' beside it. 'Link to workspace' stays for membership without a hand-off. The blocks a link cannot resolve -- and the load fence -- still disable the action, marked '○' as before.

One predicate, not two: `library_conversation_link_would_unblock(state, loaded_metadata)` answers both the reader's affordance and the screen handler's decision to link before staging, so the button's enabled state and the press's behaviour cannot disagree. `_link_selected_conversation_to_workspace` now returns the workspace display name and writes `_workspace_link_receipt` into `reader_loaded_metadata`; `_undo_selected_conversation_workspace_link` is its exact inverse (same fence, same registry, `unlink_membership`, receipt cleared). The receipt is per-load: each conversation load replaces the whole metadata mapping, pinned by test rather than by an added clear.

The task-32101 pins were UPDATED, not loosened: one control, one action name, one sentence -- the sentence now says what the press will do ('This conversation is not in this workspace. Pressing this adds it to the active workspace first, and you can undo that.') instead of naming a second button, and it still does not repeat the action name. The '○' marker follows the button's real disabled state, so a pressable action never wears the glyph the legend reserves for a blocked one.

AC#2 second half: the rail's `#library-use-in-console` keeps TASK-716's pressable-with-reason grammar, and the reason is recorded in the Decision -- it acts on a SET whose members can be blocked differently, so no single link would unblock it.

Live-verified at 235x52 on the seeded profile: press links + stages (Console opens), returning to Library shows the receipt and Undo, Undo removes the membership and the refusal sentence returns.

Files: tldw_chatbook/Widgets/Library/library_conversation_reader.py, tldw_chatbook/Widgets/Library/__init__.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_crit10_layout.py, Tests/UI/test_library_crit8_conversation_handoff.py, Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->

## Decision

Decided by the user, 2026-09-11 (critique-10 fix wave, branch `fix/library-crit10-layout`).

**Link-on-use.** Pressing "Use as source" on a conversation whose only block
is one a link can resolve (`not_in_active_workspace`, `cross_workspace`)
links it into the active workspace and proceeds, in one step, with a
"✓ linked · <workspace>" receipt and an "Undo link" beside it. The separate
"Link to workspace" button stays for membership without a hand-off.

What the gate protects, and keeps protecting: workspace membership decides
which items a Console turn is allowed to read, so a hand-off that widened it
silently would change what the model can see without anyone saying so — which
is why the link is a visible, reversible act with its own receipt rather than
an implicit side effect, and why `no_active_workspace` and the aggregate
`LIBRARY_GENERIC_WORKSPACE_BLOCK` fallback still refuse exactly as they do
today.

The rail's `#library-use-in-console` keeps TASK-716's pressable-with-reason
grammar: it acts on a SET whose members may be blocked for different reasons,
so there is no single link that would unblock it.
