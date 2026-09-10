---
id: TASK-32228
title: Library Conversations footer offers neither 'esc focus rail' nor a '/' hint
status: In Progress
assignee: []
created_date: '2026-09-10 14:56'
updated_date: '2026-09-10 19:24'
labels:
  - library
  - conversations
  - footer
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Conversations canvas footer is nearly empty while every sibling list advertises Escape and `/`. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 27.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Conversations footer advertises the same list keys as its siblings, and they work
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the Conversations footer against its siblings.
2. Give the branch the list keys it honours without weakening the pinned two-step 'focus Items'/'focus Library' Escape grammar.
3. Docs + live-verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Half delivered, half declined on evidence -- AC#1 left UNTICKED.

Delivered. Live at 100x30 the Conversations canvas showed its Filter box, "/"
focused it, and the footer said only "F6 next pane" while every sibling list
advertised two or three keys. The "/" chip is gated on
`reader_layout.items_open`, and the footer is registered from
`compose_content`, before the shell has resolved its panes: the flag was still
False when the chip set was decided and nothing revisited it. Fixed at the seam
task-32225 added -- a pane-visibility change now re-registers the footer
unconditionally (the message itself is the gate: the shell posts it only when
an APPLIED visibility flips). One re-registration, two chips fixed. Live after:
"/ focus filter | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit"
(scratchpad crit9/wave/shell/caps/32228-conversations-100-*.txt).

Declined, and why. The remaining difference from the siblings is the Escape
chip. Siblings advertise ("esc", "focus rail") unconditionally, including in
states where the hop moves nothing; Conversations advertises Escape under the
label it will actually perform ("focus Items", then "focus Library") and omits
it exactly while the hop would move nothing. That grammar is a deliberate prior
decision shared with the Media Reader and is pinned by
Tests/UI/test_library_conversation_reader.py::
test_conversations_escape_moves_to_nearest_visible_prior_role (asserting
("esc", "focus Library") is present with focus in the Reader and absent after
the hop lands). Making this footer literally match its siblings would mean
restoring the dead-key lie task-31272 removed, and would contradict this wave's
own copy rule against advertising a key that does nothing.

Every key the Conversations footer advertises works, and every key that works
is advertised -- verified against the on_key handler, which gates "/" on the
same `items_open` flag the chip does. What is left is a product question the
implementer should not settle alone: EITHER make the siblings honest (drop
"esc focus rail" where the hop would not move focus, i.e. adopt the
Conversations grammar screen-wide) OR give Conversations entry focus on arrival
the way task-2856 gives it to the other list canvases, which would make its
Escape chip appear immediately and the footers converge without a lie. The
second is the smaller change and is recommended; it touches focus contracts
this branch does not own.

Files: tldw_chatbook/UI/Screens/library_screen.py (the pane-visibility footer
re-registration, shared with task-32225), Tests/UI/test_library_crit9_shell.py,
Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
