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
1. Failing tests for link-on-use, Undo, and a non-linkable refusal.
2. Reader: a link-resolvable block no longer disables Use as source; receipt + Undo ride the metadata seam.
3. Screen: link first then proceed; Undo mirrors the link.
4. Record the user decision under ## Decision; update the 32101 pins to the new grammar.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implements the user's recorded decision (`## Decision` in this file): LINK-ON-USE. Pressing 'Use as source' on a conversation whose only block is one a link can resolve (`not_in_active_workspace`, `cross_workspace`) links it into the active workspace and proceeds, in one step, with a '✓ linked · <workspace> · this conversation can now be used in Console' receipt and an 'Undo link' beside it. 'Link to workspace' stays for membership without a hand-off. The blocks a link cannot resolve -- and the load fence -- still disable the action, marked '○' as before.

One predicate, not two: `library_conversation_link_would_unblock(state, loaded_metadata)` answers both the reader's affordance and the screen handler's decision to link before staging, so the button's enabled state and the press's behaviour cannot disagree. `_link_selected_conversation_to_workspace` now returns the workspace display name and writes `_workspace_link_receipt` into `reader_loaded_metadata`; `_undo_selected_conversation_workspace_link` is its exact inverse (same fence, same registry, `unlink_membership`, receipt cleared). The receipt is per-load: each conversation load replaces the whole metadata mapping, pinned by test rather than by an added clear.

The task-32101 pins were UPDATED, not loosened: one control, one action name, one sentence -- the sentence now says what the press will do ('This conversation is not in this workspace. Pressing this adds it to the active workspace first, and you can undo that.') instead of naming a second button, and it still does not repeat the action name. The '○' marker follows the button's real disabled state, so a pressable action never wears the glyph the legend reserves for a blocked one.

AC#2 second half: the rail's `#library-use-in-console` keeps TASK-716's pressable-with-reason grammar, and the reason is recorded in the Decision -- it acts on a SET whose members can be blocked differently, so no single link would unblock it.

Live-verified at 235x52 on the seeded profile: press links + stages (Console opens), returning to Library shows the receipt and Undo, Undo removes the membership and the refusal sentence returns.

Files: tldw_chatbook/Widgets/Library/library_conversation_reader.py, tldw_chatbook/Widgets/Library/__init__.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_crit10_layout.py, Tests/UI/test_library_crit8_conversation_handoff.py, Docs/User_Guide/library/media-and-conversations.md.

### Fix round 1 (review P2s)

- The refusal/promise sentence is yielded directly under "Use as source"
  instead of after the Archive/Restore pair. The copy says "Pressing this",
  so its position is load-bearing and it had been pointing at "Archive
  conversation". Pinned by child order.
- Undo now unlinks the workspace the RECEIPT names, by id
  (`_workspace_link_receipt_id`), not `get_active_workspace()`. Creating a
  workspace from the rail activates it and recomposes the reader from the
  same metadata mapping, so a standing receipt can outlive its workspace;
  the old code would then have removed a membership the press never added.
  Pinned with a workspace switch between the link and the Undo -- verified
  the pin fails against the old one-line shape (it removed workspace-b's
  membership and left workspace-a's).
- The link is now gated on the same `freshness == "fresh"` answer the
  hand-off itself checks, so a press can no longer widen the workspace and
  then stage nothing (review nit 10).
- The "receipt does not survive a different conversation" pin drives a real
  second selection through the load path instead of hand-assigning the
  metadata mapping (review nit 6).
- `Docs/User_Guide/library/media-and-conversations.md`: the stale "Open in
  Console" control row (it is `Resume conversation`/`Restore and resume`, and
  it does not stage source context) is corrected while the page is open
  (review nit 9).

### Fix round 2 (re-review P3)

The other half of the stale-receipt problem the id-based Undo fix did not
cover: activating a workspace this conversation is NOT in brings the block
straight back, and the reader painted the refusal sentence with a receipt
claiming "this conversation can now be used in Console" directly under it.

`_workspace_link_receipt()` now withholds the receipt (and its Undo) while any
workspace block stands. That is the receipt's own claim restated as its
display rule, so it needs no extra state and no second copy of "which
workspace is active": the block is recomputed on every sync, and the receipt
reappears untouched if that workspace is made active again. Pinned red-first
with a workspace switch (`test_the_receipt_never_stands_beside_the_refusal_it_resolved`).

Chosen over the suggested id-match because the reader is a pure widget: an
id comparison would need the active workspace id injected at BOTH metadata
sites, one of which is outside this branch's owned ranges, to answer a
question the existing block already answers. Residual, deliberately not
covered: switching to a DIFFERENT workspace that also holds this membership
leaves a receipt naming the first one while its claim is still true; Undo
still removes the membership that receipt names, so the two stay consistent.

### Bot round (Qodo, PR #2603)

Three findings, all real, all fixed red-first.

- **High — Undo could delete an existing link.** `link_membership` is
  `INSERT OR IGNORE` and returns the EXISTING row when the membership is
  already there, so a press arriving through stale cached eligibility got a
  receipt for a link it never made and Undo would have removed someone
  else's membership. The link path now reads `get_item_memberships` first and
  records the receipt only for a membership it actually inserted; the
  hand-off still proceeds either way, and the depth-state refresh still runs
  (skipping it left the cached eligibility stale and made the hand-off refuse
  the conversation it had just accepted -- caught by the pin).
- **Medium — a stale page offered a dead action.** `Use as source` was
  pressable for a link-resolvable block even while the Conversations page was
  not fresh, where the hand-off returns without staging. Staleness is now
  reported through the ONE workspace-block seam as a block a link cannot
  resolve, so the action disables with its "○" marker and its reason, no link
  is offered, and the reader, the tooltip and the `c` accelerator cannot
  disagree about it. Resume is unaffected -- reopening the original never
  needed a fresh list.
- **Medium — repeated protocol strings.** Fixed by DELETION: the handler's
  own `freshness != "fresh"` check (added in review fix round 1) is gone,
  because a non-fresh page is now a block and the predicate is already False
  for it. The remaining `_workspace_link_receipt` metadata key is declined as
  a constant: it is the fifth key on an established controller-to-reader
  seam whose four siblings (`_workspace_block`, `_workspace_block_detail`,
  `_workspace_block_linkable`, `_list_status`) are all plain literals, and
  hoisting one of five would make the seam less consistent, not more.

Pins: `test_undo_is_withheld_when_the_membership_already_existed`,
`test_a_stale_page_refuses_instead_of_promising_a_link`.
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
