---
id: TASK-2855
title: Collections speaks plain language and hides its internals
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 01:32'
labels:
  - library
  - collections
  - ux-copy
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-07, observed at dev `6ffa56516`). Owner ruling 2026-08-07: keep the
row, rewrite to plain language (not hide until adapters exist).

The Collections canvas shows internal spec/roadmap language to end users: "Item reader
readiness", "Authority: local", "Content use boundary", "Blocked later: item reader, Search/RAG,
Study, Console handoff, server sync", "Next: collection item adapters are required before
item-level actions unlock", "Write Sync Safety … Sync: dry-run only". The empty state renders
"No stored collection items are available locally yet." twice on one screen, and three helper
sentences repeat the same enable-Create rule (all three persist unchanged after a valid name is
typed). No surface anywhere offers "Add to collection", so the canvas can only name empty sets.

Related P3s folded in: the triple-redundant helper text and the duplicated empty-state sentence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The canvas's status copy is one plain-language line (e.g. "Collections hold saved items for review — adding items is coming; you can create and name collections now."); spec/architecture vocabulary (adapters, authority, content use boundary, blocked-later lists) no longer appears on the canvas
- [x] #2 Sync-safety/internal detail moves behind the Details disclosure or is removed
- [x] #3 The empty state renders its message once, and the enable-Create guidance is a single sentence that disappears (or updates) once a valid name is typed
- [x] #4 Live TUI verification of empty and one-collection states
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce current state: grep the forbidden strings in library_collections_panel.py and confirm they still exist at HEAD (done -- all present, empty-state duplicate confirmed, 3 create-guidance sentences confirmed static/non-reactive).
2. Write failing tests first (TDD):
   - Tests/Widgets/test_library_collections_panel.py: widget-level tests for (a) the one plain-language status line replacing the spec block and forbidden vocabulary being absent, (b) sync detail/item-count/updated-at living inside a collapsed-by-default Details Collapsible, (c) empty-state message rendering once, (d) create-guidance sentence present with empty name / absent with a valid unique name.
   - Tests/UI/test_library_content_hub.py and Tests/UI/test_product_maturity_phase39_library_collections.py: update assertions that currently key on the old spec strings (they will fail once the copy changes) to assert the new plain copy and absence of the retired strings.
3. Implement in tldw_chatbook/Widgets/Library/library_collections_panel.py: add LIBRARY_COLLECTIONS_STATUS_LINE constant; replace the per-selection spec block with the one status line + kept "Action status/Available now" line; move item-count/sync-status/sync-detail/updated-at into a Collapsible("Details"); delete the duplicate empty-state reader Static; collapse the 3 create-guidance sentences into 1 driven by state.create_action.disabled_reason (shown only when create is disabled).
4. Wire the in-place updater: tldw_chatbook/UI/Screens/library_screen.py's _refresh_collections_panel_action_state_widgets must mount/update/remove the single guidance Static to match compose()'s conditional (recompose-discipline constraint), so typing a valid name makes it disappear without a full recompose.
5. Run targeted tests (Tests/Widgets/test_library_collections_panel.py, Tests/UI/test_library_content_hub.py -k collections, Tests/UI/test_product_maturity_phase39_library_collections.py, plus --collect-only over Tests/Library) until green.
6. Live tmux verification: empty Collections canvas and after creating one collection, using socket sddT1lib$RANDOM.
7. Update backlog ACs + Implementation Notes, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced the finding at current HEAD first: the spec-block strings ("Item reader
readiness", "Authority: local", "Content use boundary", "Blocked later: ...", "Next:
collection item adapters...", "Write Sync Safety ...") and both duplication defects
(empty-state sentence rendered twice; 3 static, non-reactive enable-Create sentences)
were all still present in library_collections_panel.py, unchanged from the finding.

Implementation (tldw_chatbook/Widgets/Library/library_collections_panel.py):
- Added LIBRARY_COLLECTIONS_STATUS_LINE module constant and use it as the single
  plain-language status line for a selected Collection, replacing the "Item reader
  readiness"/"Authority: ..."/"Content use boundary"/"Blocked later:.../"Next:
  adapters..." block. "Action status"/"Available now: create, rename, delete records"
  survives unchanged (not spec jargon, genuinely useful).
- Moved item count, sync status, sync detail, and updated-at into a
  Collapsible(title="Details", collapsed=True) inside the selected-collection detail
  pane (AC2). The "Write Sync Safety" heading and its help sentence were pure chrome
  and were removed outright; the underlying dry-run/promotion data is real information
  and now lives behind Details.
- Deleted the duplicate "Stored content preview" heading + repeated "No stored
  collection items are available locally yet." Static in the empty branch (AC3).
- Collapsed the 3 static enable-Create sentences into 1
  (#library-collection-form-guidance), sourced from state.create_action.disabled_reason
  so it reflects the *actual* reason (empty name / too long / duplicate name), and only
  renders while Create is disabled -- it disappears once a valid name is typed (AC3).

tldw_chatbook/UI/Screens/library_screen.py: added
_sync_collections_form_guidance_widget (mount/update/remove, called from
_refresh_collections_panel_action_state_widgets, which already runs on every
Input.Changed for the name/description fields) so the guidance Static's
appear/update/disappear behavior works in place, without a full panel recompose --
required by the recompose-discipline constraint since compose() now owns the same
conditional.

Tests (TDD): wrote 8 new widget-level tests in
Tests/Widgets/test_library_collections_panel.py first (confirmed RED via ImportError
for the not-yet-defined LIBRARY_COLLECTIONS_STATUS_LINE, then via NoMatches for the
retired ids), covering the plain status line + forbidden-vocabulary absence, the
Details disclosure containment (collapsed by default, sync/item-count/updated-at are
descendants), empty-state dedup, and both guidance-sentence states (shown with reason
text / hidden once valid). Updated one pre-existing widget test and two full-screen
integration tests (Tests/UI/test_library_content_hub.py,
Tests/UI/test_product_maturity_phase39_library_collections.py) whose assertions keyed
on the retired strings; the empty-state integration test now also drives a live
Input.Changed to prove the in-place guidance removal. All landed GREEN together (48
tests across the 5 targeted files); Tests/Library --collect-only sanity: 1076
collected, 0 errors.

Live tmux verification (socket sddT1lib3485, scratch profile /tmp/sddT1): launched
the real app, navigated Home -> Library -> Collections via the command palette.
Empty state showed the title, "No Collections yet.", the single next-action sentence,
the empty-state message exactly once, "No Collection selected.", and the single
"Enter a Collection name." guidance sentence above the name Input. Typed "Research
Sources" into the name Input -- the guidance sentence disappeared and Create
Collection lit up, in place (SGR capture confirmed no recompose/focus loss). Pressed
Create Collection: the new row appeared in the list, and the detail pane showed
"Selected: Research Sources", the plain status line, "Action status"/"Available now:
...", and a collapsed "> Details" disclosure -- none of the removed spec/roadmap
vocabulary anywhere on screen. Expanded Details: revealed "0 items", "Sync: dry-run
only", and the dynamic sync/promotion detail sentence (which legitimately contains its
own "Authority: ..." label sourced from Sync_Interop -- a different, data-driven
concept from the removed static "Authority: local" line, correctly left alone).
Cleaned up (Ctrl+Q, kill-server, removed the scratch profile dir).

Docs: updated Docs/User_Guide/library/collections.md (What this screen is for, Layout
tour, Sync labels intro, Quirks) to match the new copy, and refreshed its "Verified
against" stamp. The collections.svg screenshot referenced by the page still shows the
old UI -- flagged as a concern below since regenerating it is outside this task's
scope.

Files changed: tldw_chatbook/Widgets/Library/library_collections_panel.py,
tldw_chatbook/UI/Screens/library_screen.py, Docs/User_Guide/library/collections.md,
Tests/Widgets/test_library_collections_panel.py, Tests/UI/test_library_content_hub.py,
Tests/UI/test_product_maturity_phase39_library_collections.py.
<!-- SECTION:NOTES:END -->
