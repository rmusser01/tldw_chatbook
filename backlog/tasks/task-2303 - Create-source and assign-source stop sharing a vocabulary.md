---
id: TASK-2303
title: Create-source and assign-source stop sharing a vocabulary
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: three near-synonym labels coexist for two DIFFERENT operations — the
rail's "Add source" ASSIGNS an existing source to the selected watchlist,
while the header's "Create source" and the pane's "New Source" CREATE one.
Users will click the wrong one confidently. Assignment is also only
discoverable through that ambiguous rail button: the selected source's
Inspector has no assign/move action, and the assignment modal is a bare list
with no instruction line.

UAT findings F1 (high), F18.

## Acceptance Criteria (the what)

- [x] One verb consistently means "create a new source" and a clearly
      different verb means "put an existing source into a watchlist", across
      rail, header, pane, guidance copy and Inspector.
- [x] A selected source's Inspector offers the assign/move action.
- [x] The assignment modal explains what clicking an entry does.
- [x] First-run guidance references labels that actually exist on screen.

## Implementation Plan (the how)

1. Fix the vocabulary at one point per surface. **New** = bring a source
   into existence; **Add** = put a source that already exists into a
   watchlist. No surface may use the other family's verb.
   * pane toolbar `New Source` -> `New source`
   * centre empty-state `Create source` -> `New source`
   * rail `Add source` -> `Add existing…`, tooltip naming `New source` as
     the other operation
   * Overview first-run copy and the Inspector first-run hint -> `New source`
2. Give assignment a second, discoverable entry point: an
   `Add to watchlist…` action on a selected SOURCE's Inspector (new
   `AssignSourceToWatchlistRequested` message) and an `Add existing…` action
   on a selected WATCHLIST's Inspector (reusing the tree's existing
   `AddSourceToWatchlistRequested`).
3. Add the reverse picker dialog (`WatchlistPickerDialog`: pick a WATCHLIST
   for a source) beside the existing `WatchlistSourcePickerDialog`, and give
   BOTH an instruction line stating what clicking an entry does and that
   nothing is created.
4. Screen handler for the new message: candidate watchlists = those the
   source is not already in; write via `WatchlistBundleService.add_source`;
   confirm with a toast naming both ends; reload the tree.
5. Update the two suites that assert the old literals
   (`test_destination_visual_parity_correction.py`) and add a vocabulary
   suite that fails if the two verb families ever overlap again.

## Implementation Notes

### The vocabulary decision

**NEW brings a source into existence; ADD files one that already exists into
a watchlist.** No affordance may use the other family's verb, and every
membership affordance must say, in the label itself, that the thing being
moved already exists.

That last clause is the one doing the work, and it was added because the
first version of the guard did not catch the original defect. `Add source`
starts with "Add" and contains no create word, yet it is a near-synonym of
`New source` -- two words ending in the same noun -- which is exactly why the
UAT misread it. Naming the pre-existence is what actually separates the
families, so it is asserted rather than left to the reader.

Shipped labels: create is `New source` (pane toolbar, centre empty state,
binding, guidance copy). Assignment is `Add existing` from the watchlist side
(rail, watchlist Inspector) and `Add to watchlist` from the source side
(source Inspector) -- the same verb with the object made explicit by
direction, never a second verb.

No ellipses. `Add existing…` was the first choice (the "opens a dialog"
convention), and the rail's own parity suite rejected it: its
truncation detector treats any `…` in a composited rail row as a clipped
label, and it cannot distinguish a literal one. That is the right call for
that detector -- and this screen does not use the convention anyway
(`Delete` opens a confirmation with no ellipsis), so consistency with the
screen won over consistency with the platform idiom.

### AC#2 -- assignment gets a second entry point

Assignment was reachable from exactly one place: the rail, with a watchlist
already in scope. A user looking at the source they wanted to file had to
first find its intended watchlist in the tree, then pick the source back out
of a list. Both directions now exist:

* source Inspector -> `AssignSourceToWatchlistRequested` -> a new
  `WatchlistPickerDialog` listing the watchlists the source is NOT in.
* watchlist Inspector -> the rail's own `AddSourceToWatchlistRequested`, so
  both entry points land in one screen handler and cannot drift apart. The
  id is re-derived from the level at press time, not closed over from
  `compose()` -- the same re-derive the breadcrumb branch already does,
  for the same reason.

The screen handler refuses anything that is not a LOCAL `subscription`
(membership rows key on a raw local subscription id, so a server entity with
a numeric `source_id` would file an unrelated local source), and unlike
`handle_resume_source_requested` it NOTIFIES on refusal: the press is a real
gesture on a button that is on screen, and a silent refusal is the
dead-affordance shape this task removes.

### AC#3/#4

Both pickers now carry an instruction line stating what pressing a row does
and, explicitly, that nothing is created -- the modal is the last place the
distinction can be drawn before the write. The empty states name `New source`
rather than "create a source in the Sources tab".

The guidance assertions read the copy and the mounted BUTTON in the same run,
so renaming either alone fails. That is the point: a constant renamed in one
place and not another is the drift this suite exists to catch, and a
constant can be renamed without the button the user presses changing at all.

### Verification

* New file `Tests/UI/test_watchlists_source_vocabulary.py` (10 tests), every
  label read off a mounted widget in the production screen. The assign test
  presses the Inspector's own button, answers the picker with a real click,
  and then asks the BUNDLE SERVICE whether the membership row exists.
* Mutation-verified: **7** mutations (rail label, pane create label,
  empty-state label, the membership write, `_post_add_existing_source`, the
  picker instruction line, and the guidance copy's casing), each reverted
  individually -> RED -> restored byte-exact.
* Gates: vocabulary + visual-parity + inspector + tree + dialogs-escape +
  overview-pane **211 passed**.

### Live verification (235x52, fresh profile)

```
rail        New    Rename    Delete
            Add existing    Remove
centre      No sources yet.
              New source     Import OPML
guidance    2. Open Sources above and press New source to add a feed to it…
inspector   Sources … Start with New in the rail, then New source under Sources.

watchlist Inspector (scope = Reading)     source Inspector (HN Front Page)
   Reading                                   Selected: HN Front Page
   Type: Watchlist                           Type: source
            Add existing                              Preview
             Check now                               Check now
              Delete                             Add to watchlist
                                                 Stage in Console
                                                      Delete

rail Add existing ->  "Add an existing source to Reading"
                      "Choose a source below to add it to this watchlist.
                       No new source is created."
                      toast: Added "Loose Feed" to "Reading".
Inspector Add to watchlist -> "Add HN Front Page to a watchlist"
                      "Choose a watchlist below to add this source to it.
                       No new source is created."
                      toast: Added "HN Front Page" to "Later".
```

Read back out of the live profile database after both assigns:

```
Later|HN Front Page
Reading|HN Front Page
Reading|Loose Feed
```

### Files

* `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`,
  `sources_pane.py`, `overview_pane.py` -- labels and copy.
* `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py` -- the new message
  and both assign actions.
* `tldw_chatbook/UI/Watchlists_Modules/opml_dialogs.py` -- the instruction
  lines and `WatchlistPickerDialog`.
* `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` --
  `_assign_source_to_watchlist_flow` and the empty-state label.
* `tldw_chatbook/css/features/_watchlists.tcss` (+ regenerated bundle).
* Tests: new `Tests/UI/test_watchlists_source_vocabulary.py`;
  `Tests/UI/test_destination_visual_parity_correction.py` (the renamed
  literals, plus both new action labels added to its intact-label checks).

### Review wave (whole-branch adversarial review: I1, M2, M5)

**I1 -- the watchlist-side `Add existing` this task added had no backend
gate.** On the Server backend it opened a picker full of LOCAL sources the
screen was not showing and wrote a `watchlist_sources` row with a success
toast -- one control away from the rail's copy of the same verb, greyed out
explaining that the server API carries no membership fields. The source-side
twin guarded this from the start; this one did not.

Gated at three layers, because a backend switch can land between any two:
`InspectorPane.write_disabled_reason` (a new screen-seeded reactive carrying
the SAME string the rail is handed, seeded in `_build_inspector_pane` and
pushed in `watch_runtime_backend`) renders the button disabled with that
reason; `_post_add_existing_source` refuses to post; and
`handle_add_source_to_watchlist_requested` -- the single point every poster
of that message reaches -- refuses and says why. The source-side handler
gained the identical check, so both directions are now refused by ONE
condition rather than two that can drift apart.

**M2** -- `WatchlistPickerDialog` claimed "This source already belongs to
every watchlist" on a profile with zero watchlists. It now takes
`total_watchlists` and distinguishes the two causes of an empty candidate
list.

**M5** -- the source-first candidate query was N+1 (`list_sources` per
watchlist). New `WatchlistBundleService.list_watchlists_for_source` answers
it in one query, and its unit test asserts it agrees with `list_sources`
watchlist by watchlist -- a disagreeing membership set would offer a
watchlist the source is already in.

**M1/M4** -- the duplicated parity-suite comment (which still described the
dropped ellipsis) and the stale `New Source` narration are corrected,
including the one that reached a user in an assertion message. Comments that
quote the old labels as history are deliberately left.

Verification: 11 review-wave mutations across both tasks, all RED, zero
survivors. Live on a fresh profile, Server backend, watchlist in scope:
pressing the Inspector's `Add existing` opened **0** pickers and wrote **0**
membership rows, with the rail's reason painted beside it.
