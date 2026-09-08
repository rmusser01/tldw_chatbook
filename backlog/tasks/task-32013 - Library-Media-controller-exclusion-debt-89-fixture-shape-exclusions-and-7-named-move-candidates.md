---
id: TASK-32013
title: >-
  Library Media controller exclusion debt - 89 fixture-shape exclusions and 7
  named move candidates
status: To Do
assignee: []
created_date: '2026-09-07 22:06'
updated_date: '2026-09-07 22:07'
labels:
  - library
  - tests
  - tech-debt
  - refactor
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Media controller extraction (wave-7) left 111 of 251 media-named LibraryScreen methods behind, the largest exclusion set of the Library decomposition program and by far its highest rate at 44% of the cluster. The cause is TEST DEBT, not code entanglement: 89 of the 111 are fixture shapes -- 73 unbound-fake-self call sites and 16 instance-attribute monkeypatches -- each removable by retargeting one test, against just 4 genuine module-globals couplings and 5 screen-identity/lifecycle couplings. Nobody owns draining them, and until someone does, the Media cluster keeps two homes: the controller for 140 methods and LibraryScreen for 111 that are there only because a test reaches them through a bypass. The debt is also LIVE and growing: dev's own 306 commits merged at the wave-7 close added the first post-cleanup bypass fixture for an already-MOVED name and the first bound call to an already-PRUNED delegator, both of which the merge had to absorb by hand.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 73 unbound-fake-self exclusions are each classified as RETARGETABLE (the fixture can bind the controller method instead) or GENUINELY BLOCKED (with the reason recorded per name), and the counts are stated in this task
- [ ] #2 The 16 instance-attribute-monkeypatch exclusions are classified the same way, and the 9/7 split between those with a mover caller and those held only by recipe §3's conservative opening rule is re-derived at the time of the work rather than carried from the wave-7 docstring
- [ ] #3 Every exclusion whose classification comes out RETARGETABLE either moves to LibraryMediaController with its fixture retargeted in the same commit, or has its refusal recorded here with the evidence
- [ ] #4 The 7 named zero-mover-caller candidates (_analyze_one_library_media_item, _exit_library_media_select_mode, _focus_library_media_grip_if_current, _library_media_unanalyzed_ids, _notify_library_media_analysis_warning, _request_library_media_type, _start_library_media_analyze) each reach a verdict, and any that stay on LibraryScreen carry the live blocker that holds them
- [ ] #5 Tests/Architecture/test_library_media_wiring.py's pinned mover set, the controller module docstring's exclusion arithmetic, and both size ratchets agree with whatever the work lands
- [ ] #6 The full Library media battery (8 wiring suites, both size ratchets, the recompose ratchet, the preimport closure, the ADR-055 one-flag interlock and the canvas_sync dotted-path guard) is green or its reds are proven pre-existing on a same-Python, same-package isolated origin/dev baseline
<!-- AC:END -->

## Evidence and starting points

The census that produced these numbers is not repeated here; it is
reproducible from three places, and all three must be re-derived rather than
carried, per the recipe's "never trust a carried-over count" rule:

- `tldw_chatbook/UI/Library_Modules/library_media_controller.py`'s module
  docstring -- the full 111-name breakdown by class (73 unbound-fake-self,
  16 instance-attribute monkeypatch, 8 `inspect.getsource` source-census, 4
  module-globals coupling, 3 screen-identity in the `self in
  <widget>.ancestors` MEMBERSHIP form, 2 class monkeypatch, 2
  bypassed-construction lifecycle, 1 callback-identity, 1 shared shell
  helper, 1 generic dispatcher), each with its evidence.
- `backlog/docs/library-decomposition-recipe.md` §3 (the bypass shapes and
  the per-shape fix recipes) and §22 (the media series as landed, including
  why the 7 candidates were re-evaluated at the cleanup and DECLINED --
  "movable once the fixtures retarget" was never a prediction that a cleanup
  PR would make it so).
- `Tests/Architecture/test_library_media_wiring.py`, which pins the 140-name
  mover set, the 3 staticmethods and the 22 pruned delegators.

**Recipe §3's opening rule governs, and it is deliberately conservative:** a
name a test monkeypatches keeps its whole call graph screen-routed until that
surface's tests are retargeted. An exclusion held by that rule is correct
even when no mover would actually bypass the patch -- which is true of 7 of
the 16 monkeypatch exclusions. Draining this debt means retargeting the
fixtures FIRST and only then moving the names; it does not mean re-arguing
the exclusions against a narrower test.

## Why this is not TASK-31249

TASK-31249 is a census of Library UI tests that FAIL on clean dev and that
nobody owns; its acceptance criteria are about making named failing tests
pass. This debt is about test fixture SHAPES that block further extraction
while every one of those tests passes. The two do not share a fix, an
owner, or a done condition, so this is filed separately rather than appended
there.

## The debt is live, not static (wave-7 `origin/dev` reconciliation merge, 306 commits)

Two new instances arrived during the merge and were absorbed by hand. Both
are worth recording because they show the shape recurring, and because the
next series inherits the precedent:

1. `Tests/UI/test_library_media_reader_flow.py` gained the wave's first
   unbound-fake-self fixture for a name that had ALREADY MOVED
   (`LibraryScreen.handle_library_media_reader_more`, 2 call sites). It was
   resolved by retargeting the fixture -- a small `_MoreHandlerControllerFake`
   reproducing the delegator's one hop -- rather than by reverting the mover
   and re-shaping the pinned 140-mover set inside a merge commit. The
   alternative (revert the mover, per §3's opening rule read literally) is a
   defensible reading and is recorded here so the choice is visible.
2. `Tests/UI/test_library_media_viewer_speaker_rename.py` (new on dev) called
   `_build_library_media_viewer_display_state` on a real screen -- one of the
   22 delegators the cleanup PR pruned. Retargeted to
   `screen._media_controller.<name>(...)`, its one home; restoring the
   delegator was rejected because its only caller would be a test, which is
   exactly what the prune census exists to remove.

**A census gap the merge found, which this work must not repeat:** the
pruned-name census as run to date searched for the UNBOUND
`LibraryScreen.<name>` spelling and the bare-quoted-string spelling. Neither
sees `screen.<name>(...)` on a real instance -- a BOUND call -- which is how
instance 2 above went unnoticed until the test ran. Any census over method
names here must include the bound-receiver spelling alongside the unbound and
quoted-string ones.

## A third census gap, found by the round-2 reconciliation

`Tests/Architecture/test_library_media_wiring.py::test_media_controller_binds_
every_name_its_moved_bodies_need` (the `_MEDIA_CONTROLLER_BOUND_NAMES`
coverage check) is **one-directional**. It asserts that all 92 LISTED names
resolve to a `property` on `LibraryMediaController`, and that the list has no
duplicates. It does not assert the converse -- that every name a moved body
actually references is IN the list -- so it stays green when a new reference
arrives with no binding, which is the direction the failure travels.

Measured: this branch added NINE group-(e) bindings across two `origin/dev`
reconciliations (eight in round 1, one in round 2) and the check passed
before and after each, unchanged, because the new names were simply not in
its tuple. What actually caught the missing bindings both times was an ad-hoc
AST resolver run by hand over the controller class.

**And the shape it cannot see at all:** round 2's ported
`handle_library_media_review_selected` reads `library_media_int_backing_id`
as a BARE MODULE GLOBAL (dev promoted it out of a `LibraryScreen` method
into `Library/library_media_state.py`), so it resolves against the
controller module's own `__globals__`. That is recipe SS3's
module-globals-coupling shape; no `self.<attr>` census and no property check
can reach it, and unbound it is a `NameError` on every "Review selected"
press.

**Fix shape** (deliberately NOT done inside a merge commit, same reasoning as
the two rulings above): make the check bidirectional -- AST-walk the moved
bodies for `self.<attr>` loads, subtract the generated state shims and the
framework-service properties, and assert the remainder is exactly
`_MEDIA_CONTROLLER_BOUND_NAMES`; then separately assert that every bare
`ast.Name` load in a moved body resolves in the controller module's globals.
Both are cheap, and both are checks this program has had to run by hand at
every reconciliation.

## Renumbering provenance

**Filed as TASK-31976 on 2026-09-07 22:06; renumbered to TASK-32013 the same
day.** `origin/dev` had already minted TASK-31976 at 20:53 ("Recover Console
sends after trace boundary construction failure") in the 75-commit
`fix/media-riders` race that landed between this branch's first `origin/dev`
reconciliation and its second. Both sessions swept for a maximum before
filing; both observed the same pre-merge maximum from different worktrees,
which is precisely the failure mode `backlog/docs/lessons-backlog-hygiene.md`
records from 2026-08-21 (TASK-19573/TASK-19601) and precisely why the fix
there was a tie-break rule rather than a wider leapfrog.

Applying that rule: the OLDER `created_date` keeps the id regardless of
status, so dev's 20:53 task keeps TASK-31976 and this one moved. The new id
was taken after a fresh sweep across every remote ref and every local backlog
directory (observed maximum 32012). Inbound references updated in the same
commit: TASK-31249's cross-pointer and
`Tests/UI/test_review_set_walker.py`'s fixture comment.

Two references are deliberately NOT rewritten, because rewriting them would
falsify a record rather than fix a pointer: the round-1 merge commit
(`3c11f902a`) and follow-up commit (`2c4c4d709`) messages name TASK-31976,
and they are published history on PR #2501. This section is the bridge for
anyone who follows them here.
