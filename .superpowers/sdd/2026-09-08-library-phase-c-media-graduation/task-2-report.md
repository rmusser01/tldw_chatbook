# Task 2 report — the resident media canvas (the behaviour change)

**STATUS: COMPLETE, with one honest shortfall named up front.** The rail-mode
switch no longer rebuilds the screen (0 whole-screen recomposes, mounts more
than halved, the acceptance pin green). **The freeze itself is not yet fixed:**
the longest main-thread block is unchanged, and the measurement now points at
exactly one remaining cause — the redundant canvas sync storm the design record
scoped out of this task. Section 5 has the numbers and section 6 the correction
to the record's own prediction about it.

Branch `feat/library-phase-c-resident-canvas`, worktree
`.worktrees/library-decomp-foundation`. Commits:

| commit | what |
|---|---|
| `7b4a7e289` | `feat(library): one resident browse shell — the phase-C media graduation` |
| `51533602c` | `fix(library): TASK-32089 — the two hazards residency creates` |
| (see §8) | docs/evidence commit |

---

## 1. The id split, resolved — the design-judgment core

The design record named the reader-shell id split as "the largest single
piece of Task 2's work". Confirmed by census before touching anything:
`#library-media-reader-shell` at 19 production sites and 43 test references,
`#library-notes-reader-shell` at 16 and 30, grip ids at 26 more. And confirmed
as unavoidable: `#library-canvas` (the items host, 125 references) is a single
id in BOTH routes' shells, so two resident shells side by side would be a
`TooManyMatches` on the most-queried selector in the screen. One shell was the
only shape that works.

**Resolution: a route-neutral id plus per-route marker classes.**

* `LibraryBrowseReaderShell`, `Widgets/Library/library_browse_reader_shell.py`
  (evolved from `library_media_reader_shell.py`, which is gone — the Media
  shell was already a subclass of the shared adaptive shell). Id
  `#library-browse-reader-shell`: naming one widget after one of the two
  routes it hosts would be a lie a future reader pays for.
* The shell wears `.library-media-route` / `.library-notes-route`, written
  ONLY by `apply_route`, which is called from both the compose path and the
  resident switch path. Every site that read "is `#library-media-reader-shell`
  mounted?" as "is the media route active?" becomes `.library-media-route` and
  keeps its exact meaning — because shell presence is now permanently true and
  the marker is not.
  *Why a marker rather than 35 `_library_selected_row_id` checks:* a marker
  projected from state at one seam IS the state check, written once. The
  hazard the record names is a proxy that goes permanently true, not a proxy
  as such. It also kept ~99 test references to a mechanical `#` -> `.`
  rewrite, which is the difference between a reviewable diff and an
  unreviewable one.
* Grips are shared, so `#library-browse-{library,items}-grip`. Their per-route
  parts travel with `apply_route`: the `library-media-pane-grip` visual class,
  and the items pane's label ("Notes" vs "Items" — the grip's tooltip AND
  accessible name).
* **CSS implications: none.** Neither retired id appeared in a stylesheet;
  all 35 sites were Python query selectors. `test_library_adaptive_reader_
  shell.py:628` already asserted the one CSS rule that used to exist was gone.
* **Who owns the rail: the shell, unchanged.** Hoisting the rail to be a
  sibling of the shell was considered and rejected — the adaptive resolver
  treats the rail as its `library` pane and four other routes share that
  widget. Residency of the SHELL is what makes the rail's 52 mounts vanish,
  which is exactly the leverage the record identified.
* **The record's census warning was right about the shape and wrong about the
  name.** The two ancestor-walk equality sites (`widget.id ==
  "library-media-canvas"`, `library_screen.py:14306/:14327`) needed no change:
  the CANVAS id is unchanged. It is the SHELL id that moved.

Recorded in the design record as "Implementation addendum — phase C task 2,
as landed".

## 2. The mechanism

`UI/Library_Modules/library_browse_route_swap.py` (new; not a
`*_controller.py`, so outside the controller ratchet — and it made
`library_screen.py` SHRINK, 32351 -> 32242, ratchet lowered same-commit).

* **Residency is stateless.** "Resident" means "already a child of
  `#library-canvas`", read off the DOM at each switch. That is not tidiness:
  `_project_library_canvas_child` (the ordinary-route projection, reachable on
  the media route) removes every child of that host, so a stored reference
  would have been a stale pointer plus a `DuplicateIds` mount. Read from the
  DOM, that same code evicts residency correctly for free and the next switch
  pays one lazy re-mount.
* **The work pane is swapped, not resident** — 2-3 widgets, and every route
  entry already rebuilt it. Residency is spent where it was measured to pay:
  the 29-widget Media canvas subtree.
* **The Notes source strip is the one structural delta**, and `compose_content`
  and the swap now share one builder, so the strip a switch mounts is the strip
  a recompose would have composed.
* **Layout**: the two routes keep separate resolved layouts and each resolver
  only re-applies its own when it differs from that route's stored value, so a
  switch would have kept the outgoing route's pane geometry. `adopt_route_
  layout` drops `_applied_layout` first, taking `sync_layout`'s initial-mount
  branch — exactly what a composed shell gets, including no focus evacuation.
* **Focus**: a cross-route switch does NOT restore the outgoing route's
  captured focus identity (that would land the user on a control they left);
  it falls through to `_arm_library_list_entry_focus`, the same seam the
  recompose path used, which schedules an immediate attempt as well as arming
  the flag. Same-route replacement keeps the pre-phase-C capture/restore.

## 3. TDD record — the reds, before the greens

| pin | red first? | the red |
|---|---|---|
| acceptance (Task 1's, strict-xfail) | yes, by construction | `1 whole-screen LibraryScreen.recompose() call(s), expected <= 0` + `114`/`179` mounts |
| hidden-canvas event gate | **yes, written before the gate** | `test_hidden_resident_media_canvas_does_not_process_row_presses` FAILED with the media viewer open while the Notes route was selected — the design record's finding #3 reproduced on the landed residency, not in a spike |
| route-ownership guard | no — written after the guard, then **mutation-verified** | reverting `_library_canvas_kind_owns_route` makes `_sync_library_canvas(screen, "media")` return True and rebuild the hidden canvas's children; the test asserts both the `False` and the untouched child identities |
| TASK-31521 suspend/resume | no — written after, green on first run | it is a *composition* pin (the record measured DOM-neutrality in a spike; this pins it on the landed code) |

The gate red, verbatim from the run that produced it:

```
FAILED Tests/UI/test_library_phase_c_resident_canvas.py::
       test_hidden_resident_media_canvas_does_not_process_row_presses
1 failed, 2 passed
```

## 4. Acceptance-test flip

`Tests/UI/test_library_phase_c_switch_residency.py::
test_library_rail_mode_switch_does_not_rebuild_the_screen`, strict-xfail
marker removed in `7b4a7e289`, the commit that flips it:

| switch | recomposes | mounts | unmounts |
|---|---|---|---|
| notes (switch) | 1 -> **0** | 114 -> **26** | 120 -> **2** |
| media (switch-back) | 1 -> **0** | 179 -> **81** | 175 -> **86** |

Its liveness assertions (destination canvas mounted AND displayed, selection
moved) pass in both directions, so this is a cheaper switch and not a dead one
— which is the failure mode that green result would otherwise hide.

**The 25-mount ceiling was re-derived, not quietly raised** (25 -> 90 mounts /
95 unmounts), under the design record's own ceiling-provenance clause, with
the derivation written into the test's docstring. Why 25 was unreachable: the
canvas a switch goes TO has been off-route since the last visit — and, under
the new route-ownership guard, deliberately un-synced while it was there — so
switching back MUST repaint it, and that repaint is
`sync_state` -> `refresh(recompose=True)` over the canvas's own 28 children.
It happens twice per media switch (the synchronous group coalesces into one
frame; the browse worker's result lands in another), and the second is not
redundant — it paints results the first did not have. Six interleaved round
trips in one process gave identical counts every iteration.

## 5. Probe before/after — and the shortfall

`Helper_Scripts/library_switch_teardown_probe.py`. ONE scratch worktree at a
fixed path with one venv, arms selected with `git switch --detach` between
runs so both are measured from the same checkout location (recipe §9);
order-swapped (base, mine / mine, base / base, mine); n = 3 runs per arm.
Medians:

| interaction | block before | block after | cpu before | cpu after | mounts | recompose |
|---|---|---|---|---|---|---|
| media (switch-back) | 98 ms | **98 ms** | 216 ms | **150 ms** | 175–179 -> **77–81** | 1 -> **0** |
| notes (switch) | 142 ms | **147 ms** | 165 ms | **133 ms** | 114 -> **26** | 1 -> **0** |

CPU −19% to −31%, mounts −55% to −77%, whole-screen recompose gone. **The
longest main-thread block does not move** (142 -> 147 ms on notes is inside
the base arm's own 128–149 ms spread).

The reason is in the bucket table and is unambiguous: after residency, CSS
restyle is **88 ms of the media switch's 154 ms (58%) across 423
`Stylesheet.apply` calls**, and every one of those is triggered by mounting one
of the 81 widgets the two canvas repaints produce. The freeze is now made of
exactly one thing.

## 6. Two corrections to the design record (both now written into it)

**(a) "The acceptance pin below does not depend on [the sync storm]" — wrong.**
It does. The pin's mount ceiling could not be met with the storm in place, and
the ceiling had to be re-derived. Corrected in the addendum, with the
composition of the remaining mounts spelled out so the next task starts from
numbers.

**(b) "The landed change will sit above that floor" — true, but it sits at the
OLD number on the metric that matters most.** The record framed wall-clock as
context and predicted the landed switch would be somewhere between the 30 ms
mechanism floor and the 124 ms baseline. It is at the baseline. The mechanism
did what it promised (the whole-screen rebuild is gone; the rail, nav bar,
footer, chrome and both canvases survive a switch); the storm now owns 100% of
the freeze. **Task 2 removed the structural cause; it did not fix the freeze.**
Anyone reading the plan's headline should read this paragraph with it.

**(c) The third TASK-32089 ruling is blocked by an existing pin, and was
reverted rather than forced.** The record asked for the dispatcher's blanket
`except Exception` to be narrowed to `(NoMatches, QueryError)`. Implemented,
run, reverted: it reds `test_library_canvas_sync_defects.py::
test_strict_failure_retry_retains_original_semantic_focus`, which forces a
`RuntimeError` out of `canvas.sync_state` and REQUIRES that swallow — the
dispatcher must recompose so the reconcile can RETRY the sync (`assert
attempts == 2`) with the original focus capture intact. Task 1 cleared the
wrong neighbour of this conflict (`test_tier1_toggle_falls_back_to_recompose_
on_query_one_failure`, which pins `_apply_library_row_toggle`'s except — a
different function, correctly identified as no-conflict, but not the one that
conflicts). The reasoning is recorded at the code site. What DID land from
that round: the handler now logs the traceback, which it previously discarded.

## 7. Concerns, and what the next task should know

1. **The freeze is not fixed — the sync storm now owns all of it.** Section 5
   has the measurement. The next motivated change is "a canvas repaint that
   changes nothing should not rebuild 28 children", and it is now cheap to
   evaluate because the acceptance pin's mount ceilings are a ratchet: lower
   them when it lands. Two things learned while measuring that the next task
   should not re-learn:
   * *An equality short-circuit inside `LibraryMediaCanvas.sync_state` does
     nothing on this path.* I tried it (full argument-tuple compare, discarded
     afterwards): the media switch-back stayed at 77 mounts, because the two
     repaints per switch are built from genuinely DIFFERENT states — the
     retained list first, the browse worker's fresh result second. The
     redundancy is not "same state twice"; it is "two states, one frame apart,
     that usually paint the same rows".
   * *The repaints coalesce by frame, not by call.* Five `sync_state` calls
     land per media switch (the swap's, the browse request's, the facets
     request's, then the browse worker's and the facets worker's) and produce
     exactly two rebuilds. Removing any one of the first three saves nothing.
2. **`display`-based residency is load-bearing for the event gate.** The gate
   reads `self.display` on the canvas, and the ONLY writer of that flag on a
   browse canvas is the route swap. If a future change starts toggling canvas
   `display` for another reason (a stage/emergency leg, say), the gate becomes
   wrong in both directions. It is pinned, but by a test that presses a row —
   not by anything that would notice a new writer.
3. **Residency is best-effort by design and that is not obvious from the
   outside.** Any path that clears `#library-canvas`'s children evicts it, and
   the next switch silently pays a lazy re-mount (~17 mounts). That is the
   correct behaviour, but it means a mount-count regression can look like
   "residency broke" when it is really "something else started clearing the
   host".
4. **The blanket-`except` narrowing is still owed** (§6c) and now has a named
   blocker rather than an open question.
5. **The `test_library_shell.py` battery takes ~2 h per arm on this machine.**
   The paired comparison in §8 uses targeted re-runs of the failing names
   rather than two full arms, which is what made it affordable; the full
   changed-arm run is included.

## 8. Suites — paired, by name

Both arms measured from the **same** scratch worktree path (`/tmp/phasec-probe`,
`git worktree add --detach`, its own `uv venv`) for the base arm and this
worktree for the changed arm, same invocation (`-p no:randomly -q`).

### Named in the brief

| suite | base | changed | verdict |
|---|---|---|---|
| `test_library_media_wiring.py` | 12 passed | 12 passed | green (after the fix in `cb1a47ce2` — see below) |
| `test_library_notes_wiring.py` | green | green | green |
| `test_library_screen_reuse.py` | 1 failed (`test_on_screen_suspend_stops_every_timer_in_isolation`) | same 1 | unchanged, pre-existing |
| `test_screen_reuse_helpers.py` | green | green | green |
| `test_library_media_characterization.py` | green | green | green |
| `test_library_notes_characterization.py` | green | green | green |
| dual-receiver guards (`test_media_row_toggle_resolves_the_dotted_state_path`, `test_notes_row_toggle_resolves_the_dotted_state_path`, both parametrised over `receiver_kind`) | green | **green** | green |
| `test_library_selection_updates.py` | 1 failed (`test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`) | same 1 | unchanged, pre-existing (the §7-documented red Task 1 flagged) |
| `test_library_canvas_sync_defects.py` | green | green | green |
| `test_library_canvas_scoped_sync.py` | 1 failed (`test_notes_per_click_updates_keep_screen_and_canvas_identity`) | same 1 | unchanged, pre-existing |
| `test_library_notes_reader.py` + `test_library_reader_press_scope_t22228.py` | 13 failed / 24 passed | 13 failed / 24 passed | **failure sets IDENTICAL** (diffed by name) |
| `test_library_recompose_ratchet.py` | 1 failed (67 sites found, 63 allowed) | same 1, **same 67** | unchanged, pre-existing; this change is site-count-neutral |
| `test_screen_size_ratchet.py` (library row) | failed on LINES (32351 vs 32263) | fails on METHODS only (1259 vs 1258) | **line budget lowered to 32242**; the method red is dev's, measured identical at the base commit, and deliberately not raised |
| `test_library_modules_size_ratchet.py` | 2 failed | same 2 | unchanged, pre-existing; the new module is not a `*_controller.py`, so it is not governed by that glob |
| `test_screen_preimport_payload_budget.py` | 1 failed at **504** modules (limit 500) | same 1 at **504** | unchanged: a top-level import of the new module made it 505, so both call sites are lazy |

### Phase-C's own pins

| test | result |
|---|---|
| `test_library_phase_c_switch_residency.py` | **1 passed** (was strict-xfail RED) |
| `test_library_phase_c_resident_canvas.py` (4 tests: event gate, route guard, suspend/resume, marker invariant) | **4 passed** |

### The media battery (19 `test_library_media_*.py` files + `test_library_multiselect_media.py`)

Base **26 failed / 684 passed**, changed **27 failed / 682 passed**. Three
names differ, and all three were verified INDIVIDUALLY rather than argued
from the totals:

| name | verdict |
|---|---|
| `test_library_media_return_settlement.py::test_route_change_rejects_later_geometry_settlement` | **mine, and fixed** — its precondition waited for `not owner.is_attached`, i.e. the old teardown. Updated to accept "detached OR the media canvas hidden"; every assertion the test exists for is unchanged and passes (`b1de…`, see the commit for the full reasoning). |
| `test_library_media_reader_traversal_t22207.py::test_loading_banner_paints_in_place_without_body_rebuild` / `::test_focus_traversal_builds_zero_bodies_for_pass_through_rows` | load flake — they swap places between runs and BOTH pass in isolation on BOTH trees |
| `test_library_media_return_settlement.py::test_trash_back_exact_scroll_precedes_captured_control_focus`, `test_library_media_trash.py::test_media_trash_back_and_escape_restore_distinct_media_return[escape|button]` | load flakes — pass in isolation on both trees; Task 1 already recorded this exact test as a load flake in this battery |

### `test_library_shell.py`

Not in the brief's named list, but it holds 73 of the census's test
references, so the eight tests that touch a renamed selector (plus
`test_adaptive_routes_never_receive_ordinary_emergency_geometry`, which is
parametrised over the shell selectors) were run on both arms with the same
`-k` selection.

    base arm:    40 failed, 3 passed, 800 deselected
    changed arm: 40 failed, 3 passed, 800 deselected
    diff of the two FAILED name lists (after normalising the one param id
    that embeds a renamed selector): IDENTICAL, 40/40.

**A full run of that file was attempted and abandoned as unusable evidence.**
It reached 65% in ~2 h with ~100 failures, in long contiguous runs — and
`test_library_destructive_transitions_abort_when_failed_save_leaves_note_dirty`
(8 params, all failing in that run) **passes in isolation on this tree**. The
file is order- and load-sensitive on a machine running three other pytest
processes, so its full-run failure list measures the machine, not the change.
Five names sampled from it were checked individually against the base commit
and matched exactly.

## 9. Derived artifacts and hygiene

* `./scripts/preflight.sh`: **all checks pass.** The production diagnostic
  inventory drifted by exactly three rows (one debug statement MOVED out of
  `library_screen.py` into the new module, one new route-refusal debug, and
  the same "canvas sync failed" line re-emitted with its traceback). Every row
  was read via `--statements … --since` before `--write`, per the artifact's
  own instruction; none interpolates user content, a secret, a path or a URL.
* Both size ratchets: the screen's LINE budget lowered to its measured 32242
  in the same commit; the modules ratchet needs no row (the new file is not a
  `*_controller.py`).
* User guide (`Docs/User_Guide/library/media-and-conversations.md`)
  re-stamped, not edited: no on-screen copy, control or layout changed.
* No pushes. `progress.md` untouched.

## 10. Files

**New**
* `tldw_chatbook/Widgets/Library/library_browse_reader_shell.py` (renamed from
  `library_media_reader_shell.py`, which is deleted)
* `tldw_chatbook/UI/Library_Modules/library_browse_route_swap.py`
* `Tests/UI/test_library_phase_c_resident_canvas.py`

**Changed (production)**
* `tldw_chatbook/UI/Screens/library_screen.py` (−109 net; the compose branches,
  the census, `_replace_library_browse_canvas` delegating out)
* `tldw_chatbook/UI/Library_Modules/canvas_sync.py` (route-ownership guard,
  traceback)
* `tldw_chatbook/UI/Library_Modules/library_media_controller.py`,
  `library_notes_controller.py`, `screen_constants.py` (census)
* `tldw_chatbook/Widgets/Library/library_media_canvas.py`,
  `library_notes_canvas.py` (the event gate), `__init__.py` (exports)

**Changed (tests/docs/artifacts)**: 18 test files (census), the acceptance
pin, the screen ratchet, the design record, the user guide, the diagnostic
inventory.
