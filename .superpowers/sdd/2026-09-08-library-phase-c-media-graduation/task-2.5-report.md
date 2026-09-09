# Task 2.5 report — the sync storm (the felt-freeze second half)

**STATUS: complete.** The plan's Goal is met on both switch arms: the longest
main-thread block on a Library rail-mode switch falls from 97/148 ms to
55/73 ms, measured paired with non-overlapping ranges. Five commits, all on
`feat/library-phase-c-resident-canvas`, no pushes.

| commit | what |
|---|---|
| `497bc5f8e` | the attribution instrument + the largest restyle (marker-class flips) |
| `ed6979710` | the two rebuilds nobody sees (outgoing canvas, hidden pre-display) |
| `030aea02c` | the placeholder layout on a route's first entry |
| `8dd76b5cb` | design record, acceptance pin lowered, user guide re-stamped |
| *(this commit)* | this report, the two lessons, probe/test polish |

---

## 1. The attribution measurement, which came first

The task's hard rule was to open with attribution, not a fix, because the
previous task's report had to retract an inference that ranked leads by mounts
and assumed the restyle followed. So the first artifact is an instrument:

**`Helper_Scripts/library_restyle_attribution_probe.py`** — the third phase-C
probe. It wraps the five Textual entry points that reach `Stylesheet.apply`:

| trigger | what it costs |
|---|---|
| `App._register` | one apply per newly mounted widget — **mount-proportional** |
| `App.update_styles(node)` | `update_nodes(node.walk_children(with_self=True))` — one apply per DESCENDANT. Reached from every `add_class`/`remove_class`/`set_class`/`toggle_class` that changes the class set, and from pseudo-class reactives like `disabled`. **Subtree-proportional** |
| `Screen._update_focus_styles` | the `:focus-within` ancestor's subtree, per focus move |
| `App.refresh_css` | the whole tree |
| `Widget._cover` | one |

plus the multiplier that belongs to no trigger and inflates every count:
`Stylesheet._process_component_classes` runs a nested `apply` on a throwaway
node per component class, counted separately as `virt`.

Each apply is attributed to its trigger *and* to the first stack frame outside
`textual/` — that last part is what turns "restyle is expensive" into "this
line is expensive". `LIBRARY_RESTYLE_TRACE=1` additionally prints one line per
restyle FIRE (which node's subtree, how many applies, how long, when), which
is what showed the same flip happening twice per switch.

### The table (start commit `4e84a77fb` = task 2 as landed)

| switch | applies | restyle ms | `apply_route` (2 class flips) | `sync_layout` (`disabled`) | mount | other |
|---|---|---|---|---|---|---|
| media (switch-back) | 423 | 86 | **238 (56%) / 43 ms** | 0 | 89 (21%) / 17 ms | 96 |
| notes (switch), 1st | 463 | 92 | 192 (41%) / 36 ms | **205 (44%) / 41 ms** | 30 (6%) / 5 ms | 36 |
| notes (switch), later | 333 | 65 | **206 (62%) / 34 ms** | 0 | 74 (22%) / 12 ms | 53 |

**The table overturned the leads' ordering, and the task's own instruction was
to follow the table.** Leads 1 and 2 are mount-side; mounts are 6–22% of the
applies. The largest originator on all three arms was
`LibraryBrowseReaderShell.apply_route` — the marker-class seam the design
record introduced in task 2 — running TWO full-subtree restyles of the entire
Library per switch, 96–119 applies each, for two classes **no stylesheet rule
anywhere references**. Second, on the arm with the worst block time, was
`LibraryAdaptiveReaderShell.sync_layout`'s `pane.disabled = not open`, firing
four times: both panes closed at 23 ms, reopened at 76 ms.

The trace also caught a correctness smell in passing: between the two
`set_class` calls the shell wears BOTH `.library-media-route` and
`.library-notes-route`, so a query landing in that window sees a shell on two
routes at once.

**Neither measured-useless move was repeated** (an equality short-circuit in
`sync_state`; removing any of the first three coalesced sync calls), and
nothing in the table overturned those measurements — the frame-coalescing
effect they rest on was independently re-confirmed in §4 below.

---

## 2. The four fixes, in the order the table ranked them

Each is red-first with the measured number in the test docstring, and each has
its own structural pin. None relies on the blanket `except` for its silence:
every refusal is an explicit `return False` on a checked condition, and every
pin fails loudly.

### (1) `apply_route` flips its markers with `update=False` — `497bc5f8e`

`set_class(..., update=False)` is Textual's own "do not restyle" flag. The
premise it rests on — the markers are query markers, not style hooks — is
pinned, not assumed: `test_route_marker_classes_have_no_stylesheet_rules`
reads the PARSED stylesheet (widget `DEFAULT_CSS` is part of it, so grepping
`css/` would miss it) with word-boundary regexes for all three flipped class
names, and fails the moment a rule starts depending on one — naming
`apply_route` as the place to restore a single coalesced restyle.

* Red: `test_route_switch_does_not_restyle_the_whole_shell_subtree` →
  `{'media (switch-back)': 2, 'notes (switch)': 2}` shell-subtree restyles.
* After: 0, and 192–238 fewer applies per switch.

### (2) The outgoing-canvas rebuild — `ed6979710` (lead 1)

`_supersede_library_notes_navigation()` repainted the Notes canvas from
`_select_library_rail_row_after_source_admission` **two statements before the
destination row is set**, so the route-ownership guard still saw Notes as the
owner and let it through — the guard was right, the ORDERING was the bug. It
now renders only when the press stays on Notes (`LIBRARY_NOTES_RAIL_ROWS`, new
in `screen_constants.py`). Re-entry repaints for real, because
`_adopt_library_browse_canvas` syncs every resident canvas it shows.

* Red: `test_switching_away_does_not_rebuild_the_canvas_being_left` — the
  outgoing canvas's children came back with different object identities.
  Identity, not count: a canvas-scoped `sync_state` rebuilds same-looking
  children, so a count assertion would not notice.
* 21 of a media switch-back's 81 mounts.

### (3) The pre-display hidden rebuild — `ed6979710` (lead 2)

The second half of the residency guard, beside the route-ownership half it
extends: `_library_resident_canvas_awaits_display` refuses a sync for a canvas
parked hidden in `#library-canvas`. Placed at the two branches that resolve a
resident canvas rather than at the top of the dispatcher, so the state
builders, `rail.apply_selection` and the work-pane sync are skipped too — and
so it costs no extra DOM query on the high-frequency row-toggle path.

Safe because showing a resident canvas is exactly what repaints it, and the
route swap is the only writer of a browse canvas's `display`.

* Red: `test_destination_canvas_is_not_rebuilt_while_it_is_still_hidden` —
  `5 canvas repaint(s) ran while the canvas was hidden`, matching the probe's
  five `sync_state` calls exactly.
* 25 of a resident Notes switch's 62 mounts.

The resident-canvas id set is now spelled in two modules (`canvas_sync` cannot
import `library_browse_route_swap` — that module imports it), so
`test_the_two_residency_guards_agree_on_which_canvases_are_resident` pins the
copies equal rather than trusting the comment that says they must be.

### (4) The placeholder layout — `030aea02c` (from the table, not the leads)

`adopt_route_layout` applied the destination route's STORED layout, which for
a never-resolved route is the all-zero default: a route's first entry CLOSED
both panes and reopened them ~50 ms later when the real resolver landed. Four
`disabled` flips, each restyling a pane's whole subtree.

A route with no resolved layout has nothing to adopt. The `_applied_layout`
reset still happens, so the resolver that follows (the swap schedules one; the
canvas sync calls one) still takes `sync_layout`'s initial-mount branch and
its application is now the FIRST one instead of the third. The all-zero test
is this codebase's existing spelling of "never resolved" — all six
`_sync_library_*_reader_layout_from_shell` resolvers drop `previous` on
exactly that condition.

* Red: `test_first_entry_into_a_route_does_not_collapse_then_reopen_its_panes`
  → `{'rail': 2, 'canvas host': 2}` pane-subtree restyles. It also asserts the
  panes end up ENABLED, since "never apply a layout at all" would satisfy the
  counter while breaking the screen.
* 205 of the first Notes switch's 463 applies, 41 ms — and a ~50 ms flash of a
  collapsed shell, which is the one user-visible change in this task.

---

## 3. Acceptance — the felt-freeze evidence

Recipe §9 pairing, done the way the record's erratum requires: **ONE scratch
worktree at a fixed path** (`/private/tmp/phasec-storm`, `git worktree add
--detach`, its own `uv venv`), arms selected by `git switch --detach` between
runs so both sit at the same KIND of checkout location; **interleaved and
order-swapped** (`base mine mine base base mine mine base base mine mine
base`); **n = 6 per arm**. Instrument:
`Helper_Scripts/library_switch_teardown_probe.py`, unchanged by this task.
Medians, with the full block range across the six runs:

| interaction | block before | block after | cpu | restyle | applies | mounts |
|---|---|---|---|---|---|---|
| **media (switch-back)** | 97 ms (95–107) | **55 ms (51–65)** | 149 → 86 ms | 87 → 33 ms | 423 → 135 | 77 → 60 |
| **notes (switch)** | 148 ms (140–152) | **73 ms (68–78)** | 139 → 69 ms | 97 → 21 ms | 463 → 64 | 26 → 26 |
| media (switch-in) | 133 ms | 130 ms | — | — | 537 → 534 | 177 |
| media (re-click) | 65 ms | 63 ms | — | — | 129 | 60 |
| notes (re-click) | 47 ms | 49 ms | — | — | 89 | 38 |

**The block ranges on the two switch arms do not overlap.** Task 2's result
could not claim that — there, both arms sat inside one band (98 → 98,
142 → 147 with a 128–149 spread), which is exactly why this task existed.

Unchanged on purpose: `media (switch-in)` is the ordinary-route entry and
still takes the whole-screen recompose (it is not a resident-shell switch);
the two re-click rows move inside their own spread.

Against the program's founding numbers — the plan's 139–380 ms freeze, recipe
§25's 264–485 ms settle band — a Library rail-mode switch now blocks the main
thread for 55–73 ms.

### Pins tightened, with fresh provenance

`Tests/UI/test_library_phase_c_switch_residency.py`: **90/95 → 69/74**
mounts/unmounts. Derived fresh from three consecutive runs of that test, which
measured identically every time (notes 26/2, media 62/67), plus the same ~11%
headroom the 90/95 pair used; the derivation is in the docstring per the
ceiling-provenance clause. That file's own "when the storm is fixed, LOWER
these numbers" paragraph is now the lowering, and its warning that a green
result does not mean "the switch is cheap" is kept and updated with the new
apply counts.

New file `Tests/UI/test_library_phase_c_switch_storm.py`, six tests: the four
invariants above plus the stylesheet-independence guard and the
two-modules-agree guard.

---

## 4. The double destination rebuild — measured, and kept

The plan asked for a decision between paint-retained-then-patch (today) and
render-once-when-worker-lands, on perceived latency vs wasted work, with
numbers if cheap. It was cheap. Option A was implemented as a throwaway (the
adopt-sync short-circuited) and probed three times:

| media (switch-back) | mounts | block |
|---|---|---|
| paint-retained-then-patch (landed) | 62 | 55 ms (51–65, n=6) |
| render-once-when-worker-lands | 58, 62, 62 | 51, 52, 57 ms |

**Option A buys nothing measurable.** The reason is the effect task 2 already
documented for the first three coalesced calls: repaints coalesce by FRAME,
not by call, so removing one of two syncs landing in the same frame removes a
call, not a rebuild. The 58 in the first run did not reproduce.

It is also not merely neutral. With hidden repaints now refused, the
adopt-sync is the ONLY thing that shows a result which arrived while the
canvas was hidden — and on the measured timeline the notes tree worker lands
at ~15 ms while the swap displays the canvas at ~74 ms. Under option A that
result would never be painted. Kept, decided by measurement.

---

## 5. Guards — none weakened

| guard | result |
|---|---|
| both dual-receiver canvas-sync guards (`test_media_row_toggle_resolves_the_dotted_state_path`, `test_notes_row_toggle_resolves_the_dotted_state_path`, parametrised over `receiver_kind`) | 4 passed |
| route-ownership guard + its refusal-direction test (`test_off_route_media_sync_is_refused_rather_than_repainting`, `test_on_route_sync_is_not_refused_by_the_ownership_guard`) | passed — untouched; the new hidden-canvas refusal is a SECOND condition beside it, not a replacement |
| `test_strict_failure_retry_retains_original_semantic_focus` (requires the blanket swallow) | passed — the blanket `except` is untouched, and no fix in this task routes through it |
| hazard pins (`test_hidden_resident_media_canvas_does_not_process_row_presses`, suspend/resume, marker-class invariant) | 5 passed |
| acceptance pin | passed at the LOWER ceilings |

The event gate reads `self.display` on the canvas, and task 2's concern #2
warned that a new writer of that flag would silently break it in both
directions. This task added a READER of `display`
(`_library_resident_canvas_awaits_display`), not a writer, so that concern is
unchanged.

---

## 6. Suites — paired, by name

Every red below was verified BY NAME at this task's start commit
`4e84a77fb` (via `git switch --detach` in this worktree, then restored), not
argued from totals.

| suite | changed arm | verdict |
|---|---|---|
| `test_library_phase_c_switch_residency.py` | 1 passed (at 69/74) | green |
| `test_library_phase_c_resident_canvas.py` | 5 passed | green |
| `test_library_phase_c_switch_storm.py` (new) | 6 passed | green |
| `test_library_canvas_sync_defects.py` | green | green (one one-off red, `test_notes_row_press_to_editor_keeps_focus_inside_the_canvas`, did not reproduce on re-run or in isolation — load flake) |
| `test_library_canvas_scoped_sync.py` | 1 failed (`test_notes_per_click_updates_keep_screen_and_canvas_identity`) | pre-existing (task 2 recorded the same) |
| `test_library_selection_updates.py` | 1 failed (`test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`) | pre-existing, §7-documented |
| `test_library_notes_reader.py` | 13 failed / 21 passed | **failure NAME sets IDENTICAL at base and changed** (both 13, diffed name by name) |
| `test_library_notes_characterization.py`, `test_library_media_characterization.py` | green | green |
| adaptive-shell battery (`test_library_adaptive_reader_shell.py`, `_closeout`, `test_library_media_reader_shell.py`, `test_library_media_side_by_side.py`) | 119 passed / 4 failed | all four verified by name at `4e84a77fb`, failing identically there |
| `Tests/Architecture/test_library_*_wiring.py` + support-layer surface | green | green |
| `test_library_modules_size_ratchet.py` | 2 failed | pre-existing (neither file is one this task touched) |
| `test_library_recompose_ratchet.py` | 1 failed, **67 sites found, 63 allowed** | pre-existing and site-count-NEUTRAL: 67 is exactly task 2's number |
| `test_screen_preimport_payload_budget.py` | 1 failed at **504** modules (limit 500) | unchanged: 504 is exactly task 2's number, so nothing new is imported at boot |
| `test_screen_size_ratchet.py` (library row) | LINES pass at the lowered 32241; METHODS still 1259 vs 1258 | the method red is dev's, pre-existing, deliberately not raised |
| media battery (19 `test_library_media_*.py` + `test_library_multiselect_media.py`) | see below | see below |

**The media battery, run on both arms from this worktree** (base arm via
`git switch --detach 4e84a77fb`, then restored), same invocation
(`-p no:randomly -q`), one process at a time:

    base arm:     26 failed, 683 passed  (709 tests, 15m13s)
    changed arm:  24 failed

**The changed arm's failure set is a strict SUBSET of the base arm's — zero
branch-unique failures.** The two names that fail only at base are both
wall-clock probes (`test_library_media_reader_match_nav_t22209.py::
test_a_new_document_rescans_for_the_same_query`,
`test_library_media_reader_traversal_t22207.py::
test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke`), i.e.
load-sensitive rather than a claim that this change fixed them.

Also run, because they reference the route markers this task changed:
`test_library_entry_compose_once.py`, `test_library_reader_press_scope_
t22228.py`, `Tests/UI/Library_Modules/test_library_notes_sync_controller.py`
— 219 passed / 3 failed, and those three
(`test_library_graduation_announcement_survives_reconcile_and_same_route_
replace`, `test_library_notes_recompose_does_not_steal_newer_focus[reconcile|
replace]`) fail identically at `4e84a77fb`.

---

## 7. Ratchets and derived artifacts

* **Screen size ratchet: 32242 → 32241, re-pinned in the same commit.** A
  one-line give-back rather than a wave. The rail-switch handler needed one
  named constant to ask "is this press LEAVING Notes?"; putting
  `LIBRARY_NOTES_RAIL_ROWS` in `screen_constants.py` (new Library code belongs
  in `Library_Modules`, per the plan's constraint) and using it at the three
  sites that spelled the same pair inline paid for the import line and the
  explanatory comment. Net: the screen shrank.
* **Library_Modules ratchet:** no row moves. `canvas_sync.py` grew by the
  guard, and that file is not a `*_controller.py`, so the glob does not
  govern it.
* **`./scripts/preflight.sh`: all four checks pass.** The production
  diagnostic inventory drifted by exactly one row (the refusal debug line in
  `canvas_sync.py`); it was read with `--statements … --since` before
  `--write`, per the artifact's own instruction — it interpolates a widget id
  from our own code, and no user content, secret, path or URL.
* **User guide** (`Docs/User_Guide/library/media-and-conversations.md`)
  re-stamped, and this time with a real (small) user-visible change: a route's
  first entry no longer collapses and re-opens the panes.
* No pushes. `progress.md` untouched. No subagents.

---

## 8. Concerns, and what the next task should know

1. **`update=False` is a standing contract, and its guard is a test rather
   than a mechanism.** If someone adds a CSS rule for `.library-media-route`,
   `.library-notes-route` or `.library-media-pane-grip`, the styles will not
   update on a switch — but `test_route_marker_classes_have_no_stylesheet_
   rules` fails first and says what to do. It reads the parsed stylesheet, so
   it also covers `DEFAULT_CSS` added anywhere in the app. Worth knowing that
   this is a *class of fix* the codebase can reuse: any class flipped purely
   as a query marker is paying a full-subtree restyle today.
2. **The two residency refusals are now spelled in two places** (route
   ownership at the top of `_sync_library_canvas`, hidden-canvas in the two
   resident branches). They are cross-referenced in both directions, and the
   id sets are pinned equal by a test — but a third resident canvas has to
   touch both.
3. **A refused sync drops its `then=` follow-up**, and four notes-tree call
   sites pass one (`then=guard` / `then=restore_focus`, always a focus-restore
   or a deferred-authority veto). This is the route-ownership guard's own
   established shape — it refuses before the follow-up is queued too, and
   shipped that way in task 2 — and restoring focus INTO a hidden canvas would
   be wrong in any case. The empirical check is that
   `test_library_notes_reader.py`'s failure set is byte-identical at both
   arms. Still worth knowing if a future follow-up ever carries a non-focus
   side effect.
4. **The remaining per-switch cost is the destination canvas's own repaint**
   — a media switch-back's 131 applies are 66 mount + 65 class-flip, and both
   halves are that repaint (the class flips are the row widgets' own
   compose-time `set_class` calls). The next lever is the canvas
   widgets painting in place instead of recomposing their entire child list —
   a canvas-widget design change, not a storm fix. Nothing in this task's
   scope requires it, and §4 shows the two-rebuild framing is not where the
   money is.
5. **`test_library_shell.py` was not run in full** (task 2 measured ~2 h per
   arm on this machine, and it is order- and load-sensitive). The phase-C pins
   and every suite that exercises the switch path were run instead. Its
   route-switch and adaptive-layout tests are the ones a future full run
   should watch.
6. **`sync_layout` fires more often than it applies anything.** Even after
   fix (4), a switch calls it 3–5 times with identical layouts (the resolver
   is invoked from the swap, from the canvas sync, from two resize paths, and
   from `call_after_refresh`). Those calls are now cheap because nothing
   changes, but the fan-in is real and would be the place to look if the
   resolver ever gets expensive.
