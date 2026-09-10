# Task 3 report — region ownership (media), scope-honest

**STATUS: complete, with the scope finding stated up front.** The spec's
phase-C rule is an ORIGIN rule, and applying it honestly to media produced a
**three-way** census, not the two-way one the rule's wording implies. Of the
79 media `@on` rows on `LibraryScreen`, **16 migrated** to
`LibraryMediaCanvas`, **20 are canvas-origin but blocked**, and **43 are
permanent**. Phase C's table-shrink for media is real but bounded, and the
20 blocked rows are the honest finding of this task: their phase-A
prerequisite is not met, and inventing a migration for them was declined.

Branch `feat/library-phase-c-resident-canvas`, worktree
`.worktrees/library-decomp-foundation`. Base commit `e028f13a0`.

| commit | what |
|---|---|
| `480eadbba` | `feat(library): the media canvas owns its sixteen canvas-origin rows` |
| *(this commit)* | design-record addendum, user-guide re-stamp, two lessons, this report |

---

## 1. The ownership census — the task's judgment core

Every row below was classified by reading where its control is **composed**
(and, for the three message classes, where the message is **posted**), never
from the handler's name. `#library-media-back` and `#library-media-review`
share a vocabulary and differ by origin; the id token alone would have put
them in the same bucket and been wrong about both.

Method: `ast` over `library_screen.py` for the 79 media `@on` rows, then a
compose-site search for each selector across all of `tldw_chatbook/`, then a
cross-reference against `LibraryMediaController`'s own `@on` set and the
wiring suite's `_MEDIA_CLUSTER_METHOD_NAMES`. Four ids had no literal
`id="..."` site and were resolved individually (they are built by the
canvas's own action-strip helper at `library_media_canvas.py:642/650/668/682`).

| bucket | rows | why | disposition |
|---|---|---|---|
| **canvas-origin, behaviour already on the controller** | **16** | composed by `LibraryMediaCanvas.compose` (or posted by `LibraryMediaRowScroll`, defined in that module), and the body is already a `LibraryMediaController` method | **MIGRATED** |
| canvas-origin, body screen-native | 20 | composed by the canvas, but the handler body is still on `LibraryScreen` and outside the 140-name media cluster | **DEFERRED** (blocker below) |
| not canvas-origin | 43 | posted by a sibling pane or an ancestor — the canvas is not on the message's path | **PERMANENT** |

The 43 break down as **26** posted by the Reader (`LibraryMediaViewer`: 25
controls plus its `SpeakerRenamed` message), **13** by
`LibraryMediaTrashCanvas`, **3** by `LibraryMediaContent` inside the Reader,
and **1** — `MediaShellResized` (= `AdaptiveReaderShellResized`) — by the
adaptive shell. That last one is the sharpest case: the shell is the canvas's
**ancestor**, and a bubbling message travels sender → screen, so it is
structurally impossible for a descendant to catch it. The Reader and the
Trash canvas are the media canvas's *siblings* under `#library-canvas`
(`_build_library_media_active_child` returns one or the other; the Reader is
the shell's work pane), so their messages never pass through it either.

### The 16 that moved

| message | selector | handler |
|---|---|---|
| `Input.Changed` | `#library-media-filter` | `handle_library_media_filter_changed` |
| `Input.Submitted` | `#library-media-filter` | `handle_library_media_filter_submitted` |
| `Button.Pressed` | `#library-media-filter-clear` | `handle_library_media_filter_clear` |
| `Button.Pressed` | `#library-media-sort` | `handle_library_media_sort` |
| `OptionList.OptionSelected` | `#library-media-sort-choices` | `handle_library_media_sort_choice` |
| `Button.Pressed` | `#library-media-previous` | `handle_library_media_previous` |
| `Button.Pressed` | `#library-media-next` | `handle_library_media_next` |
| `Button.Pressed` | `#library-media-retry` | `handle_library_media_retry` |
| `Button.Pressed` | `#library-media-select-all` | `handle_library_media_select_all` |
| `Button.Pressed` | `#library-media-select-clear` | `handle_library_media_select_clear` |
| `Button.Pressed` | `#library-media-open-viewer` | `handle_library_media_open_viewer` |
| `Button.Pressed` | `#library-media-export` | `handle_library_media_export` (the one `async` row) |
| `Button.Pressed` | `#library-media-review` | `handle_library_media_review_these` |
| `Button.Pressed` | `#library-media-review-selected` | `handle_library_media_review_selected` |
| `Button.Pressed` | `#library-media-review-sets` | `handle_library_media_review_sets` |
| `LibraryMediaRowGeometryChanged` | — | `_handle_library_media_row_geometry_changed` |

### The 20 that did not — the STOP-and-report case, resolved as a defer

They are canvas-origin (`.library-media-row`, `#library-media-select-toggle`,
the bulk-delete family, the analyze family, the review-dismiss receipt,
`#library-media-type-filter`/`-choices`, the empty-state pair,
`#library-media-trash-open`, `#library-media-export-selected`,
`#library-media-delete-selected`) — but their **bodies live on
`LibraryScreen`**, not on `LibraryMediaController`, and they are not among
the media series' 140 moved names at all. Moving the `@on` row alone would
make the canvas call back through the screen, which the
named-constructor-dependency canon retired; moving the bodies is a phase-A
extraction, and phase C rides *after* that, never instead of it.

**This corrects a phase-C graduation criterion.** "The subsystem's phase-A
series is fully landed including cleanup" is true of the *series* and false
of *every handler the series left behind*. Filed as a follow-up rather than
absorbed here: extracting those 20 bodies is a pure-move PR with its own
wiring-pin update (`_MEDIA_CLUSTER_METHOD_NAMES` 140 → ~160), and mixing it
into a designed behaviour change is exactly what the recipe's
one-kind-of-change-per-PR rule forbids.

### Bindings: zero moved, deliberately

The spec says bindings move with behaviour ownership. None of the 16 has an
`action_*` counterpart, and Textual resolves bindings along the **focus
chain** rather than by bubbling — so moving one of the media cluster's 5
`action_*` bindings onto the canvas would narrow where its key works from
"anywhere on the Library screen" to "only while focus is inside the canvas".
That is a behaviour change this migration explicitly does not make.

## 2. The mechanism

`LibraryMediaCanvas.__init__` takes `actions: "LibraryMediaController | None"
= None`, bound at both production construction sites:

* `library_media_controller.py:3603` (`_build_library_media_active_child` —
  also the path the resident route swap uses) → `actions=self`
* `library_screen.py:13274` (`compose_content`'s media branch) →
  `actions=self._media_controller`

The **concrete controller type** is named rather than hidden behind a bespoke
protocol, per this spec's own "visible coupling over a concealed facade"
ruling (the one that retired the `LibraryScreenHost` idea). The import is
`TYPE_CHECKING`-only, because `library_media_controller` imports the canvas at
runtime.

`None` is the harness default so the ~5 test files that build a bare canvas
keep working — which means a production site that forgot the binding would
produce a canvas whose toolbar is silently inert. Both sites are therefore
pinned **by source** (`test_both_media_canvas_construction_sites_bind_the_
actions_collaborator`), because there is no runtime symptom to assert against
short of pressing every control.

Behaviour does not move with the routing: each canvas row forwards to the
same-named controller method, unchanged. `event.stop()` still happens inside
those controller bodies (14 of the 16 call it; `handle_library_media_filter_
changed` never did, at either receiver). Verified that nothing sits between
the canvas and the screen that used to see these events: the only bare
`@on(Button.Pressed)` on the bubble path is `LibraryAdaptiveReaderShell`'s
grip handler, which is on the grip **Button itself** (`if event.button is not
self: return`), not on a container.

## 3. The hazard the migration creates — red first, and its own Textual finding

**Textual 8.2.8: a node's `@on`-decorated handlers run BEFORE that node's
`on_<message>` convention method, and `event.stop()` cannot un-run a handler
on the same node.** `MessagePump._get_dispatch_methods` yields each MRO
class's `_decorated_handlers` first and the naming-convention method second;
`_on_message` reads `_stop_propagation` only after the whole dispatch loop.
`prevent_default()` does not help either — `_no_default_action` is checked
only at the top of each `for cls in MRO` iteration, so it skips parent
classes, never the rest of the current one.

Verified by a 20-line spike before designing around it (order came back
`['decorated', 'gate']`), then **reproduced on the landed code**: with the
migration in place and the refusal removed, the red was

```
E  AssertionError: a press on the hidden resident canvas ran its migrated handler
E  assert True == False
```

— the sort chooser opening on the hidden, off-route Media canvas while the
user was on Notes.

So task 2's residency gate (`LibraryMediaCanvas.on_button_pressed`) protects
nothing that moves onto the canvas. The 12 migrated `Button.Pressed` rows
take the refusal themselves through one seam, `_media_actions_for_press`,
whose scope mirrors that gate **exactly** — presses only, so the filter
`Input` rows and the geometry message keep their existing (documented)
ungated treatment and the migration changes no behaviour. The gate stays,
because it is still the only thing refusing presses aimed at the deferred 20.

**The trap inside the trap, worth more than the fix.** The existing pin
`test_hidden_resident_media_canvas_does_not_process_row_presses` stayed GREEN
through the entire hazard, because the row it presses
(`.library-media-row`) is one of the deferred 20. A guard's pin covers only
the handlers that route through the guard; migrating a handler out from under
one silently narrows what the pin proves without changing its result. Both
this and the dispatch-order finding are now in `backlog/docs/lessons-textual.md`.

## 4. TDD record

| pin | red first? | the red |
|---|---|---|
| 16 × `test_a_migrated_row_is_caught_at_the_canvas_and_gone_from_the_screen` | **yes, by construction** | `'handle_library_media_sort' has not moved to LibraryMediaCanvas yet` (×16) |
| `test_both_media_canvas_construction_sites_bind_the_actions_collaborator` | **yes** | `library_media_controller builds LibraryMediaCanvas without binding 'actions=self,'` |
| `test_a_hidden_resident_canvas_refuses_its_own_migrated_presses` | **yes, against the naive migration** | verbatim in §3; captured by temporarily removing the guard, then restoring it |
| `test_a_permanent_row_stays_on_the_screen` (43) | no — scope-honesty pin, green before and after by design | its job is to red on a FUTURE over-migration |
| `test_a_deferred_canvas_origin_row_stays_until_its_body_moves` (20) | no — same shape, plus it names the blocker |
| `test_the_census_covers_every_media_row_on_the_screen_exactly_once` | no | partition check: a media `@on` row belonging to no bucket fails by name |
| live: `test_a_migrated_press_still_reaches_the_controller_end_to_end`, `test_a_migrated_input_row_still_reaches_the_controller` | no | the dispatch actually works after the move (Sort opens its chooser; the filter arms its debounce timer) |

84 tests in `Tests/UI/test_library_phase_c_region_ownership.py`, all green.

## 5. Test retargets — 11 sites, two files, assertions untouched

Two of the 16 names had live external callers (the AST census over
`tldw_chatbook/` + `Tests/` + `Docs/` + `scripts/` + `Helper_Scripts/` +
`backlog/` found no others; the remaining hits were prose in plans and
backlog task files).

* `Tests/UI/test_review_set_walker.py` (2 sites) — `LibraryScreen.handle_
  library_media_review_selected(fake, …)` → `fake._media_controller.handle_…(…)`.
* `Tests/UI/test_library_media_return_settlement.py` (9 sites) — 7 direct
  `screen._handle_library_media_row_geometry_changed(…)` calls → the
  controller, and **3 `screen.post_message(<geometry>)` calls → the canvas**.

That last group matters more than a path rename. Two of those three posts
were driving REAL assertions and went red the moment the screen stopped
handling the message (`assert screen._media_state.return_settlement is None`
→ the settlement never ran, so it stayed armed). The third was a NEGATIVE
assertion and would have kept passing **vacuously** — a post to a node that
no longer handles the message. It was retargeted with the others, through a
named `_media_canvas(screen)` helper whose docstring says why, so the file
cannot drift back into vacuous passes.

## 6. Suites — paired against `e028f13a0`, by name

Every red below was verified at the base commit in this same worktree
(`git switch --detach e028f13a0`, then restored), same invocation
(`-p no:randomly -q`), one process at a time.

| suite | result | verdict |
|---|---|---|
| `test_library_phase_c_region_ownership.py` (new) | 84 passed | green |
| `test_library_phase_c_resident_canvas.py` | 5 passed | green |
| `test_library_phase_c_switch_residency.py` | 1 passed | green |
| `test_library_phase_c_switch_storm.py` | 6 passed | green |
| `Tests/Architecture/test_library_media_wiring.py` | 13 passed (12 + the new consistency pin) | green |
| `test_library_notes_wiring.py`, `test_library_support_layer_surface.py` | green | green |
| dual-receiver guards (`test_{media,notes}_row_toggle_resolves_the_dotted_state_path`, parametrised over `receiver_kind`) | 4 passed | green |
| `test_library_media_return_settlement.py` | 44 passed | green (was 2 failed before the retarget — see §5) |
| `test_review_set_walker.py` | green | green |
| `test_library_media_characterization.py`, `test_library_notes_characterization.py` | green | green |
| `test_library_canvas_sync_defects.py` | 1 failed (`test_notes_row_press_to_editor_keeps_focus_inside_the_canvas`) | load flake — passes in isolation; task 2.5 recorded the same name |
| `test_library_canvas_scoped_sync.py` | 1 failed (`test_notes_per_click_updates_keep_screen_and_canvas_identity`) | pre-existing (tasks 2 and 2.5) |
| `test_library_selection_updates.py` | 1 failed (`test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`) | pre-existing, §7-documented since task 1 |
| `test_library_screen_reuse.py`, `test_screen_reuse_helpers.py` | 1 failed (`test_on_screen_suspend_stops_every_timer_in_isolation`) | pre-existing and deterministic |
| `test_library_recompose_ratchet.py` | 1 failed, **67 sites found, 63 allowed** | pre-existing and site-count-NEUTRAL: 67 is exactly tasks 2 and 2.5's number |
| `Tests/Performance/test_screen_preimport_payload_budget.py` | 1 failed at **504** modules (limit 500) | unchanged: 504 is exactly tasks 2 and 2.5's number |
| `test_screen_size_ratchet.py` | library row **passes on both axes now**; 2 chat_screen reds remain | improvement — see §7 |
| `test_library_modules_size_ratchet.py` | 2 failed (`library_media_browse_controller`, `library_conversations_controller`) | pre-existing; the media-controller row passes at its re-pin |
| media battery (19 `test_library_media_*.py` + `test_library_multiselect_media.py`) | **25 failed / 684 passed on BOTH arms** | see below |

### The media battery, and the one name that needed real work to dismiss

Base arm 25 failed / 684 passed (709 tests, 14m38s); changed arm 25 failed /
684 passed (15m34s). The failure NAME sets differ by exactly one in each
direction:

* base-only: `test_library_media_reader_match_nav_t22209.py::
  test_a_new_document_rescans_for_the_same_query` — a wall-clock probe task
  2.5 already recorded as base-only.
* changed-only: `test_library_media_reader_traversal_t22207.py::
  test_focus_traversal_builds_zero_bodies_for_pass_through_rows`.

**That second name then failed FOUR consecutive times in isolation on this
branch while passing twice in isolation at base**, which two prior reports
had already labelled a "load flake". Neither the precedent nor the streak was
treated as evidence. The control settled it in one command:

    git checkout 480eadbba~1 -- tldw_chatbook/   # revert PRODUCTION only
    pytest <the one test>                        # STILL FAILED

No production-caused regression survives its own revert. Re-running base then
gave fail/pass/pass, and eight runs per arm measured **7 passed / 1 failed on
BOTH arms** — an identical rate. It is a genuine ~1-in-8 intermittent failure
on an unchanged tree, and the prior "passes in isolation" characterisation was
too generous. Recorded in `backlog/docs/lessons-testing-evidence.md`.

## 7. Ratchets, felt freeze, derived artifacts

* **Screen ratchet lowered in the landing commit: 32241 → 32178 lines,
  1259 → 1243 methods.** This is the first time the library row's METHOD
  count has moved, and it **clears the +1 method red dev left there** — by
  measurement, not by raising: 1243 is what the class now has. (Without
  re-pinning, `test_budget_is_not_left_slack_after_a_wave` would have gone
  red at 15 methods of slack against a 10-method tolerance.)
* **`library_media_controller.py` 4669 → 4670**, with a dated comment. One
  line, and it is the `actions=self` wiring the whole migration rests on;
  the file's ratchet contract sanctions a raise for a landed move with a
  reason attached, and the screen shrank 63 lines in the same commit.
* **Felt freeze: no regression.** `Helper_Scripts/library_switch_teardown_
  probe.py`, 4 runs on this branch. `media (switch-back)` block **50/51/52/55
  ms** (task 2.5's band: 55 ms, range 51–65) with cpu 78–86 ms, 58–60 mounts,
  **0 whole-screen recomposes**. `notes (switch)` block **67/76/78/82 ms**
  (task 2.5: 73 ms, 68–78) at 26 mounts. Both arms of the switch sit in the
  improved band; the migration is routing-only and moves no work.
* **`./scripts/preflight.sh`: all checks pass**, with **zero drift** — no new
  diagnostic statement, no new CSS, no schema change. (Prior phase-C tasks
  each drifted the diagnostic inventory; this one adds no logging at all.)
* **User guide** (`Docs/User_Guide/library/media-and-conversations.md`)
  re-stamped, not edited: no on-screen copy, control or layout changed. The
  stamp names the one thing a user could notice — a control on a Media list
  you have switched away from can no longer act on your behalf.
* No pushes. `progress.md` untouched. No subagents.

## 8. Concerns, and what task 4 should know

1. **The 20 deferred rows are the real remaining phase-C scope for media,
   and they are gated on a phase-A extraction, not on phase C.** Their bodies
   include the heaviest media interactions (`.library-media-row`, the whole
   bulk-delete and analyze flows). Until they move, the canvas owns 16 of the
   36 rows it structurally could — 44% of its own subtree's routing. The test
   file enumerates all 20 with the blocker, so promoting one is a one-line
   edit once its body lands on the controller.
2. **`actions=None` is a silent-failure shape, mitigated by a source pin
   rather than by a mechanism.** A third production construction site added
   later without `actions=` yields a canvas whose 16 controls do nothing, with
   no exception and no log line. The pin asserts exactly one construction site
   per module and would fail on a second — but it is a string check on source,
   which is the weakest kind of guard in this repo. A runtime alternative
   (raise when a migrated handler fires with `actions is None`) was not taken
   because it would break the bare-canvas tests that legitimately press
   nothing.
3. **The residency refusal is now spelled in THREE places** — the route-
   ownership guard and the hidden-canvas guard in `canvas_sync.py` (task 2.5
   already flagged their two-module split), plus `_media_actions_for_press`
   on the canvas. All three read the same "is this canvas showing" question.
   The canvas's copy and `on_button_pressed`'s copy are two lines apart and
   cross-referenced; the `canvas_sync` pair is pinned equal by a test. There
   is no single enumerator.
4. **`test_focus_traversal_builds_zero_bodies_for_pass_through_rows` fails
   about 1 run in 8 on an unchanged tree** (§6). It is not a load flake — it
   fails alone. Worth its own investigation task; it will keep costing
   attribution time in every future paired sweep of this battery.
5. **Two of the geometry-message test call sites were saved from becoming
   vacuous passes by this migration** (§5), which is a reminder that
   `screen.post_message(X)` as a stand-in for "X arrived" is only valid while
   the screen is X's receiver. Any future region-ownership migration should
   grep for `post_message` on the screen with the message class it is moving.

## 9. Files

**Changed (production)**
* `tldw_chatbook/Widgets/Library/library_media_canvas.py` (+the `actions`
  dependency, the guard seam, the 16 rows, the gate's corrected docstring)
* `tldw_chatbook/UI/Screens/library_screen.py` (−63: the 16 delegators out,
  `actions=` in)
* `tldw_chatbook/UI/Library_Modules/library_media_controller.py` (+1)

**New (tests)**
* `Tests/UI/test_library_phase_c_region_ownership.py`

**Changed (tests/docs/artifacts)**
* `Tests/Architecture/test_library_media_wiring.py` (the new absence set +
  its consistency pin + the counts the phase changes)
* `Tests/Architecture/test_screen_size_ratchet.py`,
  `Tests/Architecture/test_library_modules_size_ratchet.py`
* `Tests/UI/test_library_media_return_settlement.py`,
  `Tests/UI/test_review_set_walker.py` (retargets, §5)
* `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md`
  (implementation addendum: the census, the mechanism, Textual finding #7)
* `Docs/User_Guide/library/media-and-conversations.md` (re-stamp)
* `backlog/docs/lessons-textual.md`, `backlog/docs/lessons-testing-evidence.md`
