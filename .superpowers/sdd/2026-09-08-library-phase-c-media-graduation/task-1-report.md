# Task 1 report — mechanism spike + measured design decision

**Status: COMPLETE (not blocked).**
**Decision: mechanism C — mount-once-then-toggle lazy canvas residency — gated on
replacing the whole-screen `Widget.recompose()` the rail switch runs today.**

Branch `feat/library-phase-c-resident-canvas`, worktree
`.worktrees/library-decomp-foundation`. All measurements taken from a **scratch
worktree** (`git worktree add --detach … b81cb98b0`, own `uv venv`, `diff -r`
against the working tree clean apart from `egg-info` build artifacts), per
recipe §9's same-checkout-location rule. Textual 8.2.8, Python 3.14.2.

---

## 1. Deliverable 1 — quantifying the teardown

New instrument: **`Helper_Scripts/library_switch_teardown_probe.py`** (landed),
a sibling of `library_click_probe.py`. It reports per click: mounts and
unmounts by widget class AND by screen region, an additive synchronous CPU
bucket table, which switch path ran, and (under `LIBRARY_TEARDOWN_TRACE=1`) a
causal timeline.

### The two instrument corrections it forced

1. **The click probe's `recompose` column is blind on this path.** It counts
   `BaseAppScreen.refresh(recompose=True)`. `_select_library_rail_row_after_
   source_admission` instead **awaits `Widget.recompose()` directly** (the
   `if not replaced:` arm, `library_screen.py:19766-19767`). Recipe §25 promoted
   "recompose 0" to a load-independent verdict column and made "recompose still
   0" phase C's success condition — while every measured rail switch was doing
   exactly one whole-screen recompose.
2. **The click probe's `removes` column is dead code.** Textual 8.2.8 prunes
   through `Widget._message_loop_exit` (which performs the
   `App._registry.discard`), not `App._unregister`. Every "N mounts / 0 removes"
   line the probe has ever printed was an artifact. Hooked correctly, a media
   switch unmounts 162–176 widgets.

### Freeze-band composition (headless; terminal-write bytes excluded)

`block` = longest single main-thread block (the click probe's `max_gap`).
`cpu` = sum of the additive synchronous buckets = measured main-thread work.

| interaction | block | cpu | mounts | unmounts | targeted swap | whole-screen recompose |
|---|---|---|---|---|---|---|
| media (switch-in) | 132 ms | 285 ms | 177 | 162 | tried → **False** (0.6 ms) | **1** |
| media (re-click same) | 59 ms | 117 ms | 85–89 | 85–89 | **True** | 0 |
| notes (switch) | 154 ms | 180 ms | 114 | 121 | tried → **False** (0.2 ms) | **1** |
| notes (re-click same) | 53 ms | 66 ms | 38 | 38 | not called | 0 |
| media (switch-back) | 90 ms | 205 ms | 175–179 | 172–176 | tried → **False** | **1** |

Media switch-in, where the work goes (additive self time):
CSS restyle **111.0 ms / 39%** (534 `Stylesheet.apply` calls) · render/paint prep
76.4 ms / 27% · widget construction 47.0 ms / 17% (182 `textual.compose` calls) ·
DOM registration 38.2 ms / 13% · reflow 11.5 ms / 4%.

Media switch-in, where the mounts go (full breakdown, sums exactly):
canvas (media) **87** · rail **52** · nav bar 19 (the bar + 16 destination
buttons + 2 overflow hints) · footer 6 · screen chrome 6 · reader shell and its
2 pane grips 3 · media viewer 2 · shell grid 1 · canvas host 1 = **177**.
Notes (switch) the same way: rail 52 + canvas (notes) 19 + nav bar 19 + screen
chrome 11 + footer 6 + shell grid 6 + canvas host 1 = **114**.

Media switch-in, the same 177 by WIDGET CLASS (18 classes; the brief asked for
this cut too, and it is the one that shows the storm is generic chrome, not
media-specific widgets):

```
mounts   Button 45, Static 39, Horizontal 26, Vertical 16, NavigationButton 15,
         LibraryRailRowButton 15, DestinationRailSectionHeader 5, Input 3,
         LibraryMediaRowScroll 3, LibraryAdaptiveReaderPaneGrip 2,
         MainNavigationBar 1, Container 1, AppFooterStatus 1,
         LibraryMediaReaderShell 1, LibraryRail 1, LibraryMediaViewer 1,
         LibraryRailSearchInput 1, LibraryMediaCanvas 1
unmounts Button 40, Static 38, Horizontal 20, Vertical 15,
         LibraryRailRowButton 15, NavigationButton 15,
         DestinationRailSectionHeader 5, ScrollBar 2, LibraryMediaRowScroll 2,
         Input 2, Container 1, AppFooterStatus 1, LibraryNavigationRailHandle 1,
         LibraryRail 1, LibraryEmergencyReturn 1, LibraryLandingCanvas 1,
         MainNavigationBar 1, LibraryRailSearchInput 1
```

Notes (switch): mounts Static 30, Button 21, NavigationButton 15,
LibraryRailRowButton 15, Horizontal 10, Vertical 7,
DestinationRailSectionHeader 5, + 10 singletons = 114; unmounts 121 across 19
classes, and its unmount list is the media route being destroyed
(`LibraryMediaReaderShell`, `LibraryMediaViewer`, `LibraryMediaCanvas`,
`LibraryMediaRowScroll` all appear once each).

### Where §25's mount numbers come from — the arithmetic closes exactly

Measured DOM: media route **119** nodes, notes route **114**; media canvas
subtree **29** widgets, notes canvas subtree **19**.

* notes (switch) = **114** = the entire notes route tree, built once.
* media (switch-in) = **177 = 119 + 2 × 29** — the whole route tree, plus the
  media canvas built twice more.

The extra builds are the **redundant sync storm**, visible in the causal trace:
after the recompose returns at 114 ms, four `LibraryMediaCanvas.sync_state`
calls land at 118.5 / 118.6 / 123.9 / 124.0 ms (each is
`self.refresh(recompose=True)`, a full canvas child-list rebuild; consecutive
ones coalesce, so four calls produce two rebuilds). Worse elsewhere: **notes
(re-click same) fires seven `LibraryNotesCanvas.sync_state` and seven
`LibraryNoteWorkPane.sync_state` calls inside 6 ms.**

**So the freeze has two causes, and the plan named only the second: a
whole-screen rebuild per mode switch, and a sync storm that rebuilds the canvas
2–7× per click.**

Why the targeted path never runs on a real switch:
`_replace_library_browse_canvas` requires the destination's contextual chrome to
be already mounted, and media↔notes always crosses that boundary
(`#library-media-reader-shell` for media, `#library-notes-source-strip` for
notes), so it returns `False` in under a millisecond and only succeeds on a
re-click of the already-selected row.

## 2. Deliverable 2 — the mechanism spikes

Throwaway spike (`spike_residency.py`, scratch worktree only, **never landed**),
one mechanism per process so the arms could be interleaved and order-swapped.
n = 8 switches per run; `ref` and `A` were run three rounds each way (six runs
per arm) from the same scratch worktree. **The order swap moved nothing**: `ref`
block medians were 124.2 / 121.3 / 123.8 running first and 125.1 / 125.8 / 123.9
running second.

| mechanism | block (median) | cpu (median) | mounts | unmounts | nodes at rest |
|---|---|---|---|---|---|
| **ref** — today's rail switch | **124 ms** | **187 ms** | 114–179 | 121–176 | 119 |
| **A** — both canvases resident, `display` toggled | **30 ms** | **15 ms** | **0** | **0** | 136 (+17) |
| **B** — `remove()` then re-mount the same instance | 60 ms | 64 ms | 29 | 29 | 119 |
| **C** — mount-once-then-toggle (lazy residency) | **29 ms** | **14 ms** | **0** | **0** | 136 after first visit |

C's first visit: block 40–47 ms, cpu 38–49 ms, 17 mounts — already ~3× cheaper
than the switch it replaces, paid once per route per app run.
Under A/C the buckets collapse to render 12 ms, reflow 3 ms, **css 1.2 ms**.

**Threshold, stated from the data before choosing:** a mechanism must take
steady-state mounts to **0** and at least **halve** the longest main-thread
block. A and C clear it; B fails the first outright.

### Textual 8.2.8 behaviour, verified rather than assumed

1. `display = False` writes `styles.display = "none"` and keeps the widget
   registered and query-visible. The setter's own docstring records that it
   **forgets the CSS-declared value** (False→True resets to `block`), so phase C
   must toggle a class, not the boolean.
2. **Hiding the focused subtree drops focus to `None`** (measured:
   `Button#library-media-review-sets` → `None`). Focus must move before hiding.
3. **A hidden widget still receives and dispatches events** — `press()` inside a
   `display:none` subtree produced `Prompt`, `Pressed`, `Callback`. The plan's
   "a resident-but-unselected canvas must not process row events" ruling is
   *required*, not precautionary.
4. **`remove()`-then-re-mount of the same instance is accepted** (canvas back in
   the DOM, 28 children, `is_running` True, `_closing`/`_closed` False) — but it
   re-composes, so it is a remount wearing the old identity.
5. **`Widget.recompose()` destroys residency**: a resident canvas beside the
   active one is gone (0 hits) after one `screen.recompose()`.

### Composition verification

* **TASK-31521 screen reuse — compatible.** Suspend/resume touch timers and
  visit surfaces, never the DOM. Measured across a suspend/resume cycle with a
  resident hidden canvas: nodes 136 → 136, hidden canvas stayed
  `display: "none"`, `_library_screen_suspended` toggled correctly. The resume
  path's own liveness predicate (`_library_media_list_surface_active`) is gated
  on `_library_selected_row_id`, a **state** check, so it stays correct for free.
* **But DOM presence is a route proxy elsewhere.** 17 non-comment sites query
  `#library-media-reader-shell` and 5 query `#library-media-canvas`; several
  read presence as "the media route is active"
  (`_library_media_escape_label`'s `"back"` fallback;
  `library_notes_controller`'s `adaptive_media = bool(self.query(…))`). All
  become permanently true under residency. This is the largest single piece of
  Task 2's work.
* **`canvas_sync` — a NEW failure mode the blanket `except` cannot see.** Today
  `_sync_library_canvas(screen, "media")` off-route raises `NoMatches`, is
  swallowed, and falls back to `screen.refresh(recompose=True)`. Measured under
  residency: the same call **returns `True` and rebuilds the hidden canvas's 28
  children** — the query now succeeds, so there is no exception to swallow. The
  TASK-31880 coverage-solvent property is not merely preserved, it is upgraded
  to a silent success. **Scoped ruling for Task 2** (its own TASK-32089-labelled
  commit): add an explicit route-ownership guard ahead of the `try`, and narrow
  the blanket `except` so an unexpected failure raises instead of becoming a
  whole-screen recompose.

## 3. Deliverable 3 — the decision

Appended to the spec as **"Design record — phase C, media: the resident-canvas
mechanism"** (ADR shape: context / options with measured deltas / verified
Textual behaviour / composition rulings / decision / consequences / rejected).

**Decision: mechanism C, with the whole-screen recompose replaced first.** The
second half is the larger half — finding 5 makes it a precondition, so Task 2's
real scope is (1) extend the targeted route update across the media↔notes
structural boundary, (2) keep each visited canvas mounted and toggle
visibility, (3) gate resident-but-unselected canvases out of event handling and
out of `_sync_library_canvas`, (4) move focus before hiding.

C over A because their steady states are indistinguishable while C does not pay
resident nodes for the eleven canvas kinds (`_sync_library_canvas`'s own
dispatch table) a user never opens.
Rejected: **B** (2× worse on every column, and depends on re-mounting after
`remove()`); **A eager** (unbounded at rest); **optimising the CSS restyle**
(largest bucket at 39%, but it falls to 1.2 ms under residency — it is a
symptom of the rebuild); **making the whole-screen recompose faster** (finding 5
rules it out at any speed).

## 4. Deliverable 4 — the failing acceptance test

`Tests/UI/test_library_phase_c_switch_residency.py::
test_library_rail_mode_switch_does_not_rebuild_the_screen`

Pins the switch **structurally** rather than on a settle band: 0 whole-screen
`LibraryScreen.recompose()` calls, ≤ 25 mounts and ≤ 25 unmounts per media↔notes
rail switch. Both quantities are load-independent, so the pin is deterministic
on a loaded machine — which a settle-time pin is not.

Run, and its red (verbatim):

```
E       AssertionError: Library rail-mode switch still rebuilds the screen:
E           notes (switch): 1 whole-screen LibraryScreen.recompose() call(s), expected <= 0
E           notes (switch): 114 widget mounts, expected <= 25
E           notes (switch): 120 widget unmounts, expected <= 25
E           media (switch-back): 1 whole-screen LibraryScreen.recompose() call(s), expected <= 0
E           media (switch-back): 179 widget mounts, expected <= 25
E           media (switch-back): 175 widget unmounts, expected <= 25
1 failed, 94 warnings in 6.51s
```

Those numbers reproduce the probe's row-for-row (114 / 179 mounts, 120 / 175
unmounts, 1 recompose each), from the working worktree, confirming the test
harness measures the same thing the scratch-worktree probe does.

## 5. Deliverable 5 — battery sanity

Landed tree changes: the spec addendum, the acceptance test, and the probe
sibling script. **No production code touched.**

Paired run, working worktree vs the unmodified scratch worktree at the same
commit, same five files, same invocation
(`-p no:randomly -q`):

| tree | result |
|---|---|
| working (this task's changes) | **9 failed, 144 passed** |
| scratch (HEAD, untouched)     | **9 failed, 144 passed** |

The failing names are **identical in both trees**, so all nine are pre-existing
dev-side reds, not this task's:

* `test_library_selection_updates.py::test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`
* `test_library_entry_compose_once.py::test_library_graduation_announcement_survives_reconcile_and_same_route_replace`
* `test_library_entry_compose_once.py::test_library_notes_recompose_does_not_steal_newer_focus[reconcile]`
* `test_library_entry_compose_once.py::test_library_notes_recompose_does_not_steal_newer_focus[replace]`
* five in `test_library_honesty_accessibility.py`
  (`test_library_disabled_contrast_rules_live_in_source_and_bundle`,
  `test_compact_notes_editor_footer_context_actually_displays_at_60_cols`,
  `test_media_canvas_actions_share_one_toolbar_row`,
  `test_escape_works_on_export_and_staging_canvases`,
  `test_rail_entry_to_export_after_media_origin_does_not_claim_media`)

**Both dual-receiver guards are GREEN** in both trees
(`test_media_row_toggle_resolves_the_dotted_state_path` and
`test_notes_row_toggle_resolves_the_dotted_state_path`, both parametrised over
`receiver_kind`), as are `test_screen_reuse_helpers.py` and
`test_library_canvas_sync_defects.py`.

### The media suites

Working tree, all 19 `test_library_media_*.py` files plus
`test_library_multiselect_media.py`: **24 failed, 686 passed**.

Control, scratch worktree at the untouched HEAD, on the four files carrying
those failures: **19 failed, 236 passed** (`_reader_no_change_sync_t22208.py`,
`_render_fixes.py`, `_side_by_side.py`, `_trash.py`).

**The two failure sets are identical**, with one exception in one direction:

| | working | control |
|---|---|---|
| the 19 control failures | all 19 also fail | 19 |
| `_trash.py::test_media_trash_back_and_escape_restore_distinct_media_return[escape]` | fails **only in the full-suite run** | passes |

That single extra working-tree failure **passes when run on its own in the
working tree** (`2 passed in 5.65s`), so it is a load flake, not a regression.

One earlier reading of this comparison was wrong and is recorded rather than
quietly fixed: the control appeared to fail
`t22208::test_no_change_traversal_builds_no_preview_and_copies_no_content`
where the working tree did not. It does not — the working tree fails it too
(`2 failed, 3 passed` running that file alone). The name was missing from the
working-tree list only because the capture command ended in `tail -20`, which
kept 19 of the 24 names and dropped the collection-order-earliest ones. **A
truncated capture looked exactly like a behavioural difference**, which is the
same shape of error as the mount-by-region table that did not sum: a display
limit read as data.

Both runs shared the machine with **two other sessions' pytest processes** on
overlapping Library files, and these suites read and write shared user-data
databases under `~/.local/share/tldw_cli/default_user/`, which is the most
likely source of the one flake.

The evidence that actually carries the claim remains structural:
**`git diff --name-only b81cb98b0..HEAD` touches zero files under
`tldw_chatbook/`.** The landed diff is one spec addendum, one test file, one
Helper_Scripts probe and one SDD report; there is no mechanism by which it could
move a media suite.

**Flag for the ledger:** `test_library_selection_updates.py::test_tier1_toggle_
falls_back_to_recompose_on_query_one_failure` is red at HEAD. That test asserts
exactly the fallback behaviour the design record proposes to narrow, so Task 2
should read it before touching `canvas_sync`; and the five
`test_library_honesty_accessibility.py` reds sit in the file TASK-31880 just
re-greened one test in — the rest of that file was already red and stayed so.

## 6. Notes, risks, and what did not get done

* **Wall-clock projection is a floor, not a forecast.** A/C's 30 ms is what the
  mechanism costs on the canvas; a landed switch also re-applies the rail
  selection, header, footer context and focus. The acceptance test therefore
  gates on mounts/recomposes, and the wall-clock delta (block 124 → 30 ms,
  −76%; cpu 187 → 15 ms, −92%) stays in the record as context.
* **The sync storm is out of Task 2's scope** by the acceptance pin's design.
  Residency makes each redundant rebuild cheap but does not remove it. It is the
  obvious next motivated change; nothing here depends on it.
* **Node-count is the memory proxy this instrument can measure.** +17 nodes for
  media + notes resident (119 → 136, +14%). No RSS measurement was taken; if it
  matters, C's lazy structure supports a "keep the last N visited" eviction that
  A's does not.
* **No dependency was installed** for any spike; the scratch worktree's venv is
  a plain `uv pip install -e ".[dev]"` of the same commit.
* Spike code (`spike_residency.py`, `dump_tree.py`) lives only in the scratch
  worktree and is not committed anywhere.

## 7. Recommended lessons entry — for Task 4 to land, not this task

This task's scope for the landed tree was "the spec addendum + the acceptance
test + the probe sibling", so no `backlog/docs/lessons-*.md` edit was made here.
One belongs in `lessons-testing-evidence.md` and Task 4 (which already owns the
recipe §25 addendum) is the right place for it:

> **A probe column that reads zero is not evidence of zero; check what the hook
> actually intercepts.** `Helper_Scripts/library_click_probe.py` reported
> `recompose 0` and `0 removes` on every Library rail click for four waves.
> Both were instrument blind spots, not findings: the removes counter hooked
> `App._unregister`, which is not Textual 8.2.8's prune path
> (`Widget._message_loop_exit` is), and the recompose counter hooked
> `refresh(recompose=True)` while the code under measurement awaited
> `Widget.recompose()` directly. The `0` was promoted to a load-independent
> verdict column in recipe §25 and then written into phase C's success
> condition ("re-click rows go to ~0 mounts with recompose still 0") —
> a doctrine-level acceptance target resting on a hook that could not fire.
> Discovered only by writing a second instrument that measured the same
> quantity a different way and got a different answer.

The generalisable rule: **a counter that has never been non-zero has never been
tested.** Before trusting one, provoke the event it claims to count and confirm
it moves.
