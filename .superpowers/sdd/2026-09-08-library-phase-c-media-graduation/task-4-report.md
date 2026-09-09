# Task 4 report — the graduation close

**STATUS: COMPLETE. The branch is PR-ready.** Media has graduated: the Library
rail-mode switch that opened the design doc as a measured 139–380 ms
main-thread freeze now blocks the main thread for **51–78 ms**, with the one
whole-screen recompose per switch eliminated (1 → 0) and the mount storm cut
7× on media and 4× on notes. Zero production code was touched in this task —
it is the evidence-and-filings close of the four-task series.

Branch `feat/library-phase-c-resident-canvas`, worktree
`.worktrees/library-decomp-foundation`, base `7e81ed55d` (the wave-8 merge
into `dev`), HEAD before this task `b2ccbca09`.

| commit | what |
|---|---|
| `a339eb518` | docs: the graduation close (spec graduation record + notes-readiness; recipe §25 instrument correction + §26; the task-1 lesson; user-guide stamps) |
| _(report commit)_ | this report |
| _(filings commit — LAST)_ | the two follow-up tasks, filed after the id sweep |

---

## A. The payoff — the before/after freeze table (the founding number)

Landed in the spec design record ("Graduation record — phase C, media") and in
recipe **§26**. Freeze metric = the teardown probe's `block` (longest single
main-thread stall on a switch), the instrument task 1 landed. Every cell
traceable to a task + commit:

| stage | commit(s) | media switch-back block | notes switch block | whole-screen recompose | mounts (media / notes) |
|---|---|---|---|---|---|
| **base (pre-phase-C)** | `7e81ed55d` | **93 ms** (86–95) | **140 ms** (126–149) | **1** | 179 / 114 |
| task 2 — structural cause removed | `7b4a7e289..5fc35f1fa` | 98 (unchanged) | 147 (unchanged) | **0** | 81 / 26 |
| task 2.5 — felt freeze halved | `497bc5f8e..4645e4c8c` | **55** (51–65) | **73** (68–78) | 0 | 60 / 26 |
| task 3 — region ownership (routing-only) | `480eadbba`, `cad52260b` | 50–55 (in band) | 67–82 (in band) | 0 | 58 / 26 |
| **now** | `b2ccbca09` | **51 ms** (49–53) | **78 ms** (66–79) | **0** | 58 / 26 |

**The definitive probe** (Deliverable F) is the `base`/`now` rows: ONE scratch
worktree at `/private/tmp/phasec-graduation`, `tldw_chatbook/` reverted between
`7e81ed55d` and `b2ccbca09` so both arms sit at the same checkout location per
recipe §9; **interleaved and order-swapped** (`b m m b b m m b b m m b`),
**n = 6 per arm**. The base arm reproduced task 1's pre-phase-C teardown to the
digit (media switch-in 177 mounts / 1 recompose; notes switch 114 / 1; media
switch-back 179 / 1), which is what proves the revert-only-`tldw_chatbook/`
protocol measured the true base.

**The block ranges do not overlap on either switch arm** — media gap 33 ms
(86–95 vs 49–53), notes gap 47 ms (126–149 vs 66–79). The durable claim is the
load-independent verdict: recompose **1 → 0** on both; mounts **179 → 58**
(media), **114 → 26** (notes); unmounts 176 → 63, 121 → 2. Unchanged on
purpose: `media (switch-in)` is the ordinary-route ENTRY (not a resident-shell
switch), still 1 recompose / 177 mounts on both arms.

Screen shrink across the phase: `32351 / 1259` → `32178 / 1243`. The −173
lines split across three stages — task 2 −109 (32351 → 32242), task 2.5 −1
(→ 32241), task 3 −63 (→ 32178); the −16 methods are all task 3 (1259 held
through task 2.5, then 1259 → 1243).

## B. The recipe correction phase C carried (the §25 instrument bug)

Recipe §25's published phase-C baseline promoted `recompose 0` to a
load-independent verdict column and wrote "recompose still 0" into phase C's
success condition — on a column that is **instrument-blind on the rail-switch
path**. Landed the correction as an in-place annotation (a `†` note on the
four switch rows of the baseline table + a correction under the success
condition), NOT a rewrite, so the blind spot stays visible:

* the click probe's `recompose` counter hooks `refresh(recompose=True)`, but
  the rail switch awaits `Widget.recompose()` DIRECTLY (bypassing `refresh`),
  so the hook cannot fire there;
* the sibling `removes` column hooks `App._unregister`, which has **zero
  callers anywhere in Textual 8.2.8** (prune finalises in
  `Widget._message_loop_exit`) — it could never fire on any path;
* the truth-measuring instrument reads **1 whole-screen recompose** on exactly
  the three switch rows, reproduced base-vs-HEAD by this task's definitive
  probe.

Task 1's review confirmed all three; recorded in the annotation. The
generalisable lesson landed in `backlog/docs/lessons-testing-evidence.md`
("A probe column that reads zero is not evidence of zero — a counter that has
never been non-zero has never been tested").

## C. Filings

Id sweep run as the LAST commit before handback (dev mints ids hourly — the
twice-struck collision lesson). Existing checked: TASK-31880 Done;
TASK-32088 / TASK-32089 filed (32089 covers the dispatcher receiver-guard debt,
NOT the two items below); no double-file.

**Id sweep (the LAST action before handback).** `git fetch origin` then the
robust all-refs sweep (`git rev-list --objects --all | grep -oE 'task-[0-9]+'
… sort -rn`) returned **true max 32169** — while local `backlog/tasks/` max was
only 32097, so the CLI's auto-assignment (32098/32099) would have collided with
ids dev minted on remote refs not in this tree. Filed above the true max after
confirming free as both filename and content ref; re-swept once more
immediately before renumbering (still 32169). Preflight's duplicate-id check is
green with both present (3470 task files).

* **TASK-32170** — *Library phase-C media: extract the 20 deferred canvas-origin
  handler bodies onto `LibraryMediaController` so their `@on` rows can
  graduate.* The phase-A-style follow-on from task 3: the 20 rows are
  canvas-origin but their bodies are still on `LibraryScreen`, outside
  `_MEDIA_CLUSTER_METHOD_NAMES`; moving the `@on` row alone would call back
  through the screen. The full per-row blocker list (the analyze and
  bulk-delete families, `delete_selected`, the empty-state pair,
  `export_selected`, the review-dismiss pair, the media row, `select_toggle`,
  `trash_open`, the type-filter pair) is in the task and in
  `test_library_phase_c_region_ownership.py::_MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS`.
* **TASK-32171** — *`test_focus_traversal_builds_zero_bodies_for_pass_through_
  rows` fails ~1-in-8 on an unchanged tree (genuine nondeterminism, not a load
  flake).* Production-revert-controlled in task 3 (still fails with production
  reverted; 7/8 on both arms). The prior "passes in isolation" label was too
  generous.

**Documented-not-filed** (the reports mark these documented-with-reasoning, not
file-worthy): the blanket-`except` narrowing owed in `_sync_library_canvas`
(task 2 §6c — reverted because `test_strict_failure_retry_retains_original_
semantic_focus` requires the swallow for the reconcile retry; reasoning at the
code site, and it belongs under the canvas_sync dispatcher-debt umbrella,
TASK-32089, if ever revisited); `actions=None` as a source-pinned silent-
failure shape (task 3 §8.2); the residency refusal spelled in three places
with no single enumerator (task 3 §8.3 / task 2.5 §8.2); the canvas widgets
painting in place rather than recomposing their child list, the remaining
per-switch lever (task 2.5 §5 — a canvas-widget change, not a storm fix).

## D. Notes-graduation readiness

Landed as a spec section ("Notes-graduation readiness"). **Transfers:** the
resident-shell mechanism (`library_browse_route_swap.py` — the Notes canvas
already shares the host and shell); the marker-class route model (route-neutral
`#library-browse-reader-shell` + per-route markers written by the single
`apply_route` seam with `update=False`); all four guards (dual-receiver,
route-ownership + refusal-direction, hidden-canvas, same-node-dispatch); the
TASK-31521 suspend/resume ruling. **Does NOT transfer:** notes has no
row-geometry message (media's 16th migrated row has no analogue); its
canvas-origin/deferred/permanent census must be taken fresh from source (the
media 16/20/43 split does not predict it); and the sync-storm shape differed —
notes' worst arm was a first-entry placeholder-layout collapse, not media's
outgoing-canvas rebuild, so a notes graduation must OPEN with the attribution
measurement, not assume media's leads.

## E. Stale-doc sweep + lessons

* **The two lessons task 3 drafted are already landed** (commit `cad52260b`):
  the same-node `@on`-before-`on_message` dispatch-order trap (with the
  "trap inside the trap": a guard's pin only covers handlers routing through
  the guard) in `lessons-textual.md`, and the four-fails-in-a-row-is-not-
  attribution / revert-control lesson in `lessons-testing-evidence.md`. Both
  read complete and high-quality; verified, not re-landed.
* **The task-1-recommended lesson** (the probe-column instrument-blindness,
  which task 1 explicitly deferred to task 4) is now landed in
  `lessons-testing-evidence.md` — see §B.
* **Stale-doc sweep:** the user guide's task-2 stamp said the perceived pause
  was "unchanged so far", which read as stale against the task-2.5 stamp
  directly below it — reconciled with a forward-pointer (the dated ledger entry
  is preserved, not deleted), and a task-4 graduation-close stamp added. The
  recipe's §8 phase-C note ("first candidates: media and notes") updated to
  record media's graduation and point to §26. No other stale phase-C claim
  found across `Docs/` and `backlog/docs/`.

## F. Full close battery

All from the working worktree, `-p no:randomly`. **This task changed zero
production code, so every red below is a documented pre-existing one, verified
unchanged.**

### Phase-C pins — all green

| suite | result |
|---|---|
| `test_library_phase_c_region_ownership.py` | 84 passed |
| `test_library_phase_c_resident_canvas.py` (event gate, route-ownership guard + refusal-direction, suspend/resume, marker invariant) | 5 passed |
| `test_library_phase_c_switch_residency.py` (acceptance) | 1 passed |
| `test_library_phase_c_switch_storm.py` | 6 passed |
| both dual-receiver guards (`test_{media,notes}_row_toggle_resolves_the_dotted_state_path`, parametrised over `receiver_kind`) | passed (within `test_library_selection_updates.py`) |

96 phase-C pins + guards green in one run (105 passed / 1 failed, the failure
being the documented `tier1_toggle` pre-existing red).

### Ratchets / wiring / reuse / characterization / preflight

| suite | result | verdict |
|---|---|---|
| `test_library_media_wiring.py`, `test_library_notes_wiring.py`, `test_library_support_layer_surface.py` | green | green |
| `test_screen_size_ratchet.py` (library row) | **2 passed** (lines + methods at 32178 / 1243) | green — 2 chat_screen reds are dev-side, pre-existing |
| `test_library_modules_size_ratchet.py` | 2 failed (`library_media_browse_controller`, `library_conversations_controller`) | pre-existing (tasks 2/2.5/3) |
| `test_library_recompose_ratchet.py` | 1 failed, **67 found / 63 allowed** | pre-existing, site-count-NEUTRAL (exactly tasks 2/2.5/3's 67) |
| `test_screen_preimport_payload_budget.py` | 1 failed at **504** modules (limit 500) | unchanged (exactly tasks 2/2.5/3's 504) |
| `test_library_screen_reuse.py`, `test_screen_reuse_helpers.py` | 1 failed (`test_on_screen_suspend_stops_every_timer_in_isolation`) | pre-existing and deterministic |
| `test_library_media_characterization.py`, `test_library_notes_characterization.py` | green | green |
| `test_library_canvas_sync_defects.py` | green | green |
| `test_library_canvas_scoped_sync.py` | 1 failed (`test_notes_per_click_updates_keep_screen_and_canvas_identity`) | pre-existing (tasks 2/2.5/3) |
| `./scripts/preflight.sh` | **all checks pass, zero drift** | green — no new diagnostic statement, no CSS change, no schema change, no duplicate task id (3470 task files) |

### The media battery (19 files + `test_library_multiselect_media.py`)

Full run, one process, `-p no:randomly -q`: **23 failed / 686 passed**
(709 tests, 14m30s). Within the documented drift band (task 1 saw 24–27,
task 2 26/27, task 2.5 24–26, task 3 25) — the drifting rows are the
load-sensitive focus-survival and wall-clock probes these suites are known to
carry, run on a machine sharing the user-data DBs with other pytest processes.

**Zero branch-unique failures, proven structurally rather than by re-running
the base.** `git diff --name-only b2ccbca09..HEAD -- tldw_chatbook/` is EMPTY
and this task's only commit (`a339eb518`) touches zero files under
`tldw_chatbook/`, so production is byte-identical to task 3's reviewed-clean
HEAD, whose battery the task-3 review confirmed matched base-vs-changed
name-for-name (25f/684p both arms). A base-vs-HEAD media-battery pairing is
NOT apples-to-apples here — the HEAD media test files carry phase-C retargets
(`return_settlement`, `review_set_walker`) that assume phase-C production — so
the structural fact is the valid evidence, exactly as task 1 established for
its own battery.

All 23 failures land in the five files the prior reports document as carrying
pre-existing reds — `reader_flow` (4: row-focus survival ×2 sizes,
background-recompose focus ×2), `reader_no_change_sync_t22208` (2: no-change
traversal + image wall-time probe), `render_fixes` (11: select-mode focus,
More layout, two-row paint, analysed-marker render, row-action grid),
`side_by_side` (2), `trash` (4: initial-focus settle, retry-focus, status
fold) — and their assertion shapes are the documented ones: `_row_is_painted_
focused(...) == False`, "Trash focus did not settle", the leading-`█`
focus-cursor render diffs, and `assert 46 > 46` wall-time boundaries. None are
import/collection errors. The flake (`test_focus_traversal_builds_zero_bodies_
for_pass_through_rows`, Filing C.2) did not fire this run — expected at its
~7/8 pass rate — and `return_settlement` (task 3 fixed, 44 tests) stayed green.

### The definitive probe

Recorded in §A and in the spec's graduation record. Base arm reproduced task
1's pre-phase-C numbers to the digit; both switch arms non-overlapping.

## Concerns / what the next task should know

1. **The next phase-C graduation is notes**, and the readiness note says what
   to reuse and what to re-measure. The single most important carry-over: OPEN
   with the attribution measurement, do not assume media's leads — the storm
   shape differed.
2. **The deferred 20 are the real remaining media scope** (Filing C.1) and are
   gated on a phase-A body extraction, not on phase C. Until they move, the
   canvas owns 16 of the 36 rows it structurally could.
3. **The `tier1_toggle`, `notes_per_click`, suspend-timer, controller-ratchet,
   chat-ratchet, recompose-67, preimport-504 reds are all dev-side / documented
   pre-existing** and unchanged by this branch. They are the §7 documented set;
   do not re-derive them.
4. **The flake** (`test_focus_traversal_builds_zero_bodies_for_pass_through_
   rows`, Filing C.2) is a genuine ~1-in-8 nondeterminism on an unchanged tree
   and will keep costing attribution time in every media-battery sweep until
   fixed.

No pushes. `progress.md` untouched. No subagents. The scratch worktree at
`/private/tmp/phasec-graduation` was removed at the close.
