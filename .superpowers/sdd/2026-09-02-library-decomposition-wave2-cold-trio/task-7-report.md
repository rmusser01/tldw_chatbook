# Task 7 report — Collections series 3/3: cleanup PR (shims out, ratchet lowered)

Branch `refactor/library-decomp-wave2-cold-trio`, worktree
`.worktrees/library-decomp-foundation`. Base commit `91feba4a7` (task 6's
fix-round-1 correction commit, collections controller PR complete). Recipe:
`backlog/docs/library-decomposition-recipe.md` §13/§14; the export cleanup
PR (task 4) is the worked example this mirrors mechanically.

## 1. Dynamic-dispatch census (before touching anything), with the new dict.get→variable→setattr guidance applied

- Re-derived task 6's own findings before deleting the shim block. The
  ONE screen-side dynamic-dispatch site touching Collections:
  `_replace_library_reader_preference`/`_persist_library_reader_
  preference`'s 7-destination `{destination: attribute_name}` dicts
  (shared shell dispatcher, same site the conversations exemplar's Task 9
  and export's Task 4 already fixed for their own subsystems). The
  `"collections"` value string changed from the flat
  `"_library_collections_reader_preferences"` to the dotted
  `"_collections_state.reader_preferences"`, read via the already-generic
  `operator.attrgetter(...)` and written via the already-generic
  `_assign_library_reader_preferences_attribute(owner, attribute, value)`
  helper — neither needed a logic change, only the string value (docstring
  extended with a short "Third use" paragraph documenting Collections as
  a second reuse, after the export series' own "Second use" paragraph;
  fixed the paragraph ORDER after an initial edit landed it out of
  chronological sequence — caught during self-review, not by a test).
- **The SAME dynamic-dispatch shape independently inside a TEST fixture**:
  `Tests/UI/test_library_adaptive_reader_closeout.py`'s `DESTINATION_
  CONTRACT` dict carries its own `"collections"` entry with the same two
  flat strings, consumed everywhere via `operator.attrgetter` (never bare
  `getattr`) — mirroring the fix this file's own `"conversations"` entry
  needed at task 9. Retargeted both strings; added a matching explanatory
  comment (mirroring the "conversations" entry's own comment) and
  corrected the "not-yet-extracted destinations" count from five to four
  in both comments now that Collections is extracted too.
- **The controller-internal `dict.get()`→variable→`setattr` site task 6's
  fix round found** (`retain_library_collection_quick_capture_input`):
  confirmed OUT OF SCOPE — lives entirely inside
  `LibraryCollectionsController` (moved byte-for-byte by task 6),
  dispatches to 3 of the controller's own generated state-shim
  properties (full `property(get, set)` pairs), never routed through
  anything this task touches. Per the task brief's own framing ("Known
  dynamic-dispatch site INSIDE THE CONTROLLER (safe, stays)"),
  re-verified rather than re-fixed.
- **Applying the new guidance itself**: grepped `library_screen.py` and
  every retargeted test file for a `.get(...)` result assigned to a
  variable that later flows into `setattr`/`getattr` within the same
  function. Zero NEW instances found touching any of the 26 Collections
  state fields or the 14 pruned method names.

## 2. Screen-side field retarget

**14 literal `self._library_collections_<field>` sites** across the
`__init__` entangled-trio lines (5), `_replace_library_reader_preference`
(1 dict-string), `_persist_library_reader_preference` (2: dict-string +
locks-dict), `request_library_reader_layout_refresh`'s
snapshot-unpack/current-values block (2), `_toggle_library_media_reader_
pane` (1), `restore_state` (2), and the Collections reader-shell `compose`
call site (1) — one mechanical, word-boundary-anchored, longest-match-
first regex pass over a 26-field mapping table rewrote all 14 to
`self._collections_state.<field>`; re-verified afterward with a
zero-result grep for every one of the 26 flat field names over the whole
file. `_library_collections_capture_controller` (the ONE field task 5
deliberately kept OFF `LibraryCollectionsState` — "wiring, not state")
was excluded by construction: it is not one of the 26 mapped names, so
the regex never touched its 11 occurrences.

Unlike export/conversations, the vast majority of Collections' 26 fields
have ZERO remaining screen-side literal references after task 6's
controller move — only the three entangled fields (`reader_preferences`,
`reader_persistence_locks`, `reader_layout`) and `requested_page` still
had any screen-side literal access, because task 6 moved all 64 cluster
METHODS (the only other consumers) onto the controller in one PR with
zero exclusions, leaving no screen-resident method to reach the other 22
fields directly.

## 3. Test retarget — per file, widened beyond the brief's stated scope

Repo-wide grep for `_library_collections_<field>` across the WHOLE
`Tests/` tree (not just `Tests/UI/`/`Tests/Library/`, the brief's named
scope) found a THIRD location the brief didn't name:
**`Tests/Live/test_library_collections_capture_walkthrough.py`** — a
non-network-gated, `@pytest.mark.asyncio` full-harness walkthrough file
collected in an ordinary pytest run (confirmed via `pyproject.toml`'s
`addopts`: no `--ignore=Tests/Live`) that reaches 22 flat Collections
field names directly on a real `screen` instance. Left unretargeted,
these would have broken silently at shim deletion — undetected by any
check scoped to `Tests/UI`/`Tests/Library` alone.

| File | Retargets | Notes |
|---|---|---|
| `Tests/UI/test_library_collections_characterization.py` | 13 (`screen`) | 1 docstring paragraph updated future→past tense |
| `Tests/UI/test_library_collections_capture_reader.py` | 8 (`screen`) | |
| `Tests/UI/test_library_adaptive_reader_closeout.py` | 5 (`screen` + 2 `DESTINATION_CONTRACT` dict strings) | dynamic-dispatch fixture fix, §1 |
| `Tests/UI/test_library_shell.py` | 1 (`screen`) | inside a shared multi-parametrization helper |
| `Tests/Live/test_library_collections_capture_walkthrough.py` | 22 (`screen`) | not named by the brief; found by widening the census root |

**49 retargets total, zero assertion VALUE changes** — every one is a
receiver-path rewrite only. No unbound-fake-self construction touches any
Collections field (task 6's own battery already confirmed none exist for
this cluster), so no fixture needed flat-kwargs→nested-`SimpleNamespace`
restructuring.

## 4. Delegator census (all 64)

Repo-wide grep (`tldw_chatbook/`, `Tests/`, including `Tests/Live/`) for
every one of task 6's 64 moved-cluster names:

| Category | Count | Names / disposition |
|---|---|---|
| `@on`-decorated handlers | 41 | KEEP unconditionally (per brief: "`@on` always stays") |
| Non-`@on`, screen-resident caller beyond own delegator | 5 | `_sync_library_collections_reader_layout_from_shell`, `_mirror_library_collections_reader_preference`, `_restore_library_collections_page`, `_library_collections_capture_presentation`, `_load_library_collections_capture_entry` — KEEP |
| Non-`@on`, direct test call on a real `screen` instance | 4 | `_refresh_library_collections_capture_reader`, `_run_library_collections_capture_transition`, `_select_library_collection_capture`, `_export_library_collection_legacy_recovery` — KEEP |
| Non-`@on`, zero consumers beyond own body + wiring-test shape-check | 14 | **PRUNED** (list below) |

**Net: 50 KEEP, 14 PRUNED.**

Pruned: `_library_collections_capture_request`,
`_ensure_library_collections_capture_controller`,
`_notify_library_collections_warning`,
`_capture_library_collection_quick_capture_draft`,
`_reset_library_collection_quick_capture_draft`,
`_submit_library_collection_quick_capture`,
`_library_collection_capture_filter_request`,
`_apply_library_collection_capture_request`,
`_page_library_collection_captures`,
`_update_selected_library_collection_capture`,
`_library_collection_loaded_capture`,
`_library_collection_capture_is_current`,
`_load_library_collection_capture_highlights`,
`_run_library_collection_capture_content_action`.

This 14/64 (~22%) prune fraction is much larger than export's 1/22 (~5%)
and closer to (but still smaller than) conversations' 18/61 (~30%) —
directly explained by task 6's own finding of ZERO method-level test-
bypass exclusions: the entire 64-method cluster moved onto ONE controller
in a single PR with no screen-resident sibling left calling any of these
back, unlike export's 11 round-2/round-3 exclusions (which each kept
calling their own delegator internally, keeping 21 of 22 alive).

**Type-annotation follow-on**: `CapturePageRequest` appeared in exactly 3
signatures, ALL THREE inside pruned delegators — pruning all three made
the import newly dead (folded into §6). `CaptureIdentity` (used by both a
KEPT delegator and a PRUNED one) and `CollectionsCaptureReaderPresentation`
(used only by a KEPT one) both stay alive — checked individually.

## 5. Shim block deletion

The task-5-generated `_library_collections_<field>` property-shim loop
was deleted wholesale once §§2–4 confirmed zero remaining consumers
anywhere in `tldw_chatbook/` or `Tests/` outside
`LibraryCollectionsController`'s own PERMANENT generated shim loop
(task 6, `self._collections_state_accessor().<field>` — untouched, per
"controller shims STAY").

## 6. Import verification

**10 dead imports removed**, each verified single-occurrence (import line
only) via per-name grep, then checked against
`Tests/Architecture/test_library_support_layer_surface.py`'s `_SURFACE`
dict — none of the 10 belongs to any of its 5 listed modules
(`screen_constants`, `screen_support_types`, `note_session_port`,
`canvas_sync`, `screen_helpers`):

- 4 flagged dead by task 5: `CaptureCapabilities`, `CaptureHighlight`,
  `SavedCaptureSearch`, `CollectionsReaderMode`.
- 4 flagged dead by task 6 (left for this cleanup PR per the export
  series' Task 3/Task 4 split): `CAPTURE_SORTS`, `CaptureSaveRequest`,
  `CollectionsCaptureError`, `ExternalNoteReference`.
- 1 flagged dead by task 5 for a different reason:
  `CollectionsCaptureControllerState`.
- 1 newly dead as a direct result of this task's own delegator prune:
  `CapturePageRequest`.

`CaptureIdentity` and `CollectionsCaptureReaderPresentation` (task 6's
own "do NOT treat as dead" warning) re-checked post-prune: still alive (2
occurrences each), left in place.

## 7. Wiring test finalization

`Tests/Architecture/test_library_collections_wiring.py`:
`test_state_object_fields_match_the_shim_surface` DELETED (screen shim
gone); `_COLLECTIONS_CLUSTER_SCREEN_DELEGATOR_PRUNED` frozenset (14
names) added, `test_screen_delegates_collections_handlers` updated to
skip those names and instead assert their genuine ABSENCE from
`LibraryScreen`; module docstring rewritten. 4 of the original 5 tests
remain (shim-surface test removed), all green.

## 8. Size ratchet

Fresh measurement via the ratchet's own `_measure` semantics (`ast`-walk
line count + `LibraryScreen` method count): **42411 lines, 1267 methods**
— 1267 = 1281 (task 6's post-move count) − 14 (exactly the pruned
delegator count). Lowered in this same commit per recipe §6.

Pin trajectory: `43410/1281 (pre-task-5) → 42486/1281 (task 6) →
42411/1267 (task 7, final)`.

**Rebase note**: `origin/dev` had diverged from this branch's merge-base
by 337 commits by the time this task ran, dozens touching
`library_screen.py`. Rebasing a multi-week, 3-task-deep decomposition
rehearsal branch onto that much unrelated concurrent work was judged out
of scope for a single cleanup task and not requested by the brief
(worktree-scoped, no pushes). Measured fresh on this branch's own HEAD
instead, consistent with tasks 2–6's own practice. Flagged in the recipe
(§15) for whoever runs the next subsystem's series.

## 9. Verification battery

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`.

- **Wiring suites**: collections (4) + export (5) + conversations (6) —
  15 passed.
- **Characterization files**: collections (5) + export (5) + conversations
  (4) — 14 passed.
- **Recompose ratchet + support-layer surface**: 14 passed (recompose pin
  unaffected — zero `refresh(recompose=True)` sites touched).
- **Both size-ratchet guards, full suite**: 3 passed, 2 failed (the two
  documented pre-existing `chat_screen.py` rows only).
- **Collections-adjacent live-functional suites** (capture reader,
  capture controller, reader geometry, phase-39 guard): 43 passed
  (matches task 6's own count for this battery).
- **`test_library_adaptive_reader_closeout.py`** (full file, incl. the
  fixed `DESTINATION_CONTRACT` table and `test_closeout_single_app_
  route_cycle`, which task 6's fix round proved DOES exercise 2
  Collections methods): 14 passed.
- **`Tests/Live/test_library_collections_capture_walkthrough.py`**: 2
  passed, 1 skipped (network-gated, unrelated).
- **`test_library_landing_continue_receipt_accepts_only_authoritative_
  source_scopes`** (contains the retargeted line): 5 passed, 1 failed —
  the SAME `[browse-collections-expected_scope4]` failure task 5's own
  report already documented, reconfirmed identical in symptom
  (`AssertionError: assert None == {...}`).
- **`-k "collection and library"` across `Tests/UI` + `Tests/Library`**:
  **361 passed, 3 failed** — the SAME 3 names task 5/6 already
  documented, matched exactly (name-for-name). Per recipe §7's own "check
  this list before re-deriving" guidance, no separate `git stash -u`
  baseline was run for this narrow check since the match was already
  exact.
- **Combined quick battery re-run** (all wiring/characterization/
  collections-adjacent/closeout/walkthrough files in one invocation, post
  stash-pop restore): 106 passed, 2 failed (the same 2 pre-existing
  `chat_screen.py` rows), 1 skipped.
- **Preflight**: all six checks green, run twice (once mid-task, once
  final before commit).
- **Full xdist paired-baseline sweep** (`Tests/UI -k "library" -p
  no:randomly -q -n 8 --dist worksteal`), run SEQUENTIALLY per task 6's
  own forward note (branch first, then a `git stash -u` pristine
  baseline of the same `91feba4a7` tree, restored via `git stash pop`
  afterward and re-verified via `ast.parse` on all 9 touched files +
  `git status --short`):
  - Branch: **333 failed, 3906 passed** (1314.45s / 21:54).
  - Baseline: **337 failed, 3902 passed** (1306.25s / 21:46).
  - Diff: **4 branch-unique**, **8 baseline-unique**, remainder shared.
  - Branch-unique names: `test_audio_cpp_model_library_handoff.py::
    test_rapid_away_back_reclaims_request_after_old_operation_drains`,
    `test_library_media_reader_traversal_t22207.py::test_loading_banner_
    paints_in_place_without_body_rebuild`, `test_library_prompts_canvas.py::
    test_library_prompt_history_no_change_keeps_selection_and_retry_
    available`, `test_library_shell.py::test_library_media_page_error_
    retains_rows_and_gates_unsafe_controls`.
  - All 4 re-run single-process, combined, on the branch: 2 passed
    cleanly, 2 failed (`test_rapid_away_back_reclaims_request_after_old_
    operation_drains`, `test_loading_banner_paints_in_place_without_
    body_rebuild`). Both of those 2 **passed individually in true
    isolation**.
  - The SAME 4-test combined invocation re-run against the PRISTINE
    baseline (via a second `git stash -u`/`git stash pop` round trip):
    ONE of the two reproduced (`test_rapid_away_back_reclaims_request_
    after_old_operation_drains`), the OTHER passed on baseline this time
    — a different subset flaking each run, the same "shared-state/
    ordering sensitivity to which OTHER tests ran earlier in the
    process, identical on both code versions" shape task 6's own report
    established for a different test
    (`test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke`).
  - **Zero real regressions**: none of the 4 branch-unique names touches
    Collections code, this task's diff, or a fixture this task's diff
    shares (Audio/CPP model handoff, Media reader traversal, Prompts
    canvas, Media page-error) — confirmed by reading each test file's own
    scope, not inferred from the name alone (the recipe's own §7 lesson
    from task 6's fix round).

## 10. Recipe diff summary

`backlog/docs/library-decomposition-recipe.md`: new §15 ("The collections
series, task 3 (cleanup PR) — as landed") with the full dynamic-dispatch
census (including the new guidance's application), field/test-retarget
counts, the delegator census table, import verification, pin trajectory
(with the rebase-note deviation flagged forward), and the full battery
including the sequential paired-baseline sweep. Two lessons recorded:

1. The cleanup-PR test census should widen past the brief's stated
   directories, the same way a controller-PR battery's `-k` search
   should (§12's own lesson, now shown to generalize to a DIFFERENT task
   type) — `Tests/Live/` had 22 real consumers the brief's named scope
   would have missed.
2. A cluster with ZERO controller-PR method-level exclusions can still
   produce a LARGE cleanup-PR prune fraction — the mechanism is inverse
   to what might be assumed: exclusions are what KEEP screen delegators
   alive (an excluded sibling calls its moved neighbors internally); a
   cluster that moves entirely onto one controller in one PR has no such
   sibling left, so more purely-internal helpers lose their only caller
   in the same step.

## 11. Self-review

- Every field/test retarget was verified by re-running the affected file
  (or a scoped group) before moving to the next; the docstring-ordering
  mistake in `_assign_library_reader_preferences_attribute` (my own
  "Third use" paragraph landing before the existing "Second use"
  paragraph) was caught during a self-review diff read, not a test
  failure, and fixed before the battery ran.
- The `Tests/Live/` widening was not assumed safe because "it's not named
  in the brief" — confirmed via `pyproject.toml`'s own `addopts` that no
  ignore rule excludes it from a normal collection, then verified by
  actually running the file both before AND after the retarget.
- The 14-of-64 delegator prune was individually verified per name (not a
  batch assumption) via a repo-wide grep restricted to files outside the
  controller module, the screen's own delegator body, and the wiring
  test's shape-check list.
- The docstring/comment accuracy pass touched the `_assign_library_
  reader_preferences_attribute` helper docstring, the `DESTINATION_
  CONTRACT` test fixture's inline comments (count corrected from five to
  four not-yet-extracted destinations), and the characterization file's
  future→past-tense paragraph — each cross-checked against the actual
  post-cleanup code shape, not copy-pasted from a sibling task's
  precedent unmodified.
- The full xdist paired-baseline sweep was run SEQUENTIALLY, per task 6's
  own forward note, not concurrently — costing more wall-clock time but
  avoiding the CPU-contention-amplified-flakiness task 6's own concurrent
  run documented.
- Both `git stash -u`/`git stash pop` round trips (once for the full
  sweep baseline, once for the 4-test combined-repro baseline) were
  verified clean afterward via `git status --short` and an `ast.parse` of
  every touched file — no stray checkout state survived either round
  trip.
- Commit hashes below were read via `git rev-parse HEAD` immediately
  after each commit, not typed from memory.

## 12. Commits

- `39a9763215eab9b8a832e45e4ebb7e85893d8b60` — `refactor(library):
  collections cleanup — shims out, ratchet lowered (collections series
  3/3)` (implementation; confirmed via `git rev-parse HEAD` immediately
  after committing).
- `1e466ffac192a508b1320810629c538406eda000` — `chore(library):
  blame-ignore follow-up for collections cleanup PR` (appends the hash
  above to `.git-blame-ignore-revs`; confirmed via `git rev-parse HEAD`).
