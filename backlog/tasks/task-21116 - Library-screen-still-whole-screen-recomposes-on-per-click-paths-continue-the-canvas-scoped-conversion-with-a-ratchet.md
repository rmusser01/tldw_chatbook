---
id: TASK-21116
title: >-
  Library screen still whole-screen-recomposes on per-click paths - continue the
  canvas-scoped conversion with a ratchet
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 15:28'
labels:
  - performance
  - library
  - recompose
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21116).

library_screen.py holds ~105 statement-level `refresh(recompose=True)` sites (99 on self =
whole-screen) on a screen that grew 26k -> 34.8k lines since the audit pin. Confirmed per-click
sites: `_open_library_item_by_id` (rail row / RAG result / media open), `_apply_library_row_toggle`,
media-viewer back (Escape), skills/prompts import open/cancel, export open. The 15457
canvas-scoped seam (`library_canvas_sync`, 82 call sites) is the sanctioned conversion target;
9 whole-screen sites were re-added post-fix on low-frequency admin flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The confirmed per-click sites above no longer rebuild the whole screen (converted to the canvas-scoped seam or narrower)
- [x] #2 A ratchet test pins the statement-level whole-screen recompose count at (or below) the post-conversion number
- [x] #3 Click-to-settle timing on a rail-row open, before/after, recorded in the task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: teed runs of the Library suites covering the touched flows (canvas sync, selection updates, multiselect media/notes, media trash/side-by-side, prompts/skills canvases, export, RAG handoffs, shell) at base 30c7e1fe9.\n2. Probe each confirmed per-click site on the mounted harness (screen-recompose spy) to confirm which actually whole-screen-recompose per click, incl. whether _apply_library_row_toggle's in-place patch holds or falls back.\n3. Convert site by site: skills/prompts import open/cancel -> _sync_library_canvas(kind, then=focus); media-viewer sub-state Escape branches -> viewer-scoped _sync_library_media_viewer_state; media viewer exit + media open + _open_library_item_by_id non-entry paths + export open -> canvas-host child replacement via the strict _replace_library_canvas_child seam (+ explicit _register_footer_shortcuts), whole-screen recompose kept only as failure fallback; _refresh_library_media_detail completion -> targeted child swap.\n4. Red-first AST ratchet test pinning the statement-level whole-screen recompose count (fails on the pre-conversion count; message names the sanctioned seams).\n5. Evidence: rail-row open (media row click-to-settle) timing probe before/after, mounted identity tests for converted sites.\n6. Re-run Library suites + full --collect-only sweep; record per-site table, counts, timings in Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Converted every confirmed per-click whole-screen recompose path in `UI/Screens/library_screen.py` to a targeted seam, extended the strict entry-replacement seam with a light rail mode and a post-sync follow-up hook, and pinned the remaining count with an AST ratchet. Statement-level whole-screen recompose count: **107 (base `30c7e1fe9`) -> 97** (10 net; 15 per-click statements removed, 5 sanctioned fallback / structural-boundary arms added inside the new seams).

### Per-site conversion table (line numbers as of base)

| Site (base) | What it refreshes now | Why safe |
|---|---|---|
| `_open_library_item_by_id` prompt open (:33245) | `_apply_library_open_item_surface` -> `_replace_library_canvas_child(rail_mode="selection")`: rail selection patched in place + canvas child swapped | Same widget and state-set the entry-origin arm already used; `FAILED` still falls back to whole-screen |
| `_open_library_item_by_id` media open (:33293) | same seam, `_build_library_media_active_child` | ditto |
| `_open_library_item_by_id` notes open (:33337) | same seam, which detects the route-owned Notes source-strip mismatch and **deliberately** takes the whole-screen path | The Database/Files strip is chrome composed OUTSIDE the canvas host, so adding/removing it is structural -- the same boundary `_replace_library_browse_canvas` enforces. Pinned by `test_open_item_by_id_notes_keeps_route_owned_source_strip` |
| `_open_library_media_viewer` (:18483) -- media row press and "Open in viewer" | `call_next(_apply_library_media_active_surface)` (canvas child: list -> loading/viewer) | Ownership guard skips when media no longer owns the route |
| `_recompose_library_media_detail_if_unrendered` (:13105) -- detail worker completion | `call_next(_apply_library_media_active_surface)` | Existing identity guard unchanged; TASK-15706 rule -- skips when the surface transitioned away, detail stays cached |
| `_refresh_library_media_detail` no-service branch (:13009) | `await _apply_library_media_active_surface()` | same seam |
| `_exit_library_media_viewer` (:30850) -- "‹ Back to list" and Escape | `call_next(_apply_library_media_list_return)` -> canvas swap viewer -> list, then the task-2856 AC1 entry-focus arm via `then=` | Arm ordered AFTER the mounted list's own sync recompose (see defect 1 below) |
| `action_library_media_viewer_back` edit / delete-confirm / analysis branches (:30886/:30890/:30894) | `_sync_library_media_viewer_or_recompose()` -- viewer-scoped recompose | Only the mounted viewer's children change; mouse capture released first (the viewer hosts `Input`/`TextArea`); unmounted-viewer fallback is whole-screen |
| skills Import open / cancel (:19059/:19074) | `_sync_library_canvas("skills", then=focus)` | Canvas-scoped; caret parked in the path field on open, back on the `Import…` opener on cancel (the 15457 stranded-focus rule) |
| prompts Import open / cancel (:21994/:22010) | `_sync_library_canvas("prompts", then=focus)` | ditto |
| `_open_library_export_canvas` (:13483) -- section "Export…" | `_apply_library_open_item_surface(LibraryExportCanvas)` | Rail selection + canvas swap; "Export…" from browse-Notes falls back at the strip boundary |
| `_apply_library_row_toggle` (:1433) | **unchanged -- no defect** | Probed on the mounted harness: a select-mode media row toggle already causes **0** whole-screen recomposes on this base. The `:1533` statement is the exception-fallback arm only, so the review's cite was a false positive |

### Supporting changes

- `_replace_library_canvas_child(rail_mode=...)`: `"selection"` uses `LibraryRail.apply_selection` (a two-row in-place patch) instead of the full rail recompose. A per-click open moves only the selection; entry reconciliation keeps the default full `sync_state` because its snapshot can change counts/sections. Worth 117 -> 29 widget constructions on a media open.
- `_replace_library_canvas_child(then=...)`: a follow-up ordered after the mounted owner's own post-mount sync recompose (see defect 1).
- `_sync_library_media_viewer_state`: equality-skip when every compose input is unchanged. The seam now also serves the open path, and an unconditional `viewer.refresh(recompose=True)` remounted a fresh `Markdown` body whose mount re-parses the whole document -- the double-parse task-15458 pinned.
- Whole-screen recompose is retained ONLY as (a) the `FAILED` fallback arm, (b) the unmounted-screen parity arm, (c) the Notes source-strip structural boundary. Entry-origin (deep-link) refreshes at :13007/:13087 are outside per-click scope and unchanged.

### Two defects found by the suites and fixed (not worked around)

1. **Follow-up ordering.** The viewer-back entry-focus arm ran from the exit continuation with no ordering against the mounted list's own sync recompose, so its `scroll_to` clamped against `max_scroll_y == 0` and the compact viewer-back scroll offset came back as 0. Fixed by threading `then=` through `_replace_library_canvas_child` into the canvas's `PostRecomposeCallback` -- the same ordering the 15457 conversion established. (`test_compact_media_viewer_back_restores_semantic_row_and_scroll`.)
2. **Builder-resolves-route-key supersede.** `_build_library_media_active_child` mirrors the RESOLVED `_selected_media_id` back onto the screen, and that field is part of the route key. Capturing the key before `build()` made the seam treat its own builder's resolution as a newer navigation and SUPERSEDE the projection -- viewer Back after the opened row had been deleted stranded the dead viewer on screen. Fixed by capturing the key after the builder runs. (`test_compact_media_viewer_back_falls_back_after_row_removed`.)

### One test retargeted (tests-stopped-measuring, not a regression)

`test_library_graduation_announcement_clears_on_direct_item_open` waited on `_library_media_view == "viewer"`. That is a transient the feature never promises: the onboarding app wires no media service, so the detail worker's unavailable-item fallback returns the view to `"list"`. Measured at **base** `30c7e1fe9`: the value immediately after the await was `"viewer"` and **every** subsequent poll was already `"list"` -- the assertion held only because the pre-conversion open had no await point after `run_worker` and returned inside that one-sample window. The test's subject is untouched: the announcement clear runs through `_acknowledge_library_destination_change` -> `_sync_library_lifecycle_status`, an in-place `Static.update` with no recompose dependency. Both trees end identically (announcement `""`, row `browse-media`), and the retargeted condition (`_library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA`) **passes at base as well**, so it measures the feature and not the change.

### Ratchet

`Tests/UI/test_library_recompose_ratchet.py` -- AST census of `self`/`screen` `refresh(recompose=True)` plus bare `recompose()` statements (blind to comments, docstrings, and canvas/viewer-scoped refreshes), pinned at **97**, with a failure message naming every sanctioned seam. Red-first proof at the shipped pin (`test-logs/ratchet-red-first.txt`): base file counts 107 > 97 -> RED; converted counts 97 -> green. A second test pins the counter against the counted and not-counted spellings so it cannot silently stop measuring its subject. It earned its keep during the task: it caught a fallback arm I added while fixing defect 2.

### Evidence (mounted `LibraryHarness`, `LIBRARY_TEST_SIZE`)

- Whole-screen recomposes per interaction: media row open (click + detail arrival) **1 -> 0**; viewer back **1 -> 0**.
- Widget constructions (load-independent): media open **117 -> 29** (-75%); viewer back **113 -> 47** (-58%).
- Click-to-settle, 12-press median, media row -> viewer detail rendered, base vs converted **back-to-back under identical machine load**: **270.1 ms -> 149.3 ms (-44.7%)**. Earlier quieter-machine pairs measured 240.7 -> 93.4/118.6 ms. No timing threshold is asserted in CI (the 15457 probe rule -- wall-clock is not stable CI evidence).
- Behavior pins: `Tests/UI/test_library_per_click_recompose_t21116.py` (9 tests) -- zero-recompose plus rail-identity for open / back / sub-state Escape / export / prompts+skills import, focus assertions on the import row, and the notes-open structural strip test.

### Test A/B (every red A/B'd against base `30c7e1fe9`; dev has since moved to `736359202` -- deliberately NOT rebased, all comparisons are against this branch's own base)

- Core battery (canvas sync, canvas-sync defects, selection updates, multiselect x3, screen navigation, media side-by-side, media trash, entry-compose-once, RAG handoffs, export x3, plus the two new files): **34 failed / 400 passed**; the 34 are byte-identical to the base set (15 + 19) -- `diff` of the sorted FAILED lists is empty. Zero new reds.
- `test_library_shell.py`: base **11 failed / 717 passed**; final tree **11 failed / 717 passed** -- the sorted FAILED lists are byte-identical. (Both of the divergences an intermediate run showed are resolved: one was defect 1 above, the other is the retargeted test.)
- Prompts + skills canvases: final tree **6 failed / 474 passed**. The first base run recorded 5/475, and the delta looked like two new reds in `test_library_prompt_canvas_receives_retained_pager_on_sync[size0,size1]` -- so both were A/B'd: each fails at base in ISOLATION, and a second full base run reproduces **6 failed / 474 passed** with the same six. The only remaining difference between the two 6-failure sets is which size-param of the separately flaky `test_library_prompt_pager_first_and_filter_failure_states` fails (`[size1]` at base, `[size0]` on the branch) -- it flips run to run on both trees. Nothing attributable to this change.
- `test_screen_navigation.py`: the two media-viewer-back tests were updated to the new contract (the plain-back test drives the captured `call_next` continuation and still asserts the exact task-2856 focus/timer sequence; the sub-state test passes unchanged through the documented fallback arm).
- Full `--collect-only` sweep: **56,590 tests collected, 5 collection errors** -- all five reproduce at base (`Tests/TTS/test_chatterbox_validation.py`, `Tests/UI/test_library_file_notes_workspace.py`, three under `Tests/Web_Scraping/Confluence/`). `Tests/Chat/test_fleet_teardown_notice.py` excluded per the standing >420 s hang.
- Ruff clean on every changed file.

### Deliberately not converted

The nine low-frequency admin flows named in the finding (delete / undo / receipt and friends), the media edit and analysis ENTRY buttons (only the Escape/Cancel EXITS were in the confirmed per-click list), the export RUN press, and the entry-origin deep-link refreshes. `_apply_library_row_toggle` needed no change (already 0 recomposes per click, measured).
## Review fix round

Independent review reproduced every number in the notes above (117->29 and
113->47 widget counts, 1->0 recomposes, the 107->97 census, both self-found
defect fixes, the `_apply_library_row_toggle` false-positive call, the
Notes-boundary claim, and the graduation retarget's honesty) and returned
FIX-FIRST on four Majors. All four are fixed at the mechanism, each RED-first.

### M1 -- the media-detail worker could clobber the Trash view (CONFIRMED)

`_build_library_media_active_child` had no `"trash"` branch at all, unlike
the two sibling builders (`compose_content` and
`_build_library_entry_active_child`) which both check it first. The worker
continuation `_recompose_library_media_detail_if_unrendered` guards only on
the rail row id, so a media-detail fetch resolving after the user pressed
"Trash" mounted the LIST canvas over the mounted `LibraryMediaTrashCanvas`
while `_library_media_view` stayed `"trash"` -- DOM/state divergence plus a
broken follow-on `_sync_library_canvas("media-trash")`. The supersede guard
structurally cannot catch this: the view is INSIDE the route key, so the key
stays self-consistent.

Fixed at the source of truth rather than at the call site: the builder grows
the branch, and `_build_library_entry_active_child` now DELEGATES instead of
duplicating the check -- that duplication is precisely what allowed the
shared builder to go missing one. The surface seam additionally early-returns
for trash, because Trash is a distinct surface with its own updater and
remounting it from a stale worker would drop its selection (the TASK-15706
rule). RED-first: with both guards mutated off, the integration test fails
on the exact symptom (`the media LIST canvas was mounted over the Trash
view`) and the builder unit test fails on the returned type.

### M2 -- the footer hint went stale on a viewer sub-state Escape (CONFIRMED)

`_sync_library_media_viewer_or_recompose` never called
`_register_footer_shortcuts()`; the retired whole-screen recompose did, via
`compose_content`. The sets genuinely differ --
`_library_route_shortcuts_for_current_state` branches on
`_library_media_viewer_substate_active()` -- so Escaping out of edit mode
left the footer advertising the sub-state's "back a step" when Escape had
already returned to the plain viewer, where it means "back to list".
Registered once in that method rather than in each of Escape's three
branches, so a future sub-state cannot be added without it. The test guards
itself against passing vacuously: it asserts the two sets differ before
asserting the restore.

### M3 -- the targeted swap raced the canvas-sync whole-screen fallback (CONFIRMED)

While a projection has the canvas host's child detached across its
remove/mount awaits, any `_sync_library_canvas` landing in that window
cannot find its canvas and took its `allow_screen_fallback=True` arm: a
whole-screen `refresh(recompose=True)`. That both re-introduced the exact
recompose this task removes AND raced a fresh same-id canvas into the
in-flight `mount` (`DuplicateIds` on `#library-media-canvas`, then a pilot
stall). Not confined to deleted rows: `_open_library_media_viewer` and
`_exit_library_media_viewer` each kick a list reload one line before
scheduling the swap, so any reload slower than the swap hits it -- the
shipped tests missed it only because their in-memory service always won the
race.

Fixed with a projection DEPTH counter (`_library_canvas_projection_depth`)
held across the swap via try/finally -- the region has eight early returns,
so the swap body was extracted into `_project_library_canvas_child` purely
so the caller can hold the marker across all of them. The repair loop holds
it too. Suppression is safe rather than lossy: the projection rebuilds the
destination from CURRENT screen state, so the state the suppressed sync
wanted is picked up by the projection; `then` is deliberately not run there
because the projection owns focus through its own ordered `then=`.

Two tests, both RED with the guard mutated off: a mechanism test carrying
its own CONTROL arm (same call with the marker cleared MUST still fall back
-- otherwise the treatment proves nothing), and a realistic one that injects
the delay INSIDE the guarded region (the host's `remove_children` await),
which is where the real race lives.

### M4 -- a routine snapshot dropped the projection's follow-up (SUSPECTED -> CONFIRMED)

Verified before fixing. `_library_entry_reconcile_is_current` compares
`_library_snapshot_state_generation` as well as the route key, and the
ordinary local-source snapshot apply bumps that generation -- one that
`_exit_library_media_viewer` itself kicks (via
`_load_library_media_list_if_needed`) a line before scheduling the swap. So
a routine snapshot landing mid-await returned SUPERSEDED and the seam
skipped `then()`, silently dropping BOTH the task-2856 AC1 first-row focus
and the media_return scroll restore that the pre-conversion code armed
unconditionally.

Fixed by distinguishing the two supersede kinds rather than by running the
follow-up unconditionally: a GENERATION-only supersede (route key unchanged)
still runs it, because the destination is still ours; a genuine ROUTE-change
supersede still skips it, preserving the TASK-15706 transitioning-surface
rule. The test asserts the route key did NOT change, so it cannot silently
degrade into testing the route-change path.

### Also taken

* **m3** -- `_repair_library_entry_canvas_owner` captured the route key
  BEFORE `_build_library_entry_active_child()`, the identical shape as the
  self-found defect 2 (the media arm mirrors the resolved
  `_selected_media_id`, a route-key field, back onto the screen). It
  self-healed in one iteration but only by spinning, and every `continue`
  path in that loop is await-free. The key is now captured after the builder
  and the loop is bounded at 8 passes as a hot-spin guard.
* **m5** -- the ratchet failure message now lists every site as
  `library_screen.py:<line>  <function>()  ->  <spelling>` instead of a bare
  count, on a 34.8k-line file. The counter also now matches the
  `self.screen.refresh(...)` spelling, and its remaining blind spots
  (aliased receiver, non-literal flag, `**kwargs`, getattr/partial indirect
  dispatch, `self.app.screen`) are both documented in the docstring AND
  pinned by their own test, so the documented list cannot drift from real
  behaviour.

### Not fixed (ledgered by the reviewer)

m1/m2/m4/m6/m7: arrival-note side effects, the latent notes-files arm, the
`follow_up` discard, the notes-canvas export fallback, and the armed-focus
re-request.

### Review-round verification

* Census unchanged at **97** -- none of these fixes adds a whole-screen
  recompose; the pin still holds.
* Widget-recreation probes re-run after all six changes: media open
  **0 recomposes / 29 widgets**, viewer back **0 / 47** -- identical to the
  pre-review numbers, so the fixes cost nothing.
* `Tests/UI/test_library_review_round_t21116.py` -- 6 tests, each RED-first
  against its own mechanism (mutation-based, Edit-restored).
* Core battery re-run (17 files): **34 failed / 408 passed** -- the 34 are
  byte-identical to the base `30c7e1fe9` pre-existing set (15 + 19); the
  passed count rose from 400 by exactly the 8 new tests. Zero new reds.
* Ruff clean on every changed file.
* **`scripts/check_persistent_diagnostic_inventory.py`**: RED on arrival with
  four drifted rows. Attributed against base `30c7e1fe9`: only
  `library_screen.py` (+5) is mine -- the other three
  (`chachanotes_fts_backfill.py` new, `ChaChaNotes_DB.py` +3, `app.py` +3)
  are untouched on this branch, i.e. pre-existing pin staleness from a
  sibling merge. All 12 added statements were read before regenerating. Mine
  are five `logger.debug` calls with static text plus the fixed internal
  `{kind}` vocabulary -- no user content, secret, or path. Of the absorbed
  foreign rows, worth flagging for the ledger: the `ChaChaNotes_DB.py`
  V46->V47 migration lines interpolate `self.db_path_str` (a local
  filesystem path) at INFO -- consistent with the 353 pre-existing calls in
  that file's migration logging, not introduced here, but absorbed into the
  pin by this regeneration. Pin regenerated with `--write`; gate now green
  (531 owners, 7265 TASK-494 calls).
<!-- SECTION:NOTES:END -->
