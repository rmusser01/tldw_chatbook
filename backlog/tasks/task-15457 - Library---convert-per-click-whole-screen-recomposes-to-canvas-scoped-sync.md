---
id: TASK-15457
title: 'Library: convert per-click whole-screen recomposes to canvas-scoped sync'
status: Done
assignee:
  - codex
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `UI/Screens/library_screen.py` carries 147 statement-level `refresh(recompose=True)` / `await recompose()` sites — regrown past the July task-281 fix (124 sites then). Per-click confirmed: notes row click (`:21914`), select toggle/all/clear (`:21926-:21967`), notes sort/filter strip (`:13236-:13287`), skills row (`:14087`), prompts row (`:16638`), media-type chooser (`:12328/:12358`), ingest checkbox toggles (`:20384`), RAG mode/scope toggles (`:24316/:24342`), choice-strip close (`:18160`), notes-sync toggles (`:19582-:19626`) — each tearing down and remounting the whole 26k-line screen (~120-200 widgets + CSS apply + relayout). Cascades stack: "back to list" = 3 recomposes (documented at `:7580-7597`); note creation = 6 refresh sites; a one-line import status change is a full recompose (`:13732/:16347`). The blocker for the Notes canvas is documented at `:21925`: a `LibraryNotesCanvas` constructor param shadows the `sync_state` hook name, so it was deliberately left un-converted in task-252.

Fix direction: rename the shadowing param to unblock the targeted hook; extend the existing `_sync_library_canvas` (`:1281`) pattern; convert per-click sites to canvas-scoped sync, keeping whole-screen recompose only for true canvas transitions. Stability constraints: the recompose-lifecycle traps from the July programme apply (fresh widgets carry current state — per-widget caches must live on the widget instance, not screen dicts); convert in reviewed slices with the existing recompose-discipline patchers (`_apply_library_row_toggle` `:1188` is the model), each slice behavior-pinned before conversion. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 None of the confirmed per-click interactions (notes row/toggles, sort/filter strips, media-type chooser, ingest toggles, RAG toggles, status lines) triggers a whole-screen recompose (evidence per site class)
- [x] #2 The LibraryNotesCanvas targeted hook is unblocked and used; behavior of every converted surface unchanged (tests)
- [x] #3 Remaining recompose site count re-measured and recorded; per-click latency before/after on the notes canvas
<!-- AC:END -->

## Implementation Plan

1. Record the current statement-level screen-recompose count and a mounted Library Notes per-click baseline, then add characterization tests for each confirmed interaction class.
2. Rename the Notes sync-panel constructor field that shadows the canvas update method, add a canvas-owned state update hook, and extend `_sync_library_canvas` to build and route Notes state without remounting the Library screen.
3. Convert the confirmed Media, Notes, Prompts, Skills, Ingest, Search/RAG, choice-strip, notes-sync, and status-line handlers to the narrowest mounted canvas or child-region update that preserves existing controls, focus behavior, and state ownership.
4. Run focused mounted tests after each slice, record the remaining recompose count and identical before/after Notes latency probe, then run the full relevant Library UI suite serially.
5. Perform a self-review and isolated live UAT at the supported terminal sizes; document implementation evidence, deviations, and any generalizable testing lesson.

ADR required: no
ADR path: N/A
Reason: this task applies the existing Library screen/rail/canvas ownership and targeted-update pattern without changing storage, service contracts, security boundaries, dependencies, or long-lived application structure.

## Implementation Notes

- Extended the existing Library canvas-sync seam to Notes, Prompts, Skills, Ingest, Search/RAG, and Export, with complete screen-owned snapshots and canvas-local recomposition. Renamed the Notes sync-panel field so `LibraryNotesCanvas.sync_state()` is callable, and moved Notes/Prompt/Skill loading views into their owning canvases so list-to-editor loads preserve shell, rail, and canvas identity.
- Converted the confirmed per-click Notes row/select/sort/filter/sync paths, Prompt and Skill rows/sort/filter paths, Media type chooser, Ingest structural options, Search/RAG toggles, choice-strip close, and import status receipts. Status receipts patch their mounted `Static` in place; asynchronous receipts arriving after navigation are retained without recomposing an unrelated surface.
- Added mounted identity and timing coverage in `Tests/UI/test_library_canvas_scoped_sync.py`, including real Prompt/Skill service-backed transitions, Notes sync and row transitions, Media/RAG interactions, Ingest routing, status updates, Unicode loading copy, and shell-recompose spies. Existing Prompt, Skill, Notes create/filter/sync, Media filter, and Search/RAG regressions also pass after rebasing onto `origin/dev` at `ab42f0831`.
- Evidence: the current pre-change base contained 143 statement-level whole-screen recompose sites and the final implementation contains 102 (41 removed). The identical 12-click Notes select-toggle probe improved from a 632.199 ms median to 123.315 ms after the final rebase (80.5%). Mounted UAT passed at 160×45 and 235×52.
- Verification: changed files compile; Ruff reports no new findings when the seven pre-existing `E721` findings in `library_screen.py` are excluded; direct isolated runs passed all new tests plus the focused existing regressions listed above. Normal pytest collection remains unavailable in this Windows sandbox because the repository's network guard blocks Textual's internal localhost socketpair, so async test bodies were run directly under fully isolated config/data/temp paths as prescribed by `lessons-testing-evidence.md`.
- ADR required: no. This applies the existing canvas ownership/update contract and introduces no new architectural boundary. No new lessons entry was warranted; the review-found Unicode-copy and off-surface asynchronous receipt issues are directly regression-pinned in the task test file.

## Rebase reconciliation (parallel implementation)

A second session implemented this task independently on branch
`task/15457-library-recompose` while the codex implementation above landed on
dev as `976dbafcb`. Reconciled by semantic rebase onto `origin/dev` at `74c8cf7043`, reading each
dev commit before resolving. Outcome:

**Superseded by `976dbafcb` — dropped, no hunk kept.** The conversions
themselves: notes select strip, notes sort/filter strips, notes-sync toggles,
media type chooser, ingest option toggles, Search/RAG mode+scope toggles, and
the prompts/skills import-status receipts. `976dbafcb` converted the same
sites and went further (Prompts, Skills, Export, and the list→editor loading
views moved into their owning canvases; 143→102 sites vs this branch's
152→132). The parallel branch's `LibraryNotesCanvasState` dataclass is also
superseded: dev's keyword-argument `sync_state` covers the same ground, and
its `sync_panel_state` rename resolves the same constructor shadowing the task
was blocked on. Minor 7 (the skills-import stale-screen guard gap) is
superseded too — dev already added the guard.

**Merged — dev's site, this branch's fix.** Four defects present in
`976dbafcb`'s own implementation, each verified RED against dev before the
port and green after:

1. `_sync_library_canvas`'s media branch never mirrored the RESOLVED selection
   back into `_selected_media_id` (`compose_content` and
   `_replace_library_browse_canvas` both do). Filtering the selected item out
   left the canvas highlighting one row while "Open in viewer" opened another.
2. Focus escaped the canvas on every converted site. `call_after_refresh` has
   no ordering against a recompose driven by the canvas's own message pump, so
   dev's nine focus follow-ups ran against removed children; measured landing
   spot `console-rail-section-toggle-library-details`. Fixed by the new
   `PostRecomposeCallback` mixin plus `_sync_library_canvas(..., then=...)`,
   with a default follow-up that restores the portable Notes focus identity the
   whole-screen seam restores for itself.
3. The Notes list's scroll offset was dropped (12→0), visible only below
   `LIBRARY_NOTES_COMPACT_BREAKPOINT`. Restored on the same follow-up, but
   deferred via `call_after_refresh` — run inline the new children are not laid
   out, max scroll is still 0 and `scroll_to` clamps the offset away.
4. **Not a reproduced defect — hardening that was probed and then REMOVED.**
   A footer re-derivation was added at the sync choke point on the theory that
   neither footer tier survives a targeted sync. It does not reproduce red on
   dev. Probed by disabling both branches: the Notes select-mode footer and the
   media type-strip footer both stayed current, because dev's
   `LibraryScreen.refresh` calls `_apply_library_notes_footer_context()` on
   EVERY refresh (not only `recompose=True`), and every footer-flipping sync
   has a focus follow-up whose `set_focus`/`scroll_visible` triggers one — the
   call chain was traced (`refresh <- _restore_library_notes_focus_identity <-
   _restore_library_notes_after_targeted_sync`). The branch is gone rather than
   kept unfalsifiable; the coupling it would have guarded is listed under
   Residuals below. The earlier claim that this reproduced red was wrong: the
   test written for it passes with the mechanism disabled, i.e. it was never
   discriminating.

Also ported: `PostRecomposeCallback` skips its callback when
`Widget.recompose` early-returns on a detached/pruning widget, and
`LibraryNotesCanvas` re-runs `on_mount`'s post-compose wiring through an
`_after_recompose()` hook ordered BEFORE the follow-up (dev's `sync_state`
recomposed without re-running it at all).

Evidence lives in `Tests/UI/test_library_canvas_sync_defects.py` (7 tests),
kept separate from dev's `test_library_canvas_scoped_sync.py` rather than
merged into it: the two files pin different things — that the conversions
happen, and that they do not lose selection, focus, scroll, or footer.

### Reconciliation: residuals, counting method, and what is actually closed

**Focus is closed, including the transition.** The first pass of this
reconciliation left the notes row→editor path still stranding focus outside the
canvas, and the task file read as if focus were done. It is now genuinely
closed, by three changes rather than one, because that path is a DOUBLE canvas
sync (list → loading → editor):

* `PostRecomposeCallback` re-queues its follow-up when `_recompose_required` is
  re-armed during a recompose — Textual's own signal that a second `sync_state`
  landed while this one was still awaiting `mount_all`. Without it the
  follow-up ran against the loading children (traced: `still_queued=True` on the
  loading pass, fired on the editor pass).
* the default notes focus restore is installed only for IN-SURFACE syncs
  (mounted mode == synced mode). On a transition the capture is worthless: the
  handlers flip the notes view *before* calling the sync, so
  `_capture_library_notes_focus_identity` already reports the destination region
  with an empty semantic role, and restoring it resolved to a fallback target —
  measured landing spot, the rail row.
* every remaining `call_after_refresh` follow-up on a canvas sync is now a
  `then=`: the notes editor arm + editor identity, and the skills and prompts
  `_arm_*` calls. Arming is dirty-tracking, not cosmetics — a lost follow-up
  leaves the editor unarmed. **Zero `call_after_refresh` follow-ups remain on a
  canvas sync.**

**Residual (recorded, not closed): footer honesty is coupled to focus.** Both
footer tiers stay current only because a focus follow-up triggers a screen
refresh and dev's `refresh` override re-derives the footer unconditionally. A
future footer-flipping sync with no follow-up would go stale silently, and no
test would catch it — the invariant test in
`test_library_canvas_sync_defects.py` is a regression net, not a discriminating
probe (it passes with the mechanism disabled, which is why the explicit branch
was removed rather than kept).

**Counting method.** This branch reports 122 whole-screen recompose sites in
`library_screen.py`, counting `self.refresh(recompose=True)` +
`await self.recompose()` + the module-level helpers' `screen.refresh(...)`
fallbacks, on non-comment lines. Dev's notes report 102; that number is **not
reproducible with this measure** and the two should not be compared — the
difference is method, not regression. Whoever cares about the headline should
pick one script and re-run it over both trees.

**Also carried from the parallel branch:** `RecomposeCaptureGuard` was added to
`LibraryNotesCanvas`, which dev's version does not carry — a canvas that
recomposes itself bypasses `BaseAppScreen.refresh`'s mouse-capture release, and
this canvas mounts `Input`/`TextArea` children where a stale capture is exactly
the app-wide click-dispatch failure that guard exists for.
