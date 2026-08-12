---
id: TASK-15457
title: Library: convert per-click whole-screen recomposes to canvas-scoped sync
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
