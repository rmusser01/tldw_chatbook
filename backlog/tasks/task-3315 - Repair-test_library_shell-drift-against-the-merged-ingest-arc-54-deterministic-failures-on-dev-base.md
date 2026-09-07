---
id: TASK-3315
title: >-
  Repair test_library_shell drift against the merged ingest arc (54
  deterministic failures on dev base)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-08 21:30'
updated_date: '2026-08-09 18:34'
labels:
  - library
  - tests
  - dev-baseline
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during follow-up batch phase A (2026-08-08): `Tests/UI/test_library_shell.py` carries 54 deterministic failures on dev base `ebeae1440` (identical set across loaded and quiet runs, reproduced with the phase's product diff fully reverted). Two mechanisms: (a) `_LibraryIngestCanvasHarness` does not mirror `TldwCli._ingest_local_stt_jobs`, which `app.py._maybe_start_next_ingest_job` reads since the ingest arc (PR #1452) — the real app initializes it (`app.py:5660`), so this is stale-harness drift killing ~20 job-lifecycle tests with AttributeError; (b) a Notes 60x20 geometry off-by-one family (`shell.region.height 14 != 15`) plus dependent pilot tests — cause undiagnosed, could be arc CSS or dev's own churn; diagnose before pinning. The arc's phase batteries ran this suite only under `-k` filters, so the full suite's rot shipped unnoticed — the "suite no gate runs" lesson shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Full `Tests/UI/test_library_shell.py` runs green (or its residual failures are proven pre-arc with SHAs) with a READ pass count
- [x] #2 The geometry off-by-one family's cause is named (arc CSS vs dev churn vs stale pin) before any expectation is updated
- [x] #3 The harness mirrors the app attributes the ingest job loop reads, derived from the real initializer rather than hand-listed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both mechanisms on the current worktree (harness AttributeError + one geometry failure) and inventory the full failing set with a background full-file run.
2. Mechanism (a), TDD: write an AST-derivation guard FIRST (mixin state-reads → required harness attributes) and watch it go RED naming the missing attributes; then extract the app's inline ingest-runtime-state block into `LibraryIngestQueueMixin._init_library_ingest_runtime_state()` (behavior-identical move), call it from `TldwCli._wire_study_services` and from both headless harnesses (`_LibraryIngestCanvasHarness`, `_IngestRunnerHarness`); mutation-check the guard in three directions (seam attr removed, harness seam call removed, new unmirrored read added to app.py).
3. Mechanism (b): name the cause BEFORE touching pins — read-only `git log/diff/show` blame plus empirical bisect via `git archive` extracted trees of `78e4e2c9c` (P2 merge), `6b4ccf475` (notes-adaptive PR #1439 merge), `ebeae1440` (dev base), running the 60x20 family in each against the same venv.
4. Re-pin the 60x20 family to the measured settled truth (probe dump of every state), with the cause chain named in `_assert_task8_compact_chrome`'s docstring and a determinism guard (strip settle-wait) in the navigator helper; file the underlying product inconsistency as its own task (3317).
5. Repair any remaining phase-A–D drift the full run surfaces; final evidence = full-file run(s) with READ counts; keep the phase-C/D consolidated ingest battery green (task-3313's 16-file list).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** Full `Tests/UI/test_library_shell.py` is green: FINAL runs 280 passed
(tests 1–280) + 265 passed (tests 281–545) — two disjoint halves covering all 545
collected tests, zero residual failures (the known pre-arc coin-flip
`note_save_result_after_switch_is_discarded` also passed). One earlier
whole-file pass on the worktree also completed 488 passed / 56 failed pre-fix,
establishing the baseline inventory. Kept green: consolidated ingest battery 624
passed (17 files), `test_library_ingest_runner` 98, `test_library_notes_session`
41, `test_destination_shells` 104 + 1 skip.

**Mechanism (a) — harness drift (24 ingest pilots).** Fix shape: extracted the
app's inline ingest-runtime-state block into
`LibraryIngestQueueMixin._init_library_ingest_runtime_state()` (verbatim move,
behavior identical), called by `TldwCli._wire_study_services`,
`_LibraryIngestCanvasHarness` (this file) and `_IngestRunnerHarness`
(`Tests/Library/test_library_ingest_runner.py`). Guard:
`test_ingest_canvas_harness_mirrors_every_mixin_state_read` AST-derives every
`self._ingest_*`/`_local_stt_*`/`library_ingest_jobs` LOAD (plain + `getattr`)
in the mixin, minus its own methods, and asserts (1) a fresh harness carries
them all, (2) the seam sets every one. RED first (named the six missing attrs
incl. `_ingest_local_stt_jobs`); mutation-checked in three directions (seam
attr removed → RED naming it; harness seam call removed → RED naming all ten;
new unmirrored `self._ingest_new_lane_jobs` read added to app.py → RED naming
it).

**Mechanism (b) — 60x20 geometry (14 tests + 5 relatives).** Cause NAMED before
any pin change, proven empirically by running the family against `git archive`
trees: the identical failure set exists at `6b4ccf475` (the notes-adaptive PR
#1439 merge — the very commit that brought these tests to dev) and at dev base
`ebeae1440`; born broken at merge, media-ingest arc exonerated. Two causes:
(1) `#library-notes-database-purpose` (LIB-19, task-2858 `a3591b503`, PR #1420)
wraps to 3 rows + 1 margin at width 60 → list 10→6; (2) `#library-notes-source-strip`
(file-notes workspace `b83852eda`) mounts on any FULL screen recompose with
notes selected, but PR #1439's own fast path (`_replace_library_browse_canvas`)
skips it on plain rail entry → editor/sync/loading routes settle at shell 14,
plain list at 15. Pins re-measured and updated with the cause chain in
`_assert_task8_compact_chrome`'s docstring (`source_strip` param documents the
per-route fork); navigator helper got a strip settle-wait; the product
inconsistency itself is filed as task-3317.

**Product regressions found and FIXED (all reproduce at dev base, pre-phases):**
1. `action_library_notes_escape` called `_back_from_library_note_editor()` — a
   method that never existed in ANY commit (dangling call from `e453e9099`) —
   Esc in the note editor raised AttributeError. Now routes through the shared
   guarded Back seam `_exit_library_note_editor_guarded()`.
2. LIB-14 untouched-blank GC was dead: the coordinator refactor (`13cf08f90`)
   made `_read_library_note_editor_fields` project the snapshot (title =
   seeded "Untitled") instead of the presented-empty widget. The GC gate now
   treats the seed title as blank (`LIBRARY_NOTE_BLANK_SEED_TITLE`, shared with
   the create seam) and never starts a second destructive op while one is
   running/admitted.
3. The empty-title → "Untitled" save fallback (task-2858's reviewed decision)
   was dropped by the same refactor; restored at the coordinator port save.
4. The screen-wide `/` rail-search grab in `on_key` ran before bindings, so the
   notes-scoped `/` (focus filter) could never fire; `on_key` now defers to
   `check_action("library_notes_focus_filter")` — precedence lives in one place.

**Test-side re-pins to shipped truth (cause named in each comment):** the 60x20
family + `compact_surplus` (3) + `fifty_same_side_resize`;
`rail_counts_never_clip` (full "Conversations (2)" fits at 100x30 after
task-2858 width fixes); sync-conflict "ask" probe scoped to conflict-choice
labels (rail gloss "Prompts — AI asks" collided with the whole-screen sweep);
`sync_now` waits for the DOM-settled done status (was racing the finish-of-run
recompose); footer tests pin the contextual segment before task-3302's global
" | F1 help …" suffix, and the Notes-exit assertion pins the rail-stage
guidance.

**Files.** `tldw_chatbook/app.py` (seam), `tldw_chatbook/UI/Screens/
library_screen.py` (4 product fixes + seed constant), `Tests/UI/
test_library_shell.py` (guard + harness + re-pins), `Tests/Library/
test_library_ingest_runner.py` (harness via seam), `Docs/User_Guide/library/
notes.md` (re-verified stamp), `backlog/docs/lessons-testing-evidence.md`
(git-archive bisect lesson), task-3317 filed.
xhigh review + live-verify round (2026-08-09), P0 SILENT DATA LOSS in the restored LIB-14 GC:
the untouched-blank gate treated the literal string "Untitled" as blank
(`raw_title == LIBRARY_NOTE_BLANK_SEED_TITLE`), so a note the user DELIBERATELY titled "Untitled"
with an empty body was destroyed on navigate-away -- no prompt, no undo. A string can never
distinguish the create seam's seed from the same letters typed by a human; only provenance can. New
screen-session marker `_library_note_title_user_edited` is set by `handle_library_note_title_changed`
BEFORE and INDEPENDENTLY of the mutate result (the two guards above it already exclude every
programmatic echo, and a blank-seeded note renders its title placeholder-only with the draft still
holding the seed, so pasting "Untitled" is a real touch that `mutate()` reports as a no-op -- gating
the marker on mutate() left the GC free to destroy the note). It is cleared only when a NEW editor
session begins (create / row switch / deep link / full editor reset), never by a save, so the
distinction survives a save round-trip. An emptied-out title is still blank either way, so the
type-then-delete-everything case still GCs.
Same round, the paired save-seam disagreement: the port substituted the seed title for a blank one
on the wire but `DatabaseNotePortSaveReply` carries no title back, so the snapshot baseline kept the
blank while the DB row was named "Untitled" and every list row patched from that snapshot inherited
it (observed: list title '' vs persisted 'Untitled'). The substitution is a pure function of the
payload title, so both sides now derive it from one helper, `library_note_persisted_title` --
the port before the write, `_patch_library_note_list_from_session` after it.
Tests in Tests/UI/test_library_shell.py (real DB): the deliberate-"Untitled" note survives; an
untouched seed still GCs after a body round-trip; the list row agrees with the persisted name.
Mutation check (revert the gate to plain string equality) sends the first RED.

### Addendum — round 2 after the rebase onto dev `f6911b37b` (2026-08-09)

Test-side only; no product change. The rebase (201 dev commits since base
`ebeae1440`) left three notes-half failures. All three were settled by the same
method AC#2 demands — extract dev `f6911b37b`'s product tree with `git archive`
into a scratch dir, drop THIS branch's `Tests/UI/test_library_shell.py` into it,
and run the failing cases there (import isolation via a `sitecustomize.py` that
repoints the venv's editable-install finder at the extracted tree, since the
`.pth` finder sits on `sys.meta_path` and no `PYTHONPATH`/cwd trick outranks it).

1. `test_library_note_60x20_navigator_state_allocation[normal]` and
   `test_library_note_compact_surplus_allocation_expands_only_named_owner[navigator]`
   — **dev's change, stale pin.** Both fail identically against the extracted dev
   product tree. Cause: dev `d1df7d0a7` (TASK-13213, "restore file notes source
   access") gates the targeted canvas swap on matching contextual chrome
   (`notes_source_strip_mounted != (shell.canvas_kind == "notes")` → refuse), so
   the plain rail press into the notes list now lands via the full recompose and
   carries the 1-row Database|Files strip. Re-pinned: `normal` gets
   `source_strip=True` and list 6→5; the navigator surplus owner 11→10 at 80x24
   and 17→16 at 100x30. This also **closes task-3317 AC#1** — the per-route
   asymmetry that AC recorded is gone; `source_strip=False` now survives only for
   the create-note canvas (`canvas_kind "notes-create"`, which never composes the
   strip).
2. `test_library_shell_blank_note_escape_key_returns_to_list_without_crash`
   — **ours**, and the inverse direction: this test arrived from dev
   (`dd30c24e5`), where it pins `count_notes(...) == 1` with the comment "row
   survival must match the Back button's own (separately tracked) GC bug". That
   bug is the one THIS task fixed in `794ee4be5` (seed-aware `title_blank`), so on
   this branch the row is correctly GC'd. Proof of direction: against the
   extracted dev tree the Back-button GC test fails ("never GC'd") and the Escape
   test's `== 1` passes — the exact inverse of this branch. Re-pinned to the fixed
   behavior (GC parity with Back) and de-raced: dev's fixed `pilot.pause(0.2)`
   before a worker-owned delete is why it failed 8/8 alone but survived some
   loaded full-file runs. Now polls like the Back test does. Mutation check:
   `title_blank`'s seed clause disabled → both the Escape test and
   `..._untouched_is_gc_from_real_db_on_back` go RED; Edit-restored, product diff
   vs HEAD empty.
<!-- SECTION:NOTES:END -->
