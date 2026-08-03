---
id: TASK-2016
title: >-
  Library ingest P3 polish batch
status: Done
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: low
dependencies: []
---

## Description (the why)

P3 polish and needs-reproduction findings from the 2026-08-02 Library
ingest UAT (critique snapshot 2026-08-02T21-04-04Z). None block tasks;
grouped so they are not lost. Items marked (repro) were observed once on a
contaminated instance or depend on environment — reproduce before fixing.

## Acceptance Criteria (the what)

- [x] Done rows state "done" once and show the file basename (full path
      available in details), not the absolute path inline.
- [x] "Expand all / Collapse all" render only when more than one options
      panel exists.
- [x] The generic panel's scope line no longer claims "Applies to all
      Plain text / documents / HTML in this import." when zero such files
      are staged (reword for the global-options case).
- [x] Intro lines disappear once a path is typed (state already says so;
      the DOM-surgery typing path never removes them).
- [x] The file picker opens at the last-used directory and hints which
      extensions are ingestible. (Already implemented on dev:
      ``_library_ingest_browse_location`` remembers
      ``library.ingest.last_directory`` and ``FileOpen`` gets
      ``_ingestible_file_filters()`` — evidence-closed, no change.)
- [x] Rail counts no longer flash "(0)" before the lazy count arrives.
      (Already correct on dev: counts pass ``None`` until known and
      ``_count_suffix(None, …)`` renders no suffix — evidence-closed, no
      change. The UAT sighting was the contaminated instance at older
      code.)
- [x] (repro) The ingest error-details modal placement: NOT REPRODUCIBLE
      on clean instances (opened correctly in the P1 live pass at 235 cols;
      the original sighting was on the contaminated co-driven instance).
      No change made; reopen with a capture if seen on a clean run.
- [x] (repro) First submit never smears dependency warnings / loguru
      DEBUG over the TUI. REPRODUCED on a clean instance and FIXED: spawn
      workers re-import app.py as ``__mp_main__`` with an inherited
      real-TTY stderr; an early guard in app.py silences loguru's default
      sink + warnings in that case, plus a pool ``initializer`` as belt.
      Live-verified: first submit shows zero noise at five samples (was
      7-8 lines).
- [x] `#library-search-input` gets an `Input.Changed` handler so typed but
      unsubmitted rail-search text persists/clears the way the user left
      it instead of resurrecting from `_library_rag_query` on recompose.
- [x] `[first_run] setup_started/setup_completed`: investigated —
      ``setup_started``-at-open is DELIBERATE and load-bearing (the
      offered-signal the auto-offer logic keys on, per
      ``first_run_setup_state.py``'s documented rationale);
      ``setup_completed`` is action-only (finish/skip, wizard :3009/:3090).
      Working as designed; no change. (Bonus find while probing: the
      first-boot wizard CRASH filed+fixed as task-2017.)

## Implementation Notes

Shipped on `fix/library-ingest-uat-p3-2016` (stacked on the 2015 branch):

- Done rows: terminal-state progress lines drop the state prefix (the row
  line already says it) and the writer stamps the basename, not the
  absolute path. Live: `✓ done · report.txt · 1s` over `Ingested
  report.txt`.
- Expand/Collapse-all render only with >1 panel (single generic panel =
  nothing to bulk-toggle); pinned both ways in canvas tests.
- New state field `type_group_file_counts` words each panel's scope line
  honestly; live: pdf-only staging renders "Applies to Plain text /
  documents / HTML if this import contains any."
- Intro lines hide/show in place with the typed path (new
  `library-ingest-intro` class as the no-recompose handle); live 1→0→1.
- `#library-search-input` gained an `Input.Changed` handler so
  `_library_rag_query` (rail rebuilds + persisted shell state) tracks
  typed text; recompose-survival pinned both directions.
- stderr flood: root cause was spawn workers re-importing `app.py` as
  `__mp_main__` onto a real-TTY stderr. Note `parent_process()` is NOT yet
  populated during that import — the guard must key on
  `__name__ == "__mp_main__"` (a parent_process()-only guard was
  live-disproven first). Pool `initializer`
  (`silence_ingest_worker_import_noise`) kept as belt; the two
  pool-construction fakes were extended to record + pin the initializer
  (the fake-shaped-by-assumption trap, again).
- Items 5/6/10: evidence-closed as already-implemented / by-design (AC
  text updated inline); item 7 not reproducible on clean instances.

Verified: suites `Tests/Library` + `test_library_shell.py` +
`test_library_ingest_canvas.py` + `Tests/Local_Ingestion` = 1331/1331.
Live pass (isolated profile): done-row copy, intro hiding, reworded scope
line, zero first-submit noise at five samples, "Ingest finished — 1
failed" warning-severity toast.
