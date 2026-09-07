---
id: TASK-2015
title: >-
  Library ingest P2 batch — feedback, copy, and consistency fixes
status: Done
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: medium
dependencies: []
---

## Description (the why)

Confirmed P2 findings from the 2026-08-02 Library ingest UAT (critique
snapshot
`.impeccable/critique/2026-08-02T21-04-04Z__chatbook-widgets-library-library-ingest-canvas-py.md`).
Each is independently small; batched here so the sweep ships as one
coherent polish pass. Split out sub-tasks if any item grows.

## Acceptance Criteria (the what)

- [x] Path validation feedback appears while the field is still focused
      (debounced while typing, or on a short dwell) rather than only on
      blur.
- [x] Completing an ingest produces an above-the-fold success signal (e.g.
      a toast, or the queue summary + an auto-scroll/pinned result), and
      the per-file result row with "Open in Library" is reachable without
      hunting.
- [x] An empty-file failure is classified permanent: no Retry is offered
      for it (matching the unsupported-type rule), or Retry is replaced by
      an explanation.
- [x] Folder-expanded done rows resolve their media ids and carry "Open in
      Library" like single-file rows.
- [x] Pre-flight copy pluralizes correctly ("will be recorded as
      failures", "as a failure" singular).
- [x] Failure copy is unwrapped: at most one "Failed to …" prefix, with
      the underlying reason and a suggested next step.
- [x] "Clear finished" asks for confirmation and/or no longer wipes the
      "Recent ingests" list (recent list survives clearing the queue).
- [x] At 110 columns the options summary header truncates with an
      ellipsis instead of a hard mid-word clip.
- [x] A failed submit for a nonexistent path does not leave a stray
      "Choose a file…" button + "0 files" line stacked under the error.
- [x] Start ingest is disabled (with the gate line explaining why) when
      pre-flight guarantees every staged file will fail.
- [x] Elapsed time on done rows reflects what the user actually waited
      (includes pre-run latency; never "0s" for a watched multi-second
      job).
- [x] Jobs report a parsing/writing progress indication for large files
      (even a coarse per-stage state is fine; no silent jump from queued
      to done).

## Implementation Plan

Executed per `Docs/superpowers/plans/2026-08-02-library-ingest-p2-batch.md`
(surveyed anchors per item, TDD per item, grouped commits).

## Implementation Notes

All 12 ACs shipped on `fix/library-ingest-uat-p2-2015`:

- **State layer** (`library_ingest_state.py`): elapsed measures from
  `submitted_at` (sub-second renders `<1s`, no-timestamp drops the
  segment); nested `Failed to X file:` wrappers collapse to one prefix in
  `short_ingest_error` (shared by queue row + Home); estimate/breakdown
  suppressed under errors; Start disabled + explaining gate line when
  pre-flight finds zero supported files; two-press label
  `queue_clear_finished_label` derived here. Per-stage parsing/writing
  rows were already pinned by existing row-format tests (item 12).
- **Pipeline**: zero-byte files raise `PermanentIngestError` (dead-bait
  Retry gone); fresh-folder done rows proven to carry media ids (the UAT
  sighting was the task-2013 duplicate case).
- **Screen** (`library_screen.py`): 0.8s typing debounce triggers
  pre-flight (trigger/apply now use the context-preserving refresh so a
  result landing mid-word cannot steal focus); registry listener posts
  one "Ingest finished — N imported · M failed" toast when active jobs
  settle (baseline captured when the queue went active); "Clear finished"
  arms on first press and any registry mutation disarms.
- **Canvas/CSS**: plural copy "recorded as failures"; options-title
  ellipsis at narrow widths (CollapsibleTitle needed width:1fr +
  height:1 + nowrap for text-overflow to engage — stock width:auto lets
  the parent clip first).

Verified: every fix red→green TDD'd; full `Tests/Library` +
`test_library_shell.py` + canvas + `Tests/Local_Ingestion` green (one
timing-dependent first run of the new two-press test hardened with poll
loops per the task-699 state-then-DOM lesson; full shell file then
274/274). Live spot-check on an isolated profile: while-typing "Path not
found" at ~1.6s dwell without blur; "Ingest finished — 1 imported" toast;
`✓ done · report.txt · <1s` row; ellipsis at 110 cols.
