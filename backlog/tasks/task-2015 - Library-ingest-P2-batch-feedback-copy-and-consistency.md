---
id: TASK-2015
title: >-
  Library ingest P2 batch — feedback, copy, and consistency fixes
status: To Do
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

- [ ] Path validation feedback appears while the field is still focused
      (debounced while typing, or on a short dwell) rather than only on
      blur.
- [ ] Completing an ingest produces an above-the-fold success signal (e.g.
      a toast, or the queue summary + an auto-scroll/pinned result), and
      the per-file result row with "Open in Library" is reachable without
      hunting.
- [ ] An empty-file failure is classified permanent: no Retry is offered
      for it (matching the unsupported-type rule), or Retry is replaced by
      an explanation.
- [ ] Folder-expanded done rows resolve their media ids and carry "Open in
      Library" like single-file rows.
- [ ] Pre-flight copy pluralizes correctly ("will be recorded as
      failures", "as a failure" singular).
- [ ] Failure copy is unwrapped: at most one "Failed to …" prefix, with
      the underlying reason and a suggested next step.
- [ ] "Clear finished" asks for confirmation and/or no longer wipes the
      "Recent ingests" list (recent list survives clearing the queue).
- [ ] At 110 columns the options summary header truncates with an
      ellipsis instead of a hard mid-word clip.
- [ ] A failed submit for a nonexistent path does not leave a stray
      "Choose a file…" button + "0 files" line stacked under the error.
- [ ] Start ingest is disabled (with the gate line explaining why) when
      pre-flight guarantees every staged file will fail.
- [ ] Elapsed time on done rows reflects what the user actually waited
      (includes pre-run latency; never "0s" for a watched multi-second
      job).
- [ ] Jobs report a parsing/writing progress indication for large files
      (even a coarse per-stage state is fine; no silent jump from queued
      to done).
