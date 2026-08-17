---
id: TASK-15810
title: >-
  Library RAG Answer's first query on a fresh profile never returns — CPU-bound,
  reproduced twice
status: Done
assignee:
  - '@codex'
created_date: '2026-08-13 20:28'
labels:
  - rag
  - library
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of TASK-15400 (2026-08-12) and again of TASK-15700 (2026-08-13) both stalled at the same place: the FIRST Library RAG Answer run on a freshly-created profile sits on 'searching · <source>…' at ~98% CPU and does not produce an Evidence row. 15400 recorded 4+ minutes; 15700's run was left for 8+ minutes on a 36-note library (36 real User Guide pages written through add_note and indexed through index_entries, embedding model already on disk, HF_HUB_OFFLINE=1 so no download is possible) and still had not rendered a row. A 3-second macOS 'sample' of the process shows the hot stack entirely inside the CPython interpreter (coroutine step -> filter -> set membership -> id()/PySys_Audit), i.e. CPU-bound Python rather than blocked I/O; 'sample' is C-level only, so the Python frame could not be identified from it and the cause is NOT yet attributed. The user-visible effect is that the app's headline retrieval surface appears to hang on first use. Both arcs' live checks had to fall back to driving the engine directly, so this also blocks live verification of any retrieval change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The first RAG Answer run on a fresh profile renders Evidence rows in a bounded, stated time (or discloses progress honestly if a one-time warm-up is unavoidable)
- [x] #2 The spin is attributed to a named Python frame with a profile (py-spy/cProfile), not to 'the embedding stack' by assumption — the 2026-08-13 sample contradicts that attribution
- [x] #3 A live retrieval verification on a scratch profile can complete end-to-end through the UI, so a future arc's live check does not have to fall back to the engine
<!-- AC:END -->

## Implementation Plan

1. Reproduce the cold first query with the design's exact 36-note fixture and
   capture a Python-level profile before changing production code.
2. Record the named hot frame and callers, then amend the implementation plan
   with the exact production owner, deterministic RED regression, and minimal
   correction; independently re-review that amendment.
3. Implement only the measured correction with TDD, including concrete-runtime,
   event-loop heartbeat, stale-result, and non-overlapping cancellation coverage.
4. Run the focused RAG/Library/Pilot battery, Ruff, and diff checks.
5. Run the unprofiled cold-query and separate responsiveness checks through the
   real TUI, prove scratch-profile isolation, update documentation/evidence,
   self-review, and close the task only when every acceptance criterion passes.

ADR required: no

ADR path: N/A; ADR-003 and ADR-005 already govern the existing Library and
shared-local-RAG ownership boundaries.

Reason: this is a measured performance bug fix inside existing boundaries. If
profiling requires a new runtime, storage, or service-contract boundary, stop
and revisit the ADR decision before implementation.

Detailed plan: `Docs/superpowers/plans/2026-08-13-task-15810-library-rag-first-query.md`

## Implementation Notes

Diagnosed and fixed across two concurrent sessions; shipped as **PR #1640**
(merged 2026-08-15, dev merge `f542f0c2a`).

**Cause (profiled, not guessed — AC#2):** an input-mirror feedback loop
between the rail search box and the canvas RAG box. The sibling-patch
helper assigned the other widget's value; the assignment fired
`Input.Changed`; the handler re-entered and mirrored back. cProfile over
15.003s of the stall: 525 rail-handler / 1,556 canvas-handler / 1,043
sibling-patch calls, 1,568 panel refreshes, 692,228 selector checks
(`Docs/superpowers/qa/2026-08-13-task-15810.../profile-report.md`). The
15400-era "embedding stack" attribution was refuted twice over: an
independent reproduction showed the event loop was NEVER blocked (Ctrl-P
rendered the command palette 11 minutes in at 100.1% CPU) and the same
coroutine headless returns in 5.43s vs 14m25s under the TUI — retrieval
was never the cost; the TUI was the multiplier, which is also why both
prior arcs' engine-level A/B fallbacks always completed.

**Fix:** `with sibling.prevent(Input.Changed):` around the mirror write
(the TASK-15740/15673 pattern — the value-equality guard could not stop
already-queued events), an app-lifetime `asyncio.Lock` admitting one
Library retrieval at a time, and normalisation of a snapshotted transient
`searching` status on screen restore.

**AC#1 + AC#3 evidence (live, post-merge, 2026-08-15):** on merged dev, a
fresh 36-note/282-chunk scratch profile driven end-to-end through the real
RAG Answer UI: first Evidence row at **t=5.08s** ("Evidence · top 15",
15 results), CPU peaking at 20% — vs 14m25s at 90-99% CPU with no row
pre-fix. The reproduction artifacts are committed
(`Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/`) with the
headless control `Tests/Library/test_rag_answer_first_query_latency.py`.

**Residue:** Qodo finding 5 (the shielded-cancellation drain loop never
uncancels; theoretical hot-spin under repeated cancel) deliberately not
patched on the snapshot — filed as task-16450. Lesson filed in
`lessons-backlog-hygiene.md`: a task's status is not a lock — check
`.worktrees/` and branches for a live claim before starting an arc.
