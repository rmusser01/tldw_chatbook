---
id: TASK-207
title: Live parse progress for ingest jobs (progress_percent/progress_message)
status: Done
assignee:
  - '@codex'
created_date: '2026-07-12 17:34'
updated_date: '2026-08-13 08:51'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Long-running Local ingest jobs expose only coarse parsing/writing states even though Server jobs already project structured progress. The shared progress field, Server reconciliation, and secondary queue-row line now exist; this task completes the missing Local process-worker-to-UI path. Local jobs report truthful stage detail and exact percentages only when the parser has a real bounded measurement, without adding job states, blocking ingestion, destabilizing the queue UI, or persisting high-frequency telemetry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Local parse workers report controlled stage messages through a bounded, non-blocking cross-process channel; a full, closing, or failed progress channel never blocks or changes the parse result.
- [x] #2 The app accepts progress only for the current pool generation, an assigned in-flight job that remains in PARSING, and a job with no completed payload awaiting the writer; stale, late, terminal, and wrong-generation events are ignored.
- [x] #3 Queue rows show quiet, readable stage detail and include a percentage only for finite, real bounded measurements; indeterminate work has no fabricated percentage and phase changes clear the previous percentage.
- [x] #4 Ordinary progress ticks update the mounted progress line in place without replacing row/form widgets or moving focus/scroll, while progress-driven action changes and lifecycle transitions still update their structure correctly.
- [x] #5 Entering WRITING replaces parse detail with a Saving to Library message, and phase-only Local STT events render human-readable copy while preserving Cancel and Force stop behavior.
- [x] #6 Local live ticks are memory-only, lifecycle/terminal persistence remains authoritative, and existing Server progress persistence and reconciliation continue to work.
- [x] #7 Shutdown and broken-pool handling stop and clean up progress resources off the Textual thread without hanging, leaking stale updates, or weakening current parse-pool recovery.
- [x] #8 Focused registry, worker, runner, state, and canvas tests plus a real Windows spawned-process delivery/shutdown test cover the contract; static checks and relevant documentation pass.
<!-- AC:END -->

## Design

- Detailed design: `Docs/superpowers/specs/2026-08-12-task-207-live-ingest-progress-design.md`.
- ADR required: yes.
- ADR path: `backlog/decisions/061-library-ingest-parse-progress-channel.md`.
- Reason: the task adds a durable process boundary, backpressure policy, resource lifecycle, and shutdown contract while preserving ADR-014's ingest authority.

## Implementation Plan

1. Add a stdlib-only, spawn-safe progress event/sink and deterministic latest-per-job coalescer with bounded primitive normalization before and after IPC.
2. Add progress-specific registry listeners, memory-only Local tick updates, writing-stage reset, and one pure readable progress formatter while preserving Server persistence.
3. Bind job identity in the parse worker and instrument only stages observable through current parser/transcription seams; callback failure remains non-authoritative.
4. Give each parse-pool generation an atomic queue/pool resource bundle, coalescing drain thread, Windows-safe construction, and off-Textual-thread cleanup.
5. Fence progress by generation, generation membership, PARSING state, and absence of a payload-ready result before transient registry application.
6. Mount stable active-row progress widgets, patch ordinary ticks in place, retain structural recomposition for action changes, and add dedicated muted styling with generated CSS.
7. Run focused and real-spawn verification, static/CSS/backlog checks, document truthful behavior, self-review the failure boundaries, and close TASK-207 only with complete evidence.

ADR required: yes.

ADR path: `backlog/decisions/059-library-ingest-parse-progress-channel.md`.

Reason: ADR-059 defines the new cross-process message, backpressure, resource lifecycle, shutdown, and UI projection contract; ADR-014 remains authoritative for ingest ownership and recovery.

Detailed plan: `Docs/superpowers/plans/2026-08-12-task-207-live-ingest-progress.md`.

## Implementation Notes

- Added the ADR-059 process contract: each spawned parse-pool generation owns a bounded queue and a stdlib-only, picklable event containing only bounded primitive data. Worker emission uses `put_nowait`, drops full/closed-channel telemetry, and cannot alter the parse result. A latest-per-job coalescer limits UI delivery; deterministic mutation proof now directly covers `take_due(force=True)` flushing and clearing pending telemetry.
- Bound parser callbacks only at observable stages. Percentages are retained only when a provider exposes a finite bounded total; indeterminate work remains text-only. The writer transition replaces parse telemetry with `Saving to Library`, so a stale parse percentage cannot survive into WRITING.
- Local parse ticks use the registry's progress-only listener path with `persist=False`; lifecycle and terminal transitions remain authoritative and persisted. The default `persist=True` Server reconcile path is unchanged. Ordinary ticks patch a stable mounted progress widget in place, while action-signature and lifecycle changes still recompose the necessary structure.
- Each progress event is fenced after IPC by current generation, generation membership, PARSING state, known job id, and absence of a payload-ready result. Broken-pool and quit paths detach resources synchronously but terminate/join the pool, close/cancel the queue, and bounded-join the drain only on a daemon cleanup thread. Real Windows spawn coverage exercised delivery, teardown, worker death, and fileno-less stderr construction.
- Integrated verification was split after the buffered aggregate command exceeded the five-minute no-output threshold. With a repository-local pytest base temp and `allow_network` applied only to actual async test functions, the eight affected modules produced **640 passed, 1 skipped, 1 deselected, 0 product failures**: core contract/worker/local/registry/state/server reconcile **377 passed, 1 skipped, 1 deselected**; runner **132 passed**; ingest canvas **131 passed**. The one skip is the absent optional `pymupdf` dependency. The one deselection is the fixture-only Windows symlink test, which fails before product code with WinError 1314 because this account lacks symlink privilege. The passing runner evidence includes real spawned-pool progress delivery/cleanup and worker-exit recovery.
- Windows harness limitations were recorded separately rather than counted as product passes. The literal aggregate command could not use the default pytest temp root because that root is ACL-inaccessible. With only a local base temp added but no async network allowance, it completed with **462 passed, 1 skipped, 1 failed, 356 errors**: all 356 errors were Windows Proactor socketpair setup/teardown blocked by `Tests/network_guard`, and the one failure was the symlink-privilege fixture above. The literal focused screen selection likewise produced **12 Proactor setup/teardown errors, 553 deselected**; the scoped async counterpart produced **6 passed, 553 deselected**, covering in-place widget identity, registry-listener cleanup, context preservation, structural action changes, and vanished-widget tolerance.
- The exact unfiltered Ruff command reported **152 inherited whole-file findings** in `app.py`, `library_screen.py`, and pre-existing lines in the runner/shell tests. Blame and changed-hunk review confirmed none were introduced by TASK-207; scoped checks over every changed file/hunk passed after ignoring only those inherited rule classes. `compileall` passed, `git diff --check` passed, and the current CI duplicate-ID guard passed across **1,852 task files**.
- CSS was regenerated twice with UTF-8 console output. The builder changes only its generated timestamp on repeat runs; after removing that header, SHA-256 was identical (`2ED8DDB08D5150FC060A5B85D9D4EC4A17228353D31E67BD1E237D85A758B1DB`). The committed generated timestamp was restored, leaving no generated-content diff.
- Failure-oriented review of the complete branch diff explicitly checked blocking queue writes, UI-thread joins, message bounds, picklability, stale generations, payload-ready races, transient persistence, writing-stage percentage reset, lifecycle-prefix duplication, active progress-widget identity, and generated CSS provenance. No product issue was found. No new dependency, schema, license, or security boundary was introduced beyond ADR-059; ADR-014 remains the lifecycle/authority decision. No new generalized lesson was needed because the observed Windows temp, Proactor network-guard, optional-dependency, and symlink constraints are existing documented harness limitations.
- Updated `Docs/User_Guide/library.md` with the truthful transient-progress limits. Principal implementation files are the new progress contract plus worker/parser plumbing, registry/state projection, app-level resource ownership/fencing, Library screen/canvas identity updates, CSS source/generated bundle, and focused regression tests.
