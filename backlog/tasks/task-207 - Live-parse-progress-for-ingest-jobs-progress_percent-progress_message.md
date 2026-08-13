---
id: TASK-207
title: Live parse progress for ingest jobs (progress_percent/progress_message)
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-12 17:34'
updated_date: '2026-08-12'
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
- [ ] #1 Local parse workers report controlled stage messages through a bounded, non-blocking cross-process channel; a full, closing, or failed progress channel never blocks or changes the parse result.
- [ ] #2 The app accepts progress only for the current pool generation, an assigned in-flight job that remains in PARSING, and a job with no completed payload awaiting the writer; stale, late, terminal, and wrong-generation events are ignored.
- [ ] #3 Queue rows show quiet, readable stage detail and include a percentage only for finite, real bounded measurements; indeterminate work has no fabricated percentage and phase changes clear the previous percentage.
- [ ] #4 Ordinary progress ticks update the mounted progress line in place without replacing row/form widgets or moving focus/scroll, while progress-driven action changes and lifecycle transitions still update their structure correctly.
- [ ] #5 Entering WRITING replaces parse detail with a Saving to Library message, and phase-only Local STT events render human-readable copy while preserving Cancel and Force stop behavior.
- [ ] #6 Local live ticks are memory-only, lifecycle/terminal persistence remains authoritative, and existing Server progress persistence and reconciliation continue to work.
- [ ] #7 Shutdown and broken-pool handling stop and clean up progress resources off the Textual thread without hanging, leaking stale updates, or weakening current parse-pool recovery.
- [ ] #8 Focused registry, worker, runner, state, and canvas tests plus a real Windows spawned-process delivery/shutdown test cover the contract; static checks and relevant documentation pass.
<!-- AC:END -->

## Design

- Detailed design: `Docs/superpowers/specs/2026-08-12-task-207-live-ingest-progress-design.md`.
- ADR required: yes.
- ADR path: `backlog/decisions/061-library-ingest-parse-progress-channel.md`.
- Reason: the task adds a durable process boundary, backpressure policy, resource lifecycle, and shutdown contract while preserving ADR-014's ingest authority.
