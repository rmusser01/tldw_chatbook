# Watchlists URLMonitor Off-Event-Loop Design

Date: 2026-08-14
Task: [TASK-15764](../../../backlog/tasks/task-15764%20-%20Watchlists-URLMonitor-extraction-and-diff-work-off-the-event-loop.md)
Predecessor: TASK-15463

## Goal

Keep scheduled and manual `url`, `url_list`, and `sitemap` Watchlists checks responsive by moving `URLMonitor`'s remaining CPU-heavy HTML extraction and text comparison work off the caller's asyncio event-loop thread without changing fetched content, change classification, run accounting, persistence, or per-URL ordering.

## Context

TASK-15463 moved Watchlists database operations and feed parsing off the event loop. It deliberately left two pure-CPU URL-monitoring stages behind:

- `ContentExtractor.extract_text_from_html`, which parses and cleans an HTML page admitted up to `MAX_FETCH_BYTES_PAGE`;
- change comparison in `URLMonitor.check_url`, including `calculate_change_percentage`, `_segment_for_diff`, `build_change_diff`, `added_and_removed_text`, and `classify_change_type`.

`calculate_change_percentage` is included because it constructs a `difflib.SequenceMatcher`; leaving it inline would retain a substantial event-loop CPU path even if the later diff renderer moved.

## Design

### HTML extraction

`URLMonitor._fetch_url_content` keeps its rate limiting, guarded async HTTP fetch, response validation, headers, and raw-content branch unchanged. For `full` and `auto` extraction only, it awaits `asyncio.to_thread(ContentExtractor.extract_text_from_html, response.text, ignore_selectors)`.

This is one worker hop per fetched URL. The inputs are immutable strings plus a per-call selector list, and the helper has no database or event-loop affinity.

### Change percentage

After the existing content-hash mismatch, `check_url` awaits `ContentExtractor.calculate_change_percentage` through `asyncio.to_thread`. The existing threshold comparison remains on the event loop after the worker returns.

This preserves the important short circuit: a below-threshold change does not pay to segment or build a full diff.

### Significant-change details

A small synchronous module helper owns the existing significant-change calculation:

1. segment the prior and current text once each;
2. pass those shared segments to `build_change_diff`;
3. pass the same segments to `added_and_removed_text`;
4. compute `classify_change_type`;
5. return the five existing output values.

`check_url` invokes that helper through one `asyncio.to_thread` call. It continues to assemble `change_info`, update the snapshot, record the circuit-breaker result, and return the disposition on the event-loop thread.

The helper is an implementation seam, not a new public abstraction or configuration point.

### Ordering and concurrency

`url_list` and `sitemap` execution remains sequential. Each URL finishes its fetch, extraction, comparison, snapshot write, and disposition before the next URL begins. No fan-out, custom executor, queue, semaphore, or new dependency is introduced.

The standard library's default executor is sufficient because this change removes event-loop blocking; it does not claim CPU parallel speedup.

## Error and cancellation behavior

Worker exceptions propagate through the existing `check_url` `try`/`except`, so the circuit breaker records one failure and current per-URL isolation remains authoritative. Invalid ignore selectors retain the current skip-and-log behavior inside `extract_text_from_html`.

Cancelling the awaiting coroutine raises cancellation at the `to_thread` await without waiting for already-started CPU work to stop. That worker may finish in the default executor, but `check_url` does not resume afterward to persist a snapshot, record success, or assemble a result. This is accepted because input remains capped by `MAX_FETCH_BYTES_PAGE`; adding process management or a custom cancellation protocol would exceed the task's latency goal and change failure semantics.

No database operation moves into these plain `to_thread` calls. TASK-15463's `run_db_off_loop` and its in-memory SQLite guard remain unchanged.

## Compatibility constraints

- Extracted text, hashes, change percentage, threshold behavior, diff body and summary, added and removed text, change classification, timestamps, rule-matching fields, snapshots, and dispositions remain byte-for-byte or value-for-value identical.
- First-check, rebaseline, unchanged, and below-threshold paths preserve their current writes and return shapes.
- `url_list` and `sitemap` continue isolating one URL's failure without discarding successful siblings.
- No UI, schema, migration, runtime-policy, or server-backend behavior changes.

## Testing

Tests use thread identity rather than timing:

- a real `_fetch_url_content` path records that HTML extraction runs and that its thread differs from the event-loop thread;
- a real changed-item path records that change percentage and every significant-change operation run off-loop, asserts each operation occurred, and compares the resulting item/disposition to the established semantics;
- a real `url_list` execution with multiple URLs proves every URL—not only the first—takes the extraction and significant-diff worker paths while preserving sequential result order;
- a below-threshold regression proves percentage calculation is off-loop and the expensive significant-change helper is not called;
- a cancellation regression proves a cancelled `check_url` does not resume after a worker finishes to persist a snapshot or report success;
- existing TASK-15463, content-kind/diff, and local Watchlists service modules remain green.

Timing-based ticker assertions may remain as supporting evidence, but thread identity and call counts are the non-vacuous contract.

## Scope

In scope: `URLMonitor` HTML extraction and pure CPU comparison work for `url`, `url_list`, and `sitemap` sources.

Out of scope: HTTP concurrency, source scheduling, database offload, executor tuning, process pools, diff algorithm changes, page-size changes, semantic comparison, UI work, and server Watchlists search parity.

## ADR check

ADR required: no.

Reason: this is the direct residual performance fix explicitly deferred by TASK-15463. It preserves storage, runtime, service, and public contracts and introduces no new long-lived architectural boundary.
