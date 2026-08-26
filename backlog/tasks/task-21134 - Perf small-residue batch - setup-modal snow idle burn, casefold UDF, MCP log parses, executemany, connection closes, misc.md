---
id: TASK-21134
title: >-
  Perf small-residue batch - setup-modal snow idle burn, casefold UDF, MCP log parses, executemany, connection closes, misc
status: Done
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - cleanup
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21134). Each item
verified on the pin; none warrants a task alone, together they are a real tax.

1. Setup-modal snow animation ticks a full-screen refresh at 5 Hz - measured 9.9% of a core
   idle (vs 1.8% configured) for every not-yet-configured user: honor reduced-motion by default
   on low-core machines or lower the tick rate (console_setup_modal.py:88+).
2. `unicode_casefold` Python-UDF in WHERE + ORDER BY on watchlists agent-tool queries
   (Subscriptions_DB.py:2908-2985).
3. MCP execution log: two full-file JSON parses + fsync-on-close per tool invocation
   (MCP/execution_log.py:156).
4. Re-chunk per-chunk INSERT loop -> executemany (library_rechunk_service.py:265-271).
5. GC-leaked `with conn:` (no close) in sync_state / event_state / writing / research /
   tamagotchi stores.
6. `EnhancedStatusWidget` recompose-per-status-message during ingest (status_widget.py:82-140).
7. Media-viewer match-nav restyles the whole document per click (library_media_content.py:16-53
   - cache the match list, restyle two lines).
8. Trajectory brush-drag rebuilds the ledger DataTable per mouse-move
   (trajectory_screen.py:861-867 - throttle to once per frame or use the existing worker path).
9. `CAPABILITY_REGISTRY` builds 1,323 frozen dataclasses (62% server-only) and runs
   `validate_registry_completeness()` in production at every import (registry.py:1358,1414 -
   move validation to tests; lazy-build the server partition).
10. Dormant sqlite owners to verify-then-retire: Sync_Interop/notes_mirror.py (no prod caller),
    Widgets/Tamagotchi/tamagotchi_storage.py (never imported), Kanban boot connect with no UI,
    Third_Party/aider/repomap.py diskcache (no prod caller).

## Acceptance Criteria

- [x] Each numbered item is either fixed as described or explicitly declined with a reason in the task notes
- [x] No behavior change beyond the stated performance mechanics; touched areas keep their existing tests green

## Implementation Plan

1. Re-verify every item against dev `68c061984` before touching it. Five of seven findings
   re-verified for wave 7 needed correcting, and this filing predates three further
   corrections, so treat each item's cost claim as a hypothesis.
2. Measure each item on its own, before and after, with a load-independent metric where
   wall-clock is noisy (this box drifts >2x between identical windows).
3. Ship each item as its own commit so any one can be reverted independently.
4. Every new test must be proven to fail against a deliberately broken implementation.
5. Walk quit/unmount and the error path for each shipped item.
6. Drop, with evidence, any item whose cost does not reproduce or whose prescribed fix
   makes things worse.

## Implementation Notes

Five items shipped, five dropped with evidence. Every measurement below is first-hand on
dev `68c061984` with an isolated config/HOME; each shipped item is one commit.

### Shipped

**Item 1 - setup-modal snow (commit 1).** Reproduced exactly: with no provider configured
the Console setup modal blocks, its full-screen snow backdrop ticks at 5 Hz, and the screen
receives **75 Layout + 75 Update messages per 15 s** - the review's own census, to the
message. CPU 6.2% of a core on a quiet box, 13-15% on a loaded one (the review recorded
9.9%); the configured state measured 1.32% and reduced-motion 0.06%. Attribution: the
backdrop covers the whole screen, so every repaint re-composites every line
(`_compositor._render_chops`, 569 Strips per tick); density and glyph count are irrelevant
to that cost, only the repaint rate is. Deterministic per-tick cost: **15.8 ms**.
Fix: tick repaints with `layout=False` (a tick cannot resize the field), the backdrop is
`markup=False` (the field is spaces plus three glyphs), and the interval goes 0.2 s -> 0.4 s
with the per-tick displacement doubled so the flakes still drift at 2.0-7.0 rows/s.
Interleaved A/B in one process, two rounds: **6.20/6.22% -> 2.60/2.81% of a core, Layout
messages 59 -> 1 per 12 s window.**
Declined half of the filing's prescription: defaulting `reduce_motion` to true on low-core
machines. That setting is shared with the splash screen, so flipping its default would
silently disable splash animation too, and it would make a visual default depend on
hardware; the rate cut banks the same order of saving for every user.
*User-visible:* the snow updates 2.5 times a second instead of 5, so it falls in slightly
larger steps at the same speed. `[appearance] reduce_motion` still freezes it entirely.
*Quit/error walk:* timer lifecycle, `pause_snow`/`resume_snow` and the reduced-motion path
are untouched; `markup` is a `setdefault` so an explicit caller still wins; `layout=False`
introduces no new failure mode.

**Item 3 - MCP execution log (commit 2).** Reproduced, with one correction to the filing:
it is two full-file *reads* per append but only ONE full parse until the first rotation
(the rotated generation does not exist yet); after a rotation it really is two. Cost at the
500-record cap: **499 json.loads + 500 json.dumps + 2 reads, 4.599 ms per tool invocation**,
purely to re-derive bytes identical to those the previous append wrote.
Fix: each generation this instance sanitizes is remembered under an identity fingerprint
(device, inode, size, mtime_ns); a matching fingerprint returns the cached bytes, and
`append` refreshes the entry from the bytes it just wrote. Every other outcome - a change by
another process, a replaced or removed path, a symlink in the leaf, any stat failure - is a
cache MISS that falls through to `_read_bytes` and its private-path guards exactly as
before. **499 -> 0 loads, 500 -> 1 dumps, 2 -> 1 reads, 4.599 -> 0.315 ms (14.6x).**
Post-rotation (700 records): 0 reads, 0 loads.
Declined: removing the fsync-on-close. This is a security audit log, the fsync is what makes
a recorded invocation survive a crash, and `open_private_text_append` is shared.
*Quit/error walk:* no new failure mode - the fingerprint is consulted before the guarded
read and never replaces it; a stat failure of any kind is a miss, not a decision.

**Item 4 - re-chunk executemany (commit 3).** Reproduced and modest.
**50 chunks 1.03 -> 0.84 ms; 500 chunks 10.55 -> 8.89 ms; 5,000 chunks 139.73 -> 116.20 ms
(-17%).** The profile says why there is no more to take: what remains is SQLite's own work
(executemany 72 ms, the DELETE 31 ms, commit 13 ms at 5,000 rows) plus 15 ms of uuid4.
The parameter list is now built before the transaction opens, `chunk_index` still comes from
the enumerate index (a skipped chunk leaves the same gap), and the DELETE stays
unconditional.
*Quit/error walk:* a malformed chunk now fails before a transaction is opened rather than
inside one - strictly fewer partial states; the outer transaction on the auto path is
unaffected.

**Item 7 - media-viewer match-nav (commit 4).** Reproduced, and the cheap half of the fix
was worth more than the prescribed half. `sync_search` repainted with `Static.update()`,
whose `layout` defaults to True - but a search refresh restyles the SAME characters, so the
size cannot change. Mounted harness at 120x40 over a 100 KB document, interleaved A/B,
10 clicks per arm, two rounds: **Layout messages 10 -> 0, 84.75/85.18 -> 57.50/57.84 ms CPU
per click (-32%).**
Declined the filing's own prescription ("cache the match list, restyle two lines"): the
match list is 6-10% of the renderable build (0.02 ms of 0.28 ms at 10 KB, 3.46 ms of
34.23 ms at 1 MB), so caching it buys a tenth of what the one-keyword layout fix already
banked, and restyling from cached spans is a real refactor of the highlight machinery for a
click cost that is 0.28-3.5 ms at realistic media sizes.
*User-visible:* nothing - the highlight moves exactly as before, without a screen relayout.
*Quit/error walk:* `sync_search` is synchronous and unchanged apart from the keyword.

**Item 8 - trajectory brush drag (commit 5).** Reproduced, with a correction: it is not
literally per mouse-move. `_set_brush` is equality-gated on the exact time range and the
range is quantized by column, so it is once per column crossed - but a full-width drag
crosses ~70 columns, so the count is the same order. Measured with the gesture delivered as
a burst (a real terminal delivers moves faster than Textual repaints): **69 full ledger
rebuilds per gesture -> 1**, at a measured 5.93 ms per rebuild on a 600-row ledger (0.19 ms
at 8 rows) - about 0.4 s of synchronous event-loop work while the button is still down. With
a repaint forced between every move (the worst case for any throttle) it stays at one
rebuild per move, which is exactly the "once per frame" the filing asked for.
Only the drag path throttles; `apply_brush`, keyboard range selection and `clear_range` still
emit synchronously, so a live snapshot re-brush is not delayed.
*Quit/error walk:* the one hazard here is a debounce that loses state at the end. It cannot:
`on_mouse_up` issues the pending emission itself when the final column matches the last
coalesced one (the equality gate would otherwise emit nothing), a drag abandoned without a
mouse-up drains on the next repaint, and `_flush_brush_emit` is a no-op once unmounted. All
three are tested.

### Dropped, with evidence

**Item 2 - `unicode_casefold` UDF. Does not reproduce as a cost, and half the filing is
wrong.** `EXPLAIN QUERY PLAN` with `sqlite_stat1` absent (the repo rule from TASK-21126)
confirms `SCAN subscriptions` + `USE TEMP B-TREE FOR ORDER BY`. But the UDF is called
**n+2 times, not 2n**: SQLite hoists the deterministic `unicode_casefold(?)` constant, and
the ORDER BY evaluates it only for rows that survived the WHERE (52 calls for a 50-row
table). The filed "WHERE **and** ORDER BY" doubling does not happen. Cost per agent-tool
resolution: **0.034 ms at 50 sources, 0.147 ms at 500, 1.274 ms at 5,000** (miss, i.e. all
three legs: 0.085 / 0.352 / 3.074 ms). Sub-millisecond at any realistic watchlist size, once
per LLM tool call. Fixing it properly needs a normalized indexed column, i.e. a schema
change, which this does not justify.

**Item 5 - GC-leaked `with conn:`. The leak is real; the prescribed fix makes the operation
slower.** Excluded per the brief: `Writing_Interop` (TASK-21125), `Research_Interop`
(TASK-21127), `Notifications/event_state_repository.py` (TASK-21131). Of what remained,
`Widgets/Tamagotchi/tamagotchi_storage.py` has no production importer at all (see item 10),
leaving `Sync_Interop/sync_state_repository.py`.
The leak reproduces and is worse than "style": 50 consecutive `get_latest_mirror_report`
calls left **34 of the 50 connections still open** (live `sqlite3.Connection` objects
2 -> 36) - refcounting does not reclaim them because they land in a reference cycle. An
explicit close in a `_transactional_connection` helper across all 26 sites took that to
**0 -> 0** and kept `Tests/Sync_Interop` green at 301 passed.
It was reverted anyway, because it costs more than it saves. Interleaved A/B, 200 reads per
arm, two rounds: **128.2/111.5 ms (leaking) -> 160.4/185.0 ms (closing)**. Cause, proven by
a third arm: with one anchor connection held open for the run the two arms converge
(129.5/129.1 vs 125.5/126.3 ms). **The leaked handles were acting as WAL anchors** - closing
properly makes every close a last-connection close, which pays a WAL checkpoint and
WAL/shm teardown. The change that wins on both axes is a held connection (the same query on
one measured **0.002 ms against 0.619 ms**, i.e. ~310x), which is exactly the TASK-21125 /
TASK-21127 shape and needs a task of its own, not a line in a residue batch. Recommend
filing that; the leak is bounded (the generational GC reclaims it at ~33 live handles) and
is not an fd-exhaustion risk.

**Item 6 - `EnhancedStatusWidget` recompose. False positive: the widget is never
instantiated.** `EnhancedStatusWidget(` appears nowhere in the repository outside its own
class statement - not in production, not in tests. Its only would-be consumer,
`Event_Handlers/ingest_status_helper.py`, has **zero importers** (its sole other mention is
a row in the diagnostic inventory). The recompose-per-status-message costs nothing during
ingest because nothing mounts it. Dead code, not a hot path; retiring it is a deletion, not
a perf change.

**Item 9 - `CAPABILITY_REGISTRY`. Both halves are too small to justify their risk.**
1,324 rows (the filing said 1,323), 62% server-only - both confirmed.
`validate_registry_completeness()` costs **0.030 ms**, once per process. Moving it to tests
would trade a production invariant for 30 microseconds. `_build_capability_registry` is
**1.41 ms of a 7.65 ms module import**, so lazy-building the server partition could save
~0.9 ms of a 0.825 s warm boot (~0.1%) - in exchange for turning a plain dict that is used
with `in`, `[]`, `.get()` and `.values()` (including `PolicyEngine(CAPABILITY_REGISTRY)` at
app.py:5882) into a lazy mapping.

**Item 10 - dormant sqlite owners. One no longer reproduces; the other three cost zero.**
The Kanban boot connect **was already fixed by TASK-21105**: `LocalKanbanService.__init__`
now resolves the path only and defers its 24 DDL statements to the first `connect()`
(file-backed; `:memory:` stays eager). The other three are confirmed dormant *in production*
and therefore cost nothing at runtime: `Sync_Interop/notes_mirror.py` is reached only via
`notes_m1_flow.py`, whose only importer is `Tests/Sync_Interop/test_notes_m1_flow.py`;
`Widgets/Tamagotchi/tamagotchi_storage.py` has no importer outside its own package except
four test modules; `Third_Party/aider/repomap.py` is imported only by
`Coding/code_mapper.py`, which has **zero importers of its own**. Retiring them is a
dead-code deletion whose blast radius is live gate tests (the private-sqlite inventory, the
CSS consolidation ratchet, the pragma census) - a task with its own review surface, not a
line in a perf batch.

### Test and preflight evidence

Matched A/B against a detached worktree at pristine dev `68c061984`, same command, same
interpreter, three groups:

| group | pristine dev | this branch |
|---|---|---|
| `Tests/MCP` + `Tests/Sync_Interop` + private-sqlite inventory + pragma census | 1 failed, 1450 passed | 1 failed, **1454** passed |
| re-chunk + media-content + reduced-motion + background-effects (+ the new backdrop file) | 48 passed | **56** passed |
| the eight trajectory / trace suites | 201 passed | **205** passed |
| `Tests/UI/test_widget_css_consolidation.py` (the CSS cliff guard) | 33 passed | 33 passed |
| `Tests/UI/test_library_shell.py` (indirect consumer of the media body) | 12 failed, 716 passed | 12 failed, 716 passed -- identical set |

Two reds, both identical in both arms and therefore pre-existing on dev:
`Tests/MCP/test_gateway_runtime_prompts.py::test_real_keyword_search_failure_stays_private_through_prompt_gateway`,
and the twelve `Tests/UI/test_library_shell.py` failures (compared as sets, not counts).

`./scripts/preflight.sh` is green. Its diagnostic-inventory check was **already red on
pristine dev**: the pin carries three stale `RAG_Search/simplified/*` rows left by the merged
TASK-3500 RAG work one commit before this base. All four rows (those three plus this
branch's own, a pure re-indent of an existing `logger.warning` in
`library_rechunk_service.py`) were reviewed statement-by-statement with `--statements` and
the review is recorded in the regen commit rather than absorbed silently.

### Follow-up worth filing

`SyncStateRepository` should be converted to a held connection (thread-local, with the
liveness ping `AgentRuns_DB` already uses). Evidence in the item-5 note above: the current
open-per-operation shape leaks ~33 live connections under load AND costs 0.619 ms per read
against 0.002 ms on a held one, and the minimal "just close it" fix trades the leak for a
per-close WAL checkpoint.

### Files

Modified: `tldw_chatbook/Widgets/Console/console_setup_modal.py`,
`tldw_chatbook/MCP/execution_log.py`, `tldw_chatbook/Library/library_rechunk_service.py`,
`tldw_chatbook/Widgets/Library/library_media_content.py`,
`tldw_chatbook/UI/Widgets/trajectory_timeline.py`.
Added: `Tests/UI/test_console_setup_backdrop_repaint_cost.py`. Extended:
`Tests/MCP/test_execution_log.py`, `Tests/Library/test_library_rechunk_service.py`,
`Tests/Library/test_library_media_content.py`,
`Tests/UI/test_trajectory_timeline_integration.py`.
