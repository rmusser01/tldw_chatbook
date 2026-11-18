---
id: TASK-38298
title: >-
  Console: trace capture still blocks after Compact now or Edit & resend when
  the replaced history exceeds 256 recorded rows
status: Done
assignee: []
created_date: '2026-10-29 09:47'
updated_date: '2026-11-12 23:19'
labels:
  - console
  - trace
  - compaction
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-33621.29 / TASK-33621.7 review (2026-10-29). Those tasks let the next captured send after Compact now or Edit & resend replace the recorded trace tail with the current saved branch (ADR-337 amendment 2026-10-27). ADR-337 keeps the 256-node replacement range, and the range is measured in stored sequences, so dead sequences left inside it by earlier replacements count too. When the replaced history is wider than that, the planner refuses the send (unsupported_surface_change, count_or_span) and every later send in the chat stops at the 'Trace capture blocked' card again, which is the original defect for long chats.

Reproduced by the review on the fix branch: 130 short turns (259 recorded rows), Compact now reports 'Conversation memory updated', and the next captured send is refused with prefix 0, active_changed 259, incoming 3. The same bound applies to Edit & resend on an early message of such a chat, and to a chat whose earlier rewrites left more than 256 stored sequences inside the range (pinned by Tests/Chat/test_console_trace_rewritten_tail_planner.py).

Lifting the bound is a durable-format decision: SurfaceReplacement rejects a span over 256 when it is read back, so an older app version would fail to read a wider replacement. Options include a chained replacement (each link at most 256 sequences, the last one carrying the first new row), a larger bound for the memory- or branch-anchored shape with a format guard, or a cause-specific card when the bound is hit. The over-span fallback that records an UNSUPPORTED_REPLACEMENT_SPAN omission is not a fix: it drops the first new row and keeps every old row in the surface, so the trace stops matching the request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner decides in an ADR-337 amendment how a replacement wider than 256 stored sequences is recorded, or that it is not
- [x] #2 After Compact now on a chat whose recorded history exceeds 256 rows, later captured sends reach the provider with trace capture intact, or the card names the cause, as the decision specifies
- [x] #3 Edit & resend on an early message of such a chat behaves the same way
- [x] #4 A test drives a compacted history wider than 256 recorded rows through the real planner and trace ledger and fails on the commit that closed TASK-33621.29
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure on origin/dev (done 2026-11-06 on dev 8ae979d81b):
   - a real-controller repro of Compact now and of Edit & resend at 130 exchanges;
   - what the reader does with one replacement row wider than 256;
   - a chained replacement of links of at most 256 sequences, written and read with today's code;
   - trace GC over that chain, and its cost at 1,999 rows.
2. Write the ADR-337 amendment proposal (chained replacement) with the measured alternatives, and stop for the owner's decision (AC#1).
3. Only after the owner decides, if the chain is approved:
   - Write the tests first. A planner and ledger test of a compacted history wider than 256 recorded rows must be red on the commit that closed TASK-33621.29. Also cover Edit & resend on an early message and the dead-sequence shape.
   - In the planner, group the replaced live rows into links of at most 256 stored sequences.
   - In the writer, persist the placeholder links and the final link in the dispatch-binding transaction, and revalidate each link there.
   - Make the incremental projection updates follow the chain.
   - Re-measure the reservation p95, update the User Guide's known limit, and verify live with Compact now on a long chat and a real model.
4. If the owner chooses the card instead: give count_or_span a cause-specific card when a rewritten tail exceeds the bound, with tests and a guide update.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decision 2026-11-12: "Approve the chain".** The owner was asked: "record a rewritten history wider than 256 stored rows as a chain of ordinary replacement rows, each at most 256; content-free placeholders collapsed by a final link; cap 256 links (about 32,000 exchanges); no schema change, today's reader already reconstructs it exactly, old traces unaffected". ADR-337's amendment 2026-11-06 is now Accepted and records the decision, the implementation and the re-measured reservation latency. AC#1 is met.

**Implemented (plan step 3, chain approved).**

- Planner (`prepare_current_surface_delta`): the replaced live rows of an admitted rewritten tail are cut, in stored-sequence order, into links of at most 256 stored sequences (`_replacement_links`). Dead sequences inside a link count; a dead run between links is not replaced again. The whole-range `sequence_gap` check is unchanged. More than `MAX_SURFACE_REPLACEMENT_CHAIN_LINKS` (256) links is refused as `count_or_span`, as before. The rewritten-tail witness keeps its own bound of 255 saved rows plus the current user.
- Types: `SurfaceReplacementLink`; `VerifiedSurfaceReplacementRange` and `VerifiedSurfaceReplacement` carry `chain` (2 to 256 links, ordered, covering the range), and only a rewritten tail may carry one. New omission reason `replacement_chain_link`.
- Writer (`_replace_surface`, in the dispatch-binding transaction): each link writes a content-free placeholder (`surface_replacement` / omission / `replacement_chain_link`) and one ordinary replacement row with a `surface_replace` event, no gap event. A final link replaces the placeholders with the first new row; the rest append as before. `_validate_replacement_range` checks every link's start, end and contiguity, and that no live row lies between links, in preparation and again at binding. `PersistedTraceRequest` returns the placeholders and links, and dispatch reconciliation checks each one.
- Warm checkpoint: placeholder sequences precede the new rows in the prepared projection, and the placeholders enter the lineage domains, so the in-memory checkpoint equals what a restart rebuilds. `_updated_projection_entries` removes each link's range.
- The reader is unchanged: it folds the links in replacement-node order, as every version since the ledger landed does.
- User Guide (`console/semantic-trace-capture.md`): the known limit now describes the chain and its remaining limits.

**Tests.** New `Tests/Chat/test_console_trace_chained_replacement.py` reserves, plans, verifies and binds real sends through the trace service: Compact now and Edit & resend on the first message at 130 exchanges (259 rows), warm and restarted, captured with the exported request (native reader) equal to what was sent; the head recorded before still reconstructs; the next send is a plain append; the warm checkpoint equals a cold rebuild; trace GC keeps every link; 131 live rows over 261 sequences chains as (0..255)(256..260); over the link cap is refused; each link is revalidated (gap, wrong end node). `Tests/Chat/test_console_trace_rewritten_history.py` adds end-to-end tests through the real controller, store, gateway, compaction service and trace runtime: Compact now (agent) and Edit & resend on the first message (agent and plain) on a saved 129-exchange chat plus one captured send. The two pins in `test_console_trace_rewritten_tail_planner.py` that asserted the refusal now assert the chain (names kept for the CI P0 list). All new and changed tests are red on origin/dev 4b90eb860b, and the ledger tests are red on db7e04cf48 (the merge that closed TASK-33621.29), with `unsupported_surface_change` (AC#4). Mutation checks: link span 257, dropped placeholder sequence offset, placeholders left out of the lineage, no link cap, and no between-links check each turn the matching tests red.

**Reservation re-measured** (reserve plus dispatch binding, 20 interleaved rounds, M4 Pro under a shared load average of 28 to 48): append 259 rows p50/p95 3.2/10.0 ms wall, 3.2/4.9 ms thread CPU; one 256-row replacement 7.8/20.4 wall, 6.5/10.1 CPU; 259-row chain (2 links) 11.9/55.5 wall, 9.1/12.6 CPU; 1,999-row chain (8 links) 109/146 wall, 73/89 CPU. The cost is paid once, by the send after the rewrite; ordinary sends take no new work.

**Live verification 2026-11-12** (isolated profile, Anthropic claude-haiku-4-5 through the review proxy): one captured send, 256 messages appended to that chat through the DB API, another captured send (259 recorded rows), Compact now ("Conversation memory updated"), then "Say just n129.": no Trace capture blocked card, the reply arrived, the call is complete and verified, and the native reader rebuilds the memory plus the 3 rows sent. The ledger holds links (0..255) and (256..258) to placeholders 259 and 260, and a final link (259..260) to the first new row 261; no gap event; no `unsupported_surface_change` in the app log.

**Size ratchet.** `console_trace_service.py` 6600 -> 6739 (+139) under the owner rule of 2026-10-04.

Out of scope, unchanged: the voice post-dispatch import's own 256-span refusal (`post_dispatch_surface_span`).

**Proposal evidence, measured on dev 8ae979d81b (2026-11-06).** The scratch probes ran under Tests/ and are deleted. Their source and output are in the session evidence directory, bd5/F98/task38298_*.

- **The defect, end to end.** The run used the real controller, store, gateway preparation, compaction service and trace runtime (agent path, recorded provider adapter). After 130 exchanges (259 recorded rows), Compact now says "Conversation memory updated", and the next send is BLOCKED. Nothing is reserved and the provider is not called. The refusal is count_or_span with prefix 0, active_changed 259, incoming 3 and span 259. Edit & resend on the first message gives the same refusal and the same numbers. This is the review's repro, unchanged on current dev.
- **One replacement row wider than 256**, written directly. Every read of the segment then fails with "replacement range may contain at most 256 nodes", including the head recorded before it and the planning of a later send. Any version that enforces 256 would therefore lose the whole conversation's trace on reading a wider bound, not just one call.
- **A chain of ordinary replacement rows**, each at most 256 sequences, written with today's repository and read with today's reader. Both the 259-row compaction and the 131-live-rows-over-261-sequences shape reconstruct exactly the new request. The old head is unchanged, the next send plans as a plain append, and trace GC sweeps nothing. 1,999 rows need 8 links plus the final link (11 nodes, 9 replacements, 11 events), and the cold projection reads in 0.02 s.
- **The existing pin**, Tests/Chat/test_console_trace_rewritten_tail_planner.py, passes 4 of 4 on 8ae979d81b.

The proposal keeps one secondary limit: the rewritten-tail witness bounds the rows sent after the rewrite point to 256. Neither measured shape reached it. Related but out of scope: the voice post-dispatch import has its own 256-span refusal, post_dispatch_surface_span.

**PR #3515 review follow-up (2026-11-12).** Two tests added: a failure after the binding commit still reconciles a committed chain to the dispatched call (red when reconciliation compares every link's predecessor with the planned head, not just the first link's), and a later chain from the root skips the dead run 2..299 between its links (1..1) and (300..305), with its second link spanning the first chain's placeholders. ADR-337 now counts links one way: a link is one group's replacement row, and the final link is counted separately. Filed TASK-40577 (binding should refuse a link that covers no live row) and TASK-40578 (each link is validated twice while the write lock is held).
<!-- SECTION:NOTES:END -->

## Identity correction

Renumbered from TASK-37600 during PR3298 integration on 2026-10-31; dev already owns that ID for the local SSE UTF-8 fix. The 300 distinct existing remote/HEAD snapshots had a maximum integer claim of 38268; TASK-38298 adds the prescribed 30-ID buffer. Scope and To Do status are unchanged.
