---
id: TASK-38298
title: >-
  Console: trace capture still blocks after Compact now or Edit & resend when
  the replaced history exceeds 256 recorded rows
status: In Progress
assignee: []
created_date: '2026-10-29 09:47'
updated_date: '2026-11-07 01:35'
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
- [ ] #1 The owner decides in an ADR-337 amendment how a replacement wider than 256 stored sequences is recorded, or that it is not
- [ ] #2 After Compact now on a chat whose recorded history exceeds 256 rows, later captured sends reach the provider with trace capture intact, or the card names the cause, as the decision specifies
- [ ] #3 Edit & resend on an early message of such a chat behaves the same way
- [ ] #4 A test drives a compacted history wider than 256 recorded rows through the real planner and trace ledger and fails on the commit that closed TASK-33621.29
<!-- AC:END -->

## Identity correction

Renumbered from TASK-37600 during PR3298 integration on 2026-10-31; dev already owns that ID for the local SSE UTF-8 fix. The 300 distinct existing remote/HEAD snapshots had a maximum integer claim of 38268; TASK-38298 adds the prescribed 30-ID buffer. Scope and To Do status are unchanged.

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
**Awaiting the owner (2026-11-06): AC#1 is an owner decision and has not been made.** ADR-337 holds the proposal and its measured alternatives in the amendment proposed 2026-11-06. No product code has changed. The question: do you approve recording a rewritten tail wider than 256 stored sequences as a chain of ordinary replacement rows? Each row spans at most 256 sequences. Content-free placeholders are collapsed by a final link that carries the first new row. The chain is capped at 256 links (about 65,536 sequences) and needs no schema change. The alternatives are a format-guarded wider bound, or keeping the bound and giving the card a cause-specific message.

**Measured on dev 8ae979d81b.** The scratch probes ran under Tests/ and are deleted. Their source and output are in the session evidence directory, bd5/F98/task38298_*.

- **The defect, end to end.** The run used the real controller, store, gateway preparation, compaction service and trace runtime (agent path, recorded provider adapter). After 130 exchanges (259 recorded rows), Compact now says "Conversation memory updated", and the next send is BLOCKED. Nothing is reserved and the provider is not called. The refusal is count_or_span with prefix 0, active_changed 259, incoming 3 and span 259. Edit & resend on the first message gives the same refusal and the same numbers. This is the review's repro, unchanged on current dev.
- **One replacement row wider than 256**, written directly. Every read of the segment then fails with "replacement range may contain at most 256 nodes", including the head recorded before it and the planning of a later send. Any version that enforces 256 would therefore lose the whole conversation's trace on reading a wider bound, not just one call.
- **A chain of ordinary replacement rows**, each at most 256 sequences, written with today's repository and read with today's reader. Both the 259-row compaction and the 131-live-rows-over-261-sequences shape reconstruct exactly the new request. The old head is unchanged, the next send plans as a plain append, and trace GC sweeps nothing. 1,999 rows need 9 links (11 nodes, 9 replacements, 11 events), and the cold projection reads in 0.02 s.
- **The existing pin**, Tests/Chat/test_console_trace_rewritten_tail_planner.py, passes 4 of 4 on 8ae979d81b.

The proposal keeps one secondary limit: the rewritten-tail witness bounds the rows sent after the rewrite point to 256. Neither measured shape reached it. Related but out of scope: the voice post-dispatch import has its own 256-span refusal, post_dispatch_surface_span.
<!-- SECTION:NOTES:END -->
