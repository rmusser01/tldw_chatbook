---
id: TASK-38298
title: >-
  Console: trace capture still blocks after Compact now or Edit & resend when
  the replaced history exceeds 256 recorded rows
status: To Do
assignee: []
created_date: '2026-10-29 09:47'
labels:
  - console
  - trace
  - compaction
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-33621.29 / TASK-33621.7 review (2026-10-29). Those tasks let the next captured send after Compact now or Edit & resend replace the recorded trace tail with the current saved branch (ADR-097 amendment 2026-10-27). ADR-097 keeps the 256-node replacement range, and the range is measured in stored sequences, so dead sequences left inside it by earlier replacements count too. When the replaced history is wider than that, the planner refuses the send (unsupported_surface_change, count_or_span) and every later send in the chat stops at the 'Trace capture blocked' card again, which is the original defect for long chats.

Reproduced by the review on the fix branch: 130 short turns (259 recorded rows), Compact now reports 'Conversation memory updated', and the next captured send is refused with prefix 0, active_changed 259, incoming 3. The same bound applies to Edit & resend on an early message of such a chat, and to a chat whose earlier rewrites left more than 256 stored sequences inside the range (pinned by Tests/Chat/test_console_trace_rewritten_tail_planner.py).

Lifting the bound is a durable-format decision: SurfaceReplacement rejects a span over 256 when it is read back, so an older app version would fail to read a wider replacement. Options include a chained replacement (each link at most 256 sequences, the last one carrying the first new row), a larger bound for the memory- or branch-anchored shape with a format guard, or a cause-specific card when the bound is hit. The over-span fallback that records an UNSUPPORTED_REPLACEMENT_SPAN omission is not a fix: it drops the first new row and keeps every old row in the surface, so the trace stops matching the request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides in an ADR-097 amendment how a replacement wider than 256 stored sequences is recorded, or that it is not
- [ ] #2 After Compact now on a chat whose recorded history exceeds 256 rows, later captured sends reach the provider with trace capture intact, or the card names the cause, as the decision specifies
- [ ] #3 Edit & resend on an early message of such a chat behaves the same way
- [ ] #4 A test drives a compacted history wider than 256 recorded rows through the real planner and trace ledger and fails on the commit that closed TASK-33621.29
<!-- AC:END -->

## Identity correction

Renumbered from TASK-37600 during PR3298 integration on 2026-10-31; dev already owns that ID for the local SSE UTF-8 fix. The 300 distinct existing remote/HEAD snapshots had a maximum integer claim of 38268; TASK-38298 adds the prescribed 30-ID buffer. Scope and To Do status are unchanged.
