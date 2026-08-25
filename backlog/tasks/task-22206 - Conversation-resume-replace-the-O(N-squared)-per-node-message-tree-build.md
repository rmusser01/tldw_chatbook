---
id: TASK-22206
title: >-
  Conversation resume: replace the O(N-squared) per-node message-tree build
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - chat
  - database
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22206).

`Chat/chat_conversation_service.py:1136-1184`: `_build_message_tree` recurses once per
message and issues one `get_messages_for_conversation_by_parent_ids` per node
(`DB/ChaChaNotes_DB.py:9684-9716`). With `sqlite_stat1` absent (no production DB has it)
the plan uses `idx_msgs_conv_ts` and post-filters — every per-node call scans all N rows,
hydrating `content` and the `image_data` BLOB. Python adds a per-node `set(seen)` copy.
Measured in-memory with 1 KB blobs: 3.0 ms @100 msgs, 22.7 @300, 89.8 @600 (clean N^2);
the same walk with `idx_msgs_parent` forced: 2.0 ms @600 — 45x, O(N). The walk is awaited
inline on the loop (`UI/Console_Modules/workspace.py:3679`, `:3706`) on every saved-
conversation resume/session restore. Also: recursion depth equals conversation length —
a ~980-message linear conversation raises RecursionError on resume.

## Acceptance Criteria

- [ ] Resume performs O(N) total work: either one conversation-scoped query with in-memory tree assembly, or a per-parent query shape proven by EXPLAIN QUERY PLAN with sqlite_stat1 ABSENT to use `idx_msgs_parent`
- [ ] BLOB columns are not hydrated during tree construction
- [ ] A 2000-message linear conversation resumes without RecursionError (iterative build or explicit bound with a graceful path)
- [ ] Resume time measured before/after at 600+ messages; the walk runs off the event loop or its on-loop time is bounded and stated
