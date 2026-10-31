---
id: TASK-37601
title: >-
  Console: prove the rewritten-tail memory anchor against the branch's effective
  memory
status: To Do
assignee: []
created_date: '2026-10-29 09:49'
labels:
  - console
  - trace
  - compaction
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-33621.29 review (2026-10-29). The rewritten-tail proof (tldw_chatbook/Chat/console_trace_rewritten_tail.py) accepts a memory anchor when ANY active console_conversation_memories row of the conversation has the oldest replacement row's parent as its boundary_message_id. It does not check which memory is actually in effect for the current branch (the memory selection head on the active lineage, its scope, or a legacy conversations.context_summary boundary), and it never checks that the call's system_message carries that memory. Trace truthfulness holds, because every captured value is the exact value sent, but the proof is weaker than the ADR-097 amendment's wording.

The memory branch also ignores retained saved rows: a [retained saved rows][gap][rows after the boundary] request is admitted without any link from the retained tail to the memory's range. That is the shape a manual range memory (plan_manual_range, coverage RANGE) produces, so it is intended there, but nothing ties the retained tail to the range start. Today's behaviour is pinned by the unit tests; tighten it to the effective memory and its coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The memory anchor is accepted only for the memory in effect on the current branch, read the way context preparation selects it
- [ ] #2 A range memory is accepted only when the last retained saved row is the parent of the range's first message; a prefix memory admits no retained saved row
- [ ] #3 A legacy prefix summary boundary is either proven the same way or explicitly refused, as ADR-097 records
- [ ] #4 Unit tests cover each refusal and fail if the effective-memory check is removed
<!-- AC:END -->
