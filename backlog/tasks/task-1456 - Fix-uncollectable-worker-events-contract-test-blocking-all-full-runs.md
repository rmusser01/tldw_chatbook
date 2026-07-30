---
id: TASK-1456
title: >-
  Fix uncollectable test_worker_events_contract.py (StreamDone import) that aborts every default full-suite run
status: In Progress
assignee: []
created_date: '2026-07-30 09:05'
labels:
  - testing
  - bug
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Event_Handlers/test_worker_events_contract.py` (task-634's regression coverage) imports `StreamDone` from `tldw_chatbook.Event_Handlers.worker_events`. TASK-650 removed the legacy streaming branch and `StreamDone` with it, so the module no longer imports — and because pytest interrupts on collection errors by default, **every plain `pytest Tests` run on dev dies before running a single test**. Found by the 2026-07-30 test-suite audit's baseline run (`backlog/docs/test-suite-audit-2026-07-30.md` §4.2). The non-streaming error-propagation contract task-634 actually guards is still real and must stay covered; the streaming-sentinel half of the file asserts behavior that was deliberately removed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] `pytest Tests --collect-only` completes with no collection error from this module
- [ ] task-634's regression (non-streaming failures propagate; no sentinel, no message posts) remains covered
- [ ] The current task-650 contract is covered: streaming requests are rejected with `ValueError` and never reach the core chat function
- [ ] The file's own tests pass

## Implementation Plan

1. Drop the `StreamDone` import and the streaming-sentinel test (removed behavior)
2. Keep the non-streaming failure-propagation test unchanged
3. Add coverage for the current contract: non-streaming success returns the core result verbatim with kwargs forwarded; streaming requests raise `ValueError` without calling the core function
4. Verify: run the file; run full-tree `--collect-only`

## Implementation Notes

Rewrote the module against the current `worker_events.py` contract. Kept
`test_chat_wrapper_function_nonstreaming_failure_raises` byte-identical (task-634's
actual regression guard). Replaced the streaming-sentinel test with
`test_chat_wrapper_function_streaming_request_rejected` (ValueError, core never
called) and added `..._nonstreaming_success_returns_core_result` to pin kwarg
forwarding. Modified: `Tests/Event_Handlers/test_worker_events_contract.py` only.
