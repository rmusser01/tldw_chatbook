---
id: TASK-34936
title: Move-the-Console-inspector-next-send-export-write-off-the-click-handler
status: Done
assignee: []
created_date: 2026-10-13 14:56
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: low
updated_date: 2026-10-13 19:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The inspector next-send-save handler does path.mkdir + path.write_text of the export-scrubbed next-send JSON — potentially hundreds of KB — synchronously on click (console_conversation_inspector.py:1955-1956, plus a sha256 fingerprint on failure); artifacts_dreams_modal.py:617 has the same shape for its markdown export.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] File writes run via to_thread or a worker with handler only orchestrating and reporting status
- [x] Both the inspector export and the dreams modal export covered
<!-- AC:END -->


## Implementation Notes

- Inspector: the next-send save spawns an `inspector-export` worker whose write (mkdir + write_text, potentially hundreds of KB) runs via asyncio.to_thread; the sanitized failure logging contract (path fingerprint + exception class only) and both toasts are preserved verbatim inside the coroutine.
- Dreams modal: `action_export` spawns a `dreams-story-export` worker — dir resolution, mkdir, and the markdown write all via to_thread, with the feedback record, notices, and `_changed()` applying on completion on the loop.
- Verification: both modules import clean; the touched UI suites sit in the pre-broken family with unchanged baselines. Privacy contracts (export-scrubbed text never in exception text) preserved.
- ADR required: no — off-load of two click handlers.