---
id: TASK-34432
title: Small perf fixes batch B
status: Done
created_date: 2026-10-07 02:43
updated_date: 2026-10-07 23:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 7 / F19d-f: chatbooks registry re-parses per call the chat debug payload loop runs unconditionally and the post-gen dictionary is re-read from disk every response
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Two list_chatbooks calls trigger one file read,INFO level performs zero payload dump work,Two responses trigger one dictionary parse
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 20 (T20)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Three fixes: (20a) chatbooks registry cached keyed (path, size, mtime_ns), stat-first re-parse on change, own-write path (_save_registry under the per-path RLock) refreshes the cache from the persisted payload (no mtime-granularity reliance), record copies remain independent; (20b) the debug payload-summary loop extracted verbatim into _debug_dump_llm_payload_summary and gated on logging.getLogger().isEnabledFor(DEBUG) — argument evaluation genuinely suppressed at INFO (site uses stdlib module-level logging.debug delegating to the root logger; byte-identical golden at DEBUG); (20c) post-gen replacement dictionary cached by (path, size, mtime_ns), parser exceptions uncached, golden replacement output through the real pipeline. Two list_chatbooks calls -> one parse (spy); two responses -> one dict parse (spy). 9 new TDD tests; failure lists byte-identical to baseline. Files: Chatbooks/local_chatbook_service.py, Chat/Chat_Functions.py, Tests/Chatbooks/test_local_chatbook_service.py, Tests/Chat/test_chat_send_path_efficiency.py. Report: .superpowers/sdd/task-20-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
