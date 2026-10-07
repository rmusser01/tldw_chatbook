---
id: TASK-34421
title: Batch legacy RAG-context store writes on chatbook import
status: Done
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 09:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 / F7: recovery-mode citation writes re-serialize and rewrite the whole cross-conversation JSON store once per message making imports quadratic
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import of 500-cited-message chatbook triggers exactly one store write,Single-message recovery path unchanged,Store content identical to per-message writes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 9 (T9)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added stage_rag_context_record (in-memory, zero file I/O) + flush_rag_context_store (single json.dumps + write_text merging all staged records with the identical setdefault merge semantics); record_message_rag_context is now stage+flush internally (one code path, immediate-flush default preserved, canonical-mode RuntimeError ordering intact). Chatbook importer recovery-fallback stages per message and flushes once in try/finally — finally chosen because the old per-message writer persisted pre-error records and per-conversation except continues (success-only flush would lose them); pinned by tests. Evidence: 500-cited-message import 500 serializes + 500 writes -> 1 + 1; mid-import error path 4+4 -> 1+1 with identical store content; byte-identity vs per-message replay pinned with frozen clock (comparator provably exercised: 500-write replay assertion). Guarded-methods concurrency posture unchanged, no new locking. Files: Chat/chat_conversation_service.py, Chatbooks/chatbook_importer.py, Tests/Chatbooks/test_import_rag_context_batching.py (6 tests). Report: .superpowers/sdd/task-9-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
