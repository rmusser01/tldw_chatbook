---
id: TASK-19191
title: Per-row-reviewed regeneration of the persistent diagnostic inventory (dev red)
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - test-health
  - diagnostics
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_persistent_diagnostic_inventory.py` is red on dev:
`scripts/check_persistent_diagnostic_inventory.py` exits 1 at dev `7877defba`
(reproduced 2026-08-20 in a pristine worktree, PYTHONPATH+cwd pinned,
`tldw_chatbook.__file__` asserted inside the worktree). The checker's own
guidance applies: review the diff row by row before running `--write` —
regenerating without reading it is the exact failure mode the content-keyed
digest exists to prevent (task-3750).

Current rebuild-vs-committed diff (copy committed → `--write` → structured
diff → committed file restored):

Rows only in COMMITTED (1):
- `RAG_Search/enhanced_chunking_service.py` (count=6) — file deleted, row kept.

Rows only in REBUILD (22):
- 19 `Chunking/engine/` files from the chunking-engine landing (base 13,
  chunker 49, multilingual 9, process_text/options 1, regex_safety 1,
  security_logger 7, strategies: code 3, code_ast 5, ebook_chapters 17,
  fixed_size 1, json_xml 25, paragraphs 11, rolling_summarize 2, semantic 11,
  sentences 18, structure_aware 6, tokens 28, words 15; utils/metrics 2)
- `RAG_Search/parent_child_adapter.py` (1)
- `UI/Library_Modules/library_media_browse_controller.py` (2) — the known
  pre-existing 3-row library drift (post-d64608b84 regen, 1ba3d4755/b4ebe85e8
  era)
- `Widgets/Console/console_changed_files_section.py` (2) — NEW since the
  wave-3 queue was written: the changed-files rail landing (12d621071 et seq.)

Rows changed (10):
- `Chat/console_chat_controller.py` 45→45 (digest only)
- `Chunking/Chunk_Lib.py` 100→31
- `DB/Client_Media_DB_v2.py` 354→338 (library drift)
- `Event_Handlers/STTS_Events/stts_events.py` 30→29 (TASK-19043's merged
  deletion — implementer and reviewer both missed the hand-edit playbook step)
- `RAG_Search/chunking_service.py` 5→3
- `RAG_Search/simplified/enhanced_rag_service.py` 11→10
- `UI/Screens/change_review_screen.py` 1→13 (changed-files rail landing)
- `UI/Screens/chat_screen.py` 153→156
- `UI/Screens/library_screen.py` 110→109 (library drift)
- `Widgets/enhanced_file_picker.py` 6→6 (digest only)

persistent_sink_topology: +1 row in rebuild —
`Chunking/engine/security_logger.py` (an addHandler-kind sink; sink additions
are exactly what this inventory exists to review).

Summary: owner_files 494→515, persistent_sink_files 6→7, task_492_calls
1209→1209, task_494_calls 6974→7122.

The drift has GROWN past the contributors known at wave-3 close-out (the
changed-files rail rows and the chunking_service/enhanced_rag_service/
chat_screen deltas are new), so re-produce the diff at the implementation
commit before reviewing. A red gate left standing stops guarding — the next
unreviewed diagnostic lands invisibly behind it (same failure shape as the
task-19044 lesson).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The rebuild-vs-committed diff is re-produced at the implementation commit and every row delta is individually reviewed and dispositioned in the task's Implementation Notes (new diagnostics accepted or challenged, deleted rows confirmed against deleted/moved code, digest-only changes explained), with particular attention to the new persistent sink in `Chunking/engine/security_logger.py`.
- [ ] #2 The committed inventory is regenerated via the checker's `--write` only after that review, and `Tests/Architecture/test_persistent_diagnostic_inventory.py` passes on the result.
- [ ] #3 Any row the review REJECTS (a diagnostic that should not exist, e.g. one that could log sensitive content) is filed or fixed rather than silently accepted into the inventory.
- [ ] #4 No hand edits to the regenerated JSON beyond what the review justifies; the summary counts match the accepted rows.
<!-- AC:END -->
