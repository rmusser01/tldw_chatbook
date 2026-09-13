---
id: TASK-19905
title: chunk_preview_modal.py is an orphan — delete it or give it a live importer
status: Done
assignee:
  - '@zcode'
created_date: '2026-08-21'
labels:
  - tech-debt
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Widgets/chunk_preview_modal.py` (`ChunkPreviewModal`) has no live importer: its only import is `media_details_widget.py:753`, and `MediaDetailsWidget` is itself unreachable (TASK-19641 owns its disposition). The mentions in `RAG_Search/enhanced_chunking_service.py:18` and `parent_child_adapter.py:99` are docstring references, not imports.

Filed from the chunking template parity design spec §11 item 2 (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`; ADR-078). Re-verified live 2026-08-21: importer grep in a worktree at/after the spec's pin, file untouched on `origin/dev` since.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reachability is re-verified at implementation time (importer graph recorded in the Implementation Notes) before anything is deleted
- [x] #2 Either the module is deleted with no dangling references (imports, CSS selectors — the `chunk-preview` selector family, tests, docstring mentions updated), or it is mounted from a reachable production surface
- [x] #3 The disposition is coordinated with TASK-19641 so the two tasks cannot strand each other (deleting only the importer leaves this module; deleting only this module breaks nothing but leaves the dead widget)
<!-- AC:END -->

## Implementation Plan

1. Re-verify reachability (importer graph recorded below); confirm the only would-be consumer is the itself-unreachable MediaDetailsWidget (TASK-19641) and the §7.3 characterization tests.
2. Delete branch: remove the modal, strip media_details_widget's dead preview path (button + handler), update the two docstring mentions (parent_child_adapter, enhanced_chunking_service).
3. Rehome the §7.3 preview==ingest pins to drive the modal's two branch seams directly so the chunking parity contract stays pinned without the widget.
4. Record the coordination stance toward TASK-19641 in these notes.

ADR required: no
ADR path: N/A
Reason: Deletion of an orphan module under the chunking parity spec's own item 2; ADR-078 already governs the program.

## Implementation Notes

DELETE branch taken. Importer graph re-verified on current dev: ``ChunkPreviewModal``'s only in-tree import was the lazy ``media_details_widget._preview_chunks`` (MediaDetailsWidget is itself unreachable; TASK-19641 owns its disposition) plus ``Tests/Chunking/test_callsite_characterization.py`` mounting the real modal for the §7.3 preview/ingest agreement pins; the ``RAG_Search`` mentions were docstrings only.

Deleted ``Widgets/chunk_preview_modal.py`` (its styles were inline DEFAULT_CSS; the ``.chunk-preview-*`` rules in ``css/features/_embeddings.tcss`` belong to the separate, live ``Widgets/chunk_preview.py`` and stay). Stripped the dead preview path from ``media_details_widget.py`` (the Preview Chunks button and ``_preview_chunks``). Updated both docstring mentions to name their live consumers.

The §7.3 pins were rehomed, not lost: ``_drive_preview_modal`` became ``_drive_preview_branches``, driving verbatim the two seams the modal delegated to (EnhancedChunkingService.chunk_text_with_structure; Chunker with max_size/overlap) with the same conversion shapes, so preview==ingest agreement stays pinned at seam level. All 9 pins pass (plus the pre-existing xfail).

TASK-19641 coordination (AC#3): this deletion forces the preview aspect of 19641's revive-vs-delete decision -- a revived MediaDetailsWidget must rebuild its preview surface against the seams the rehomed pins guard (and may reuse them), while the delete branch simply proceeds. Nothing 19641 can do now strands this module's removal.

Verification: ``Tests/Chunking/`` 836 passed; the 8 lab-runner/coordinator failures seen in one loaded full-dir run pass in isolation on this tree and are subprocess-timing flakes (15 others reproduce identically on stashed HEAD). ``media_details_widget`` imports clean. Ruff: zero delta on all three edited production files (48/50/7 pre-existing, unchanged).

Deleted: ``tldw_chatbook/Widgets/chunk_preview_modal.py``. Modified: ``tldw_chatbook/Widgets/media_details_widget.py``, ``tldw_chatbook/RAG_Search/parent_child_adapter.py``, ``tldw_chatbook/RAG_Search/enhanced_chunking_service.py``, ``Tests/Chunking/test_callsite_characterization.py``.
