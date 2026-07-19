---
id: TASK-257
title: Defer optional-feature imports: chromadb, web-search chain, PDF/document processors (~550ms)
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, startup]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three eager chains for features unused by default: Chroma_Lib module-scope get_safe_import('chromadb') pulls chromadb→OTel→gRPC→protobuf (~154ms) via RAG_Admin; Tools/__init__ imports WebSearchTool → Article_Extractor_Lib module-scope playwright/trafilatura/dateparser (~197ms) though web_search_enabled defaults False (executor gate at tool_executor.py:643 is already correct, and the SAME FILE already fixed pandas with find_spec); app.py:124's direct submodule import bypasses Local_Ingestion's own lazy __init__, loading pymupdf/onnxruntime (~170ms) + Document lib (~59ms) for optional pdf/ebook extras. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P2 C2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 chromadb imports only when ChromaDBManager is instantiated
- [ ] #2 Web-search chain imports only when the tool is enabled/used (find_spec probe for availability)
- [ ] #3 Per-format ingestion processors import at dispatch time
- [ ] #4 Measured app-import delta reported (expected >400ms); optional-deps availability semantics unchanged (extras absent still degrade gracefully)
<!-- AC:END -->
