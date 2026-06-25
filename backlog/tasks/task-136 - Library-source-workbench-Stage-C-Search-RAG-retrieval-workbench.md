---
id: TASK-136
title: Library source workbench Stage C Search RAG retrieval workbench
status: Done
assignee:
- '@codex'
labels:
- library
- ux
- search-rag
- stage-c
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Library Search/RAG mode read as a retrieval workbench with visible query, scope, evidence, selected-evidence inspector, blocked handoff, and citation/snippet placeholder states without adding new server/runtime dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Search/RAG mode promotes query input and run action at the top of the center pane with visible blocked recovery when the query or source scope is missing.
- [x] #2 Search/RAG mode shows visible scope rows for All Library, Workspace eligible, Notes, Media, Conversations, Collections, and Import/Export recovery rather than a single dense scope line.
- [x] #3 The evidence region renders empty/searching/blocked/result states with selected evidence updating inspector content.
- [x] #4 Selected evidence inspector shows status, authority, allowed actions, blocked actions, recovery, Console handoff, and future citation/snippet placeholders.
- [x] #5 Stage C does not introduce tldw_server runtime calls, sync promotion, persistence/schema changes, or fake retrieval results.
- [x] #6 Mounted regressions and actual CDP/Textual-web screenshot evidence verify the Stage C Search/RAG workbench.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Stage C changes Library Search/RAG presentation, mounted tests, and QA evidence only. It must not change storage/schema, sync policy, provider/runtime boundaries, service contracts, security, dependencies, or server integration.

1. Add failing mounted regressions for Stage C Search/RAG query hierarchy, visible scope rows, selected evidence inspector sections, blocked Console handoff, and citation/snippet placeholder copy.
2. Update Library Search/RAG display widgets/state helpers to render structured retrieval workbench sections using existing local counts and existing retrieval service outputs only.
3. Preserve existing route IDs, action IDs, query submission, async retrieval worker, and Console handoff seam.
4. Capture actual CDP/Textual-web screenshot evidence after user approval.
5. Update QA evidence, roadmap, and TASK-136 implementation notes.
6. Run focused Library tests and diff hygiene before PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added mounted Stage C regressions for Search/RAG query hierarchy, source-scope table rows, empty evidence guidance, selected-evidence inspector content, Console handoff decision copy, and future citation/snippet placeholders.
- Updated the Library Search/RAG center pane to render Query, Scope, and Evidence terminal sections with visible blocked-run recovery and table-like source scope rows.
- Updated the Search/RAG inspector to present Retrieval Status, Console Handoff, Selected Evidence, Recovery, and Future Attribution sections instead of generic status copy.
- Preserved existing route IDs, action IDs, local retrieval service seam, async query worker, and Console handoff boundaries; no server runtime, sync, schema, or persistence scope was added.
- Captured and user-approved actual CDP/Textual-web evidence at `Docs/superpowers/qa/library-source-workbench-stage-c/library-stage-c-search-rag-cdp-2026-06-24-polish.png`.
- Verification: `Tests/UI/test_library_content_hub.py` passed, Library contract/roadmap tests passed, and `git diff --check` was clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library Search/RAG now reads as a destination-native retrieval workbench: users can see why retrieval is blocked, which source scopes exist, what evidence state they are in, and whether selected evidence can be handed off to Console.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
