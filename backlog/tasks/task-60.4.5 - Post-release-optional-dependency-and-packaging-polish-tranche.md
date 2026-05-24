---
id: TASK-60.4.5
title: Post-release optional dependency and packaging polish tranche
status: Done
labels:
- post-release
- packaging
- dependencies
- ux
priority: medium
parent_task_id: TASK-60.4
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve optional dependency recovery and package metadata after the audit confirms the app is source-honest but users still need clearer install, unavailable-feature, and recovery paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Optional dependency and packaging polish scope references TASK-60.3 actual-use audit evidence.
- [x] #2 Unavailable optional features identify the missing dependency group and safe recovery command without implying broken core functionality.
- [x] #3 Package metadata and setup docs distinguish local-first baseline from advanced media, RAG, MCP, server, and web capability extras.
- [x] #4 QA verifies missing-dependency recovery copy and package/install paths before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->
1. Add failing focused assertions for optional feature metadata, safe install commands, and local-first/advanced-capability documentation copy.
2. Centralize optional feature metadata so unavailable states can reuse a single source for dependency group, owner, capability area, and source/package install commands.
3. Wire Search/RAG missing-dependency recovery through the metadata registry without changing core local-first behavior.
4. Update README and release recovery setup docs to distinguish baseline installation from optional advanced capability groups.
5. Run focused optional-dependency, recovery-copy, and packaging documentation verification before marking the task done.
<!-- SECTION:IMPLEMENTATION_PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Scope follows TASK-60.3 audit evidence that optional dependency failures should present recoverable, source-honest states instead of making the product appear broken.
- Added optional feature metadata helpers in `tldw_chatbook/Utils/optional_deps.py` for capability area, owner, unavailable feature copy, and source/package install commands.
- Updated Search/RAG recovery copy to use the `embeddings_rag` metadata entry, preserving `Search/RAG queries` as the unavailable feature while exposing safe install actions.
- Updated README and `Docs/Development/release-recovery-setup.md` with local-first baseline guidance and advanced optional capability group install paths.
- Verified with focused optional-dependency, recovery-copy, packaging, and recovery-doc tests: 29 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-60.4.5 completed. Optional dependency metadata now has a central registry, Search/RAG recovery uses the registry, setup docs distinguish the local-first baseline from advanced extras, and focused QA verifies recovery copy plus package/install paths.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
