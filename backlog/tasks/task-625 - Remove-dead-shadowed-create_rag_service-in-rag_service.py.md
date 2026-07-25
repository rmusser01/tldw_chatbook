---
id: TASK-625
title: Remove dead shadowed create_rag_service in rag_service.py
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 17:30'
updated_date: '2026-07-25 17:39'
labels:
  - rag
  - cleanup
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The simplified RAG package's public seam (tldw_chatbook.RAG_Search.simplified) resolves create_rag_service to rag_factory.create_rag_service, which is imported into __init__.py after (and instead of) rag_service.py's own same-named function. rag_service.py's create_rag_service has a different signature and is never imported anywhere in production or test code, making it unreachable dead code that only adds confusion and maintenance burden.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No same-named shadowed create_rag_service function remains reachable from rag_service.py
- [x] #2 All existing Tests/RAG tests pass
- [x] #3 A test or import-check locks the public seam so that from tldw_chatbook.RAG_Search.simplified import create_rag_service resolves to rag_factory's implementation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify (already done in research) that rag_service.py's create_rag_service is unreachable: grep all prod+test code for direct imports from the rag_service submodule; confirm simplified/__init__.py never imports create_rag_service from .rag_service (only from .rag_factory).
2. RED: add a seam-lock test file Tests/RAG/simplified/test_create_rag_service_seam.py asserting (a) the public seam tldw_chatbook.RAG_Search.simplified.create_rag_service is rag_factory's create_rag_service, and (b) the rag_service module no longer exposes a create_rag_service attribute at all. Run it to confirm it fails on (b) (RED).
3. Delete the dead create_rag_service function (and its 'Convenience functions' comment context if it becomes orphaned) from tldw_chatbook/RAG_Search/simplified/rag_service.py.
4. Run the new seam-lock test to confirm GREEN, then run the full Tests/RAG/ suite to confirm no regressions.
5. Commit as chore(rag): remove shadowed create_rag_service (task-625).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified via full grep across prod+tests that rag_service.py's create_rag_service was never imported anywhere except its own definition -- simplified/__init__.py only imports the same-named function from rag_factory (profile-based, different signature), so the rag_service.py version was 100% unreachable dead code, not merely shadowed-but-sometimes-used. TDD: added Tests/RAG/simplified/test_create_rag_service_seam.py (RED on the 'no shadowed function' assertion), then deleted the dead function from rag_service.py, replacing it with a short NOTE comment explaining why (GREEN). No reconciliation/delegation was needed since nothing reached the dead version. Full Tests/RAG/ suite: 554 passed / 8 skipped (baseline 552/8 plus the 2 new seam-lock tests), no regressions.

Modified/added files:
- tldw_chatbook/RAG_Search/simplified/rag_service.py (removed dead create_rag_service)
- Tests/RAG/simplified/test_create_rag_service_seam.py (new)
<!-- SECTION:NOTES:END -->
