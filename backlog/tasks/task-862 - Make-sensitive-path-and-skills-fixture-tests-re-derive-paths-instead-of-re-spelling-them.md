---
id: TASK-862
title: >-
  Make sensitive-path and skills-fixture tests re-derive paths instead of
  re-spelling them
status: To Do
assignee: []
created_date: '2026-07-27 04:35'
labels:
  - security
  - tools
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two existing test sites assert one of the app's security-relevant paths by re-typing a literal that happens to match today's code rather than deriving it from the same accessor the app uses, so they would go vacuous in lockstep with a future refactor instead of catching it. This is the exact defect class TASK-846 was opened to inventory, so its own test suite should not carry an instance of it.

Tests/Utils/test_sensitive_paths.py:73-75 re-spells the three MCP store filenames joined to get_user_data_dir(), the same way Utils/sensitive_paths.py:209-211 does internally -- both agree with unified_control_plane_service.py:2430/:2073 today only because store.path's parent happens to be get_user_data_dir(). If app.py:5241 ever moves local_mcp_store.json into a subdirectory, the test and the production module drift together and neither the parent == user_data_dir fallback nor this test would catch the reintroduced bug -- the test should instead derive its expected path from app.unified_mcp_service.permission_store.path (the live object), not from a re-typed app.py expression.

Tests/conftest.py:566-575 and Tests/Skills/test_skills_library_flow.py:84-106 build their skills-fixture paths the same re-spelling way rather than reading them off the real SkillTrustService/local_skills_service objects under test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/Utils/test_sensitive_paths.py's MCP-store-path assertions derive their expected paths from the live store objects (e.g. app.unified_mcp_service.permission_store.path) instead of re-typing the get_user_data_dir()-joined literal
- [ ] #2 Tests/conftest.py and Tests/Skills/test_skills_library_flow.py's skills-fixture paths derive from the real SkillTrustService/local_skills_service attributes instead of a re-spelled literal
- [ ] #3 Both updated tests still pass against the current, correct code
- [ ] #4 A deliberately introduced path change in the corresponding production accessor (temporarily, while verifying) causes each updated test to fail, confirming it actually re-derives rather than re-asserts
<!-- AC:END -->
