---
id: TASK-866
title: >-
  Make sensitive-path and skills-fixture tests re-derive paths instead of
  re-spelling them
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:35'
updated_date: '2026-07-27 16:30'
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
- [x] #1 Tests/Utils/test_sensitive_paths.py's MCP-store-path assertions derive their expected paths from the live store objects (e.g. app.unified_mcp_service.permission_store.path) instead of re-typing the get_user_data_dir()-joined literal
- [x] #2 Tests/conftest.py and Tests/Skills/test_skills_library_flow.py's skills-fixture paths derive from the real SkillTrustService/local_skills_service attributes instead of a re-spelled literal
- [x] #3 Both updated tests still pass against the current, correct code
- [x] #4 A deliberately introduced path change in the corresponding production accessor (temporarily, while verifying) causes each updated test to fail, confirming it actually re-derives rather than re-asserts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rewrite Tests/Utils/test_sensitive_paths.py's MCP-permission-store tests to derive the expected path from a live UnifiedMCPControlPlaneService.permission_store/.execution_log property, not a re-typed Path(store.path).with_name(...) literal.
2. Rewrite Tests/conftest.py's make_trust_service fixture and Tests/Skills/test_skills_library_flow.py's two trust-service builders to derive skills_dir/trust_dir from LocalSkillsService(...).skills_dir and default_trust_store_dir(), not re-spelled 'skills'/'trust' literals.
3. Run both suites to confirm they still pass against current, correct code.
4. Deliberately break the corresponding production derivation (temporarily) and confirm the updated tests fail, then revert.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Tests/Utils/test_sensitive_paths.py's two MCP-store tests re-typed 'mcp_permissions.json'/Path(store.path).with_name(...) the same way unified_control_plane_service.py's permission_store/execution_log properties do internally, rather than reading the paths off a live instance -- rewrote both to build a real UnifiedMCPControlPlaneService (the SimpleNamespace(store=...) idiom already used in Tests/MCP/test_control_plane_permissions.py) with a default-path LocalMCPStore() (TASK-855 made this default itself derive from get_user_data_dir(), so no literal filename is spelled at all now), and assert on service.permission_store.path / service.execution_log.path / service.local_service.store.path directly.

Tests/conftest.py's make_trust_service fixture and Tests/Skills/test_skills_library_flow.py's _real_trust_service/_real_uninitialized_trust_service builders hardcoded tmp_path/'skills' and tmp_path/'trust' -- matching, by re-spelling, LocalSkillsService's private _SKILLS_DIRNAME and skill_trust_store's private _TRUST_DIRNAME constants. Rewrote both to derive skills_dir from LocalSkillsService(store_dir=tmp_path).skills_dir (a real, side-effect-free constructor call solely to read its computed attribute) and trust_dir from the already-public default_trust_store_dir(tmp_path) -- the exact function app.py itself calls.

Verified AC #4 concretely for both fixes, not just asserted: (1) for the MCP-store test, temporarily changed unified_control_plane_service.py's permission_store property to nest the file under an extra subdirectory -- the updated test correctly FAILED (is_sensitive_path no longer covers the new location), then reverted cleanly (git diff clean). (2) for the skills fixture, temporarily renamed local_skills_service.py's _SKILLS_DIRNAME and skill_trust_store.py's _TRUST_DIRNAME to different values -- the NEW fixture code still passed all 52 tests (correctly re-derives regardless of the constant's name), then reverting ONLY the test file back to the old re-spelled-literal style (while keeping the renamed production constants) reproduced 3 real failures, proving the old style really would have gone silently stale; reverted everything cleanly afterward (git diff clean on all production files touched during verification).

Files: Tests/Utils/test_sensitive_paths.py, Tests/conftest.py, Tests/Skills/test_skills_library_flow.py.
<!-- SECTION:NOTES:END -->
