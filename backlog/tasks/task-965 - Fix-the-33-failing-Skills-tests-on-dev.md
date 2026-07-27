---
id: TASK-965
title: Fix the 33 failing Skills tests on dev
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:06'
updated_date: '2026-07-27 18:42'
labels:
  - skills
  - tests
  - dev-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Skills/ reports 33 failures on pristine origin/dev (33 failed / 342 passed). Three root causes were identified while triaging: current_runtime_backend is a read-only property while a test helper tries to set it; provider_model_resolution.py raises persisted_defaults must be a mapping; and one test does not pre-create a config parent directory. These masked signal repeatedly during the path-naming audit -- every branch touching Skills had to be separately baselined against pristine dev to prove its own failures were not among them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tests/Skills passes on a clean checkout,Each of the three root causes is fixed rather than worked around in the test,No test is relaxed merely to make it pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run Tests/Skills/ on the fast-forwarded worktree to get current failure counts before assuming the three triaged root causes still apply.
2. If failures reproduce, verify each of the three suspected causes (current_runtime_backend read-only property, provider_model_resolution mapping TypeError, missing config parent directory) against the actual tracebacks rather than assuming.
3. Fix each confirmed cause at its source; do not relax any assertion.
4. Re-run Tests/Skills/ to confirm.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Already fixed on current origin/dev -- no code change made here. Tests/Skills/ ran clean twice in a row on this worktree (379 passed, 0 failed, both runs; re-run for determinism since the task's own baseline of 33 failed/342 passed implied this suite was not previously green).

This worktree's branch started 26 commits behind origin/dev at task-assignment time and was fast-forwarded to catch up (per the task instructions, work must target current origin/dev). Among the commits picked up, ee49881d2 ('test: make sensitive-path and skills-fixture tests re-derive paths', TASK-866, landed same day after this task's triage) rewrote Tests/conftest.py's make_trust_service fixture and Tests/Skills/test_skills_library_flow.py's two trust-service builders to derive skills_dir/trust_dir from the real accessors and, as a necessary side effect of that rewrite, added trust_dir.mkdir(parents=True, exist_ok=True) before constructing the trust store. That is precisely root cause #3 (a test not pre-creating a config parent directory) -- fixed incidentally by an unrelated hygiene task before this one started.

Root causes #1 (current_runtime_backend read-only property) and #2 (provider_model_resolution's persisted_defaults-must-be-a-mapping) do NOT appear anywhere in Tests/Skills/ or in the production code it exercises -- grepped Skills_Interop/, the Skills UI screen, and Tests/Skills/ for both symbols with zero hits. Tests/Skills/ has no TldwCli(...) construction and no resolve_effective_provider_model call. Those two symptoms are real and DO reproduce, but in Tests/UI/test_tools_settings_window.py (TASK-966, this same batch) and other Tests/UI files (test_console_session_settings.py, test_study_*.py, test_runtime_policy_full_app.py) that construct a real TldwCli or call resolve_effective_provider_model directly -- not in Tests/Skills. Most likely explanation: the original triage ran a combined/larger batch of failures and misfiled two UI-suite causes under the Skills task.

No test relaxed, no assertion weakened -- nothing needed changing in Tests/Skills/ itself. Recommend closing TASK-944 (Console runtime-backend fixture repair, still To Do) as the right home for root cause #1 if it resurfaces elsewhere; root cause #2 was not independently reproduced in this pass and is not otherwise tracked.

Before: task claimed 33 failed / 342 passed. After (current dev, this worktree, x2 runs): 379 passed, 0 failed.
<!-- SECTION:NOTES:END -->
