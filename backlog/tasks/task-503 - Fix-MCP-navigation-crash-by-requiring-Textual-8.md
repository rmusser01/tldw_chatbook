---
id: TASK-503
title: Fix MCP navigation crash by requiring Textual 8
status: Done
assignee: []
created_date: '2026-07-23 22:51'
updated_date: '2026-07-23 23:32'
labels: []
dependencies: []
references:
  - backlog/decisions/022-textual-8-runtime-floor.md
documentation:
  - Docs/superpowers/specs/2026-07-23-mcp-textual-runtime-floor-design.md
  - Docs/superpowers/plans/2026-07-23-mcp-textual-runtime-floor.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the application from crashing when users navigate to the MCP destination after installing a Textual version that predates the Select.NULL API used by the MCP UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The MCP destination mounts without a Select.NULL AttributeError
- [x] #2 A regression test prevents lowering the Textual runtime floor below 8.0.0
- [x] #3 Focused MCP tests pass on the supported runtime
- [x] #4 Package metadata and development requirements support Textual 8.x (>=8.0.0,<9)
- [x] #5 CI runs the focused MCP runtime suites against exactly Textual 8.0.0
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the accepted Textual 8.x runtime boundary in ADR-022 and the approved design/implementation documents.
2. Add red semantic regressions for both dependency manifests and update the existing Phase 6 packaging seam.
3. Change pyproject.toml and requirements.txt to textual>=8.0.0,<9, add the Unreleased changelog notice, and make the dependency tests green.
4. Add a red workflow contract, then a focused GitHub Actions lane that installs Textual 8.0.0 and runs the MCP workbench/tools suites.
5. Verify packaging, workflow, and MCP suites on the normal and exact-minimum runtimes; run diff hygiene; record evidence and close the task.

ADR required: yes
ADR path: backlog/decisions/022-textual-8-runtime-floor.md
Reason: this changes the supported framework/runtime policy, ends Textual 3-7 support, and fails closed on unreviewed future Textual major versions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the dependency-boundary repair without changing MCP runtime code. The authoritative pyproject.toml dependency and the CI/developer requirements.txt mirror now require Textual 8.x (>=8.0.0,<9), with an Unreleased changelog upgrade notice and contributor guides aligned to the same contract. Added Tests/CI/test_textual_runtime_contract.py to parse both manifests and require the exact SpecifierSet; updated the existing Phase 6 packaging seam. Added a contract-tested textual-minimum GitHub Actions job that installs exactly Textual 8.0.0, runs the packaging contract plus MCP workbench/tools-mode suites, and participates in aggregate workflow status. ADR-022 records the bounded-major runtime policy and rejected compatibility shims/open-ended future-major support.

Verification: observed the dependency tests RED first (3 expected failures against the old declarations); post-review normal-runtime suite passed 189 tests with 2 pre-existing MCPWorkbench._clear_tool_view coroutine warnings; exact Textual 8.0.0 replay passed 184 tests with the same 2 warnings; workflow YAML parsed and all workflow contract tests passed; wheel build succeeded with Requires-Dist: textual<9,>=8.0.0; compileall and git diff --check passed. Independent code review found no Critical issues and no production-code defects; its documentation and test-hardening findings were applied.

PR rebase note: renumbered from TASK-400 to TASK-503 after rebasing onto `dev`, where TASK-400 already identifies the Console staged-sources relocation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
MCP navigation installs can no longer resolve unsupported Textual versions: both install manifests are bounded to Textual 8.x and CI mounts the MCP path on exactly 8.0.0.
<!-- SECTION:FINAL_SUMMARY:END -->
