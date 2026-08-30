---
id: TASK-19504
title: Bind Console local path tools to run-admitted workspace roots
status: Done
assignee: []
created_date: '2026-08-21'
updated_date: '2026-08-30 16:10'
labels:
  - console
  - tools
  - security
dependencies:
  - TASK-17067
  - TASK-19637
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the in-app Console's hidden configured-root and process-CWD fallback so local filesystem and Git tools operate only on workspace bindings admitted for that run, while preserving ADR-069's selected-root behavior and unrelated local tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Project-instruction-enabled sessions preserve ADR-069's one selected binding and call-time membership, fingerprint, access, and filesystem-identity checks
- [x] #2 Disabled named workspaces capture their valid bindings at run admission and select them through stable binding-ID aliases without active-workspace retargeting
- [x] #3 Multiple admitted roots require an explicit root alias; reads honor ro or rw and mutations require current rw
- [x] #4 Binding removal, locator retargeting, identity replacement, and rw-to-ro downgrade revoke access during a run
- [x] #5 Default and binding-less named workspaces remove only local fs and Git schemas while preserving local web, Watchlists, and todo tools and built-in sandbox file access
- [x] #6 The in-app Console ignores configured workspace_root and process CWD; standalone MCP retains its explicit configured root
- [x] #7 The schema change invalidates stale persistent approvals through the existing definition-hash guard and the upgrade copy discloses reapproval
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: `backlog/decisions/102-console-run-admitted-local-path-authority.md`
Reason: this task changes the Console security authority and supersedes ADR-069's
disabled-session fallback behavior.

Approved design:
`Docs/superpowers/specs/2026-08-30-task-19504-run-admitted-workspace-roots-design.md`

Detailed plan:
`Docs/superpowers/plans/2026-08-30-task-19504-run-admitted-workspace-roots.md`

1. Pin zero/one/multiple-root schemas and runtime revocation with failing tests.
2. Add one immutable run-admitted root contract and alias-aware routing over the
   existing local provider, Virtual CLI registry, and ADR-101 executor.
3. Capture roots from the owning session workspace without active-workspace,
   configured-root, process-CWD, or scratch-backed structured-tool fallback.
4. Preserve project-instruction selection, non-path local tools, built-in scratch
   tools, standalone MCP, definition-hash invalidation, and upgrade guidance.
5. Run focused/static/diff/diagnostic verification and final review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-102 with one immutable run-admitted root contract shared by the
structured local provider and Virtual CLI. Console composition now captures the
owning named Workspace's valid bindings once per run, uses stable binding-ID
aliases, removes path schemas when no root is admitted, and revalidates binding
membership, locator, identity, and access before review and execution. ADR-069's
selected project root, non-path local tools, built-in scratch tools, and standalone
MCP authority remain unchanged. Local and Virtual CLI permission descriptors now
include the alias schema so the existing definition hash forces reapproval.

Verification covered the complete 329-test affected-file sweep in an isolated
Python 3.12 environment, including production subprocess-worker execution, plus
15 built-in scratch tests and 75 documentation contract tests. Ruff, formatting,
`py_compile`, and `git diff --check` passed. The shared development venv was not
modified; its editable install still points at an unrelated older worktree.

During PR integration, the repository-wide diagnostic-inventory gate exposed
new persona and Workspace diagnostics inherited from the updated `dev` base that
still interpolated private runtime values. Those messages now retain only their
failure categories, an AST regression test enforces constant metadata-only
templates at the affected boundaries, and the reviewed derived inventory was
regenerated. The focused privacy/logging suite passed 121 tests, the exact
inventory checker passed with 546 owners and eight sink files, and the original
329-test TASK-19504 sweep remained green.
<!-- SECTION:NOTES:END -->
