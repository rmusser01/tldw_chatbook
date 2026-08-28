---
id: TASK-17067
title: Deduplicate workspace folder-root drift filtering in workspace_file_roots
status: Done
assignee:
  - '@codex'
created_date: '2026-08-17 19:59'
updated_date: '2026-08-28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The is_dir + symlink/self-resolve drift filter for workspace folder bindings is duplicated across three functions in tldw_chatbook/Tools/workspace_file_roots.py (allowed_file_roots, folder_binding_roots, workspace_context_note). A future change to the drift/symlink policy must be made in three places or they silently diverge, which would make the agent-facing workspace-context note advertise roots the file tools actually reject (or vice versa). Extract one shared iterator so the invariant lives in a single place. Deferred from PR #1765 (workspace-context note) because consolidating touches the security-sensitive, divergently-logged allowed_file_roots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A single shared helper in workspace_file_roots.py yields the existing, non-drifted (non-symlink, self-resolving) folder bindings for a workspace
- [x] #2 allowed_file_roots, folder_binding_roots, and workspace_context_note all consume that helper while preserving caller-specific ordering: allowed_file_roots applies rw/ro filtering before shared filesystem validation and still prepends the sandbox; folder_binding_roots applies its change-review gates before registry/binding validation
- [x] #3 Roots produced by all three call sites are identical to before the refactor; existing Tests/Tools/test_workspace_file_roots.py passes unchanged
- [x] #4 The workspace-context note still matches the roots allowed_file_roots honors (no note/enforcement divergence)
- [x] #5 Each consumer reports an existing symlinked directory or self-resolve mismatch with the exact path-free warning `Workspace folder binding excluded because its path no longer resolves to itself (symlink or mount drift)`; the raw locator is absent, while missing directories and broken symlinks remain silent
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/028-settings-workspaces-category-and-folder-roots.md
Reason: ADR-028 already owns call-time validation and run-bound folder authority; this task consolidates an existing rule without changing storage, permissions, ownership, or an external contract.

Approved design: Docs/superpowers/specs/2026-08-27-task-17067-workspace-root-drift-filter-design.md
Detailed implementation plan: Docs/superpowers/plans/2026-08-27-task-17067-workspace-root-drift-filter-implementation.md

1. Add red-first routing, ordering, and exact path-free warning regressions.
2. Add one caller-prefiltered validity iterator and route all three consumers through it.
3. Run focused behavior tests and regenerate the reviewed persistent-diagnostic inventory.
4. Run scoped static/diff checks, final code review, and complete Backlog task hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Added one caller-prefiltered `_iter_valid_folder_bindings` helper shared by `allowed_file_roots`, `folder_binding_roots`, and `workspace_context_note`.
- Preserved write-access prefiltering, global/workspace change-review gates, sandbox ordering, rendering, and caller whole-operation fallbacks. Existing symlinked or self-resolve-drifted directories emit the exact path-free warning `Workspace folder binding excluded because its path no longer resolves to itself (symlink or mount drift)`; missing directories and broken symlinks remain silent.
- Regenerated the persistent diagnostic inventory after rebasing. Verification: new behavior selection, 9 passed; post-rebase `Tests/Tools/test_workspace_file_roots.py`, 36 passed; post-rebase change-review selection, 4 passed; post-rebase inventory architecture, 65 passed. The inventory checker verified 540 owners, 1260 TASK-492 calls, 7392 TASK-494 calls, and 8 sink files. Ruff check, Ruff format `--check`, `compileall`, `git diff --check`, and `git diff --check origin/dev` all passed. No full suite was run or claimed.
- Rebased onto `dev` at `a79d9e62c33bc43d91df93758465d94fc86563d4`; the branch was 0 behind. Final whole-change review found no Critical, Important, or Minor issues and marked the task ready for closeout.
- Warnings and teardown logging occurred only after successful pytest summaries and were identified as unrelated pre-existing cleanup/background-snapshot noise.
- ADR required: no; ADR-028 remains applicable. No lessons file was updated because implementation surfaced no new reusable repository incident beyond the existing documented generated-artifact/rebase practice.
