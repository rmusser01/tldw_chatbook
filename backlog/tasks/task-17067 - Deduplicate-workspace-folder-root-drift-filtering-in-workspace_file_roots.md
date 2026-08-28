---
id: TASK-17067
title: Deduplicate workspace folder-root drift filtering in workspace_file_roots
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-17 19:59'
updated_date: '2026-08-28 05:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The is_dir + symlink/self-resolve drift filter for workspace folder bindings is duplicated across three functions in tldw_chatbook/Tools/workspace_file_roots.py (allowed_file_roots, folder_binding_roots, workspace_context_note). A future change to the drift/symlink policy must be made in three places or they silently diverge, which would make the agent-facing workspace-context note advertise roots the file tools actually reject (or vice versa). Extract one shared iterator so the invariant lives in a single place. Deferred from PR #1765 (workspace-context note) because consolidating touches the security-sensitive, divergently-logged allowed_file_roots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single shared helper in workspace_file_roots.py yields the existing, non-drifted (non-symlink, self-resolving) folder bindings for a workspace
- [ ] #2 allowed_file_roots, folder_binding_roots, and workspace_context_note all consume that helper while preserving caller-specific ordering: allowed_file_roots applies rw/ro filtering before shared filesystem validation and still prepends the sandbox; folder_binding_roots applies its change-review gates before registry/binding validation
- [ ] #3 Roots produced by all three call sites are identical to before the refactor; existing Tests/Tools/test_workspace_file_roots.py passes unchanged
- [ ] #4 The workspace-context note still matches the roots allowed_file_roots honors (no note/enforcement divergence)
- [ ] #5 Each consumer reports an existing symlinked directory or self-resolve mismatch with the exact path-free warning `Workspace folder binding excluded because its path no longer resolves to itself (symlink or mount drift)`; the raw locator is absent, while missing directories and broken symlinks remain silent
<!-- AC:END -->
