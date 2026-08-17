---
id: TASK-17067
title: Deduplicate workspace folder-root drift filtering in workspace_file_roots
status: To Do
assignee: []
created_date: '2026-08-17 19:59'
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
- [ ] #2 allowed_file_roots, folder_binding_roots, and workspace_context_note all consume that helper, each keeping its own post-filtering (allowed_file_roots rw/ro + sandbox; folder_binding_roots change-review gate)
- [ ] #3 Roots produced by all three call sites are identical to before the refactor; existing Tests/Tools/test_workspace_file_roots.py passes unchanged
- [ ] #4 The workspace-context note still matches the roots allowed_file_roots honors (no note/enforcement divergence)
<!-- AC:END -->
