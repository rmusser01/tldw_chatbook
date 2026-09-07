---
id: TASK-857
title: >-
  Make workspace folder-binding consult the sensitive-path denylist, not just
  home/root
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:35'
updated_date: '2026-07-27 17:43'
labels:
  - security
  - tools
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspaces/registry_service.py:673-707 validates a folder root to bind as a workspace. Its docstring (:666-669) claims it denies "the filesystem root, the home directory itself, non-directories, and duplicate/nested roots", but the implementation only checks two exact equalities: resolved == Path(resolved.anchor) (:683) and resolved == Path.home().resolve() (:688). There is no reference to Utils/sensitive_paths anywhere under tldw_chatbook/Workspaces/. As a result, binding ~/.config/tldw_cli (the live config, API keys included), ~/.local/share/tldw_cli (every app database), or ~/.ssh as a workspace folder root all pass this check. Because a bound folder root widens what the agent file tools may reach, this is the mechanism by which the uncovered skill trust store (and any other path the denylist doesn't individually enumerate) becomes reachable in practice: is_sensitive_path() still refuses the specific enumerated files once inside such a root, so this is scope-widening up to the edge of what the denylist enumerates, not a direct bypass of an individual file check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Folder-binding validation rejects a candidate root that is, or resolves inside, any path flagged by is_sensitive_path() / the app's sensitive-path containers (user data dir, effective config dir, etc.), not just the filesystem root and home directory
- [x] #2 A test attempts to bind ~/.config/tldw_cli, get_user_data_dir(), and a subdirectory of get_user_data_dir() as workspace folder roots (derived from the real accessors, not literal strings) and confirms all are rejected
- [x] #3 A test confirms an ordinary, non-sensitive folder root still binds successfully (no regression on the common case)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the gap: bind get_user_data_dir(), the effective config directory, and ~/.ssh as workspace folder roots against origin/dev and confirm add_folder_binding accepts all three.
2. Add a containment-aware helper in Utils/sensitive_paths.py (find_root_binding_conflict) that answers the question the binding gate actually needs -- not "is this one candidate sensitive" (is_sensitive_path) but "would granting recursive access under this root reach a protected path", in both directions (root is/is-nested-in a protected dir, or root contains one), reusing the same resolved sensitive-path context so it can never drift from the read-time denylist.
3. Wire add_folder_binding to consult it and raise a clear, actionable WorkspaceRegistryServiceError naming the specific conflicting path.
4. Confirm the same three scenarios are now rejected, and that an ordinary project folder still binds.
5. Add unit tests for the new helper (both containment directions, priority/tie-breaking when several protected paths match) and integration tests on add_folder_binding (denylist paths derived from real accessors, rejection message content, ordinary-folder regression check).
6. Run Tests/Utils/, Tests/Workspaces/, Tests/DB/, and the workspace-file-roots tests in Tests/Tools/ to confirm no regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced first (see report): binding get_user_data_dir(), the effective config directory (~/.config/tldw_cli), and ~/.ssh as workspace folder roots all succeeded on origin/dev via add_folder_binding -- confirmed the audit's claim that only the filesystem-root/home-directory checks existed, with no reference to Utils.sensitive_paths anywhere under Workspaces/. All three are rejected after this change; an ordinary project folder still binds.

Added Utils/sensitive_paths.find_root_binding_conflict(root, context=None) -> Path | None: reuses the same SensitivePathContext (dirs, direct_child_denied_dirs, files, db_paths) the read-time is_sensitive_path already resolves, so the binding gate can never enumerate a different set of protected paths than file-tool reads do. It answers a different question than is_sensitive_path though: not "is this one candidate sensitive" but "would granting recursive, blanket access under this root reach a protected path", checked in three passes in priority order: (1) root IS a protected path itself, (2) root is nested INSIDE a protected directory (tie-broken toward the deepest/closest enclosing one), (3) root CONTAINS a protected directory or file (tie-broken toward the shallowest/closest contained one, so an ancestor of get_user_data_dir() is reported as containing user_data_dir itself, not an obscure subtree three levels further down).

Containment rule chosen: reject a root that IS, is NESTED INSIDE, or CONTAINS any of: the fixed sensitive directories (~/.ssh et al + the skill-trust subtree), this app's own state-container directories (get_user_data_dir(), the effective config directory, the ChromaDB persist directory, the RAG-profile directory), or its sensitive single files/DB paths (+ WAL/SHM/journal sidecars). This deliberately goes further than is_sensitive_path's own per-path rule: is_sensitive_path's direct-child-file exemption for existing directories (e.g. tool_sandbox nested under get_user_data_dir()) exists to keep specific, already-known-good containers reachable for individual file reads -- it was never meant to make an entire application-state directory safe to grant as a BINDING ROOT, since that hands over everything else nested inside it too, known-good or not. Concretely: binding get_user_data_dir() itself, or ANY subdirectory of it (not just the specific ones the direct-child rule would flag), is refused -- nothing legitimate needs to bind it directly, since Tools.workspace_file_roots.allowed_file_roots already includes the sandbox root automatically without ever going through this gate. The reverse direction (binding a coarse ancestor like ~/.local/share or ~/.config, which merely *contains* one of these directories) is refused via the same rule, which is what keeps this from degenerating into "just refuse everything below home": nothing hardcodes home as a boundary, it falls out naturally from containment of the concrete protected paths, most of which live under home in practice.

Left unchanged (per the task): what already-bound folders do at read time. is_sensitive_path's own per-path checks (and the direct-child-file exemption) are untouched; this fix is scoped to the binding gate only. Any folder bound before this fix that happens to be, or contain, a protected path is NOT retroactively unbound or re-validated -- an existing binding could still be unsafe until removed and re-added, or until a future task adds binding re-validation. Not addressed here to keep this change scoped to the acceptance criteria.

The rejection message names the specific conflicting path directly (not just "a sensitive path exists somewhere"), and is surfaced verbatim to the user via UI/Screens/settings_screen.py's existing WorkspaceRegistryServiceError handler (already displays str(exc) in the workspaces pane) -- no UI change needed, satisfying "not a silent failure or bare exception" for free.

Modified/added files:
- tldw_chatbook/Utils/sensitive_paths.py (new find_root_binding_conflict)
- tldw_chatbook/Workspaces/registry_service.py (add_folder_binding consults it, raises a message naming the protected path)
- Tests/Utils/test_sensitive_paths.py (unit tests for the new helper: is/nested-in/contains, tie-breaking, ordinary-folder no-conflict)
- Tests/Workspaces/test_workspace_folder_bindings.py (integration tests: rejects ~/.config/tldw_cli, get_user_data_dir(), a subdirectory of it, and an ancestor of it -- all derived from real accessors; rejection message names the path; ordinary folder still binds); one pre-existing test (test_add_folder_binding_validation_matrix) updated to bind a carved-out subdirectory instead of tmp_path itself, since this suite's own HOME-redirection fixture nests its fake config directory under tmp_path, which is now correctly rejected in its own right
<!-- SECTION:NOTES:END -->
