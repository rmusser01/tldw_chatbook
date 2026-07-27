---
id: TASK-857
title: >-
  Make workspace folder-binding consult the sensitive-path denylist, not just
  home/root
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
Workspaces/registry_service.py:673-707 validates a folder root to bind as a workspace. Its docstring (:666-669) claims it denies "the filesystem root, the home directory itself, non-directories, and duplicate/nested roots", but the implementation only checks two exact equalities: resolved == Path(resolved.anchor) (:683) and resolved == Path.home().resolve() (:688). There is no reference to Utils/sensitive_paths anywhere under tldw_chatbook/Workspaces/. As a result, binding ~/.config/tldw_cli (the live config, API keys included), ~/.local/share/tldw_cli (every app database), or ~/.ssh as a workspace folder root all pass this check. Because a bound folder root widens what the agent file tools may reach, this is the mechanism by which the uncovered skill trust store (and any other path the denylist doesn't individually enumerate) becomes reachable in practice: is_sensitive_path() still refuses the specific enumerated files once inside such a root, so this is scope-widening up to the edge of what the denylist enumerates, not a direct bypass of an individual file check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Folder-binding validation rejects a candidate root that is, or resolves inside, any path flagged by is_sensitive_path() / the app's sensitive-path containers (user data dir, effective config dir, etc.), not just the filesystem root and home directory
- [ ] #2 A test attempts to bind ~/.config/tldw_cli, get_user_data_dir(), and a subdirectory of get_user_data_dir() as workspace folder roots (derived from the real accessors, not literal strings) and confirms all are rejected
- [ ] #3 A test confirms an ordinary, non-sensitive folder root still binds successfully (no regression on the common case)
<!-- AC:END -->
