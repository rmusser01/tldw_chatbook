---
id: TASK-19632
title: >-
  git_diff leaks denylisted file content when path is omitted
status: To Do
assignee: []
created_date: '2026-08-21 12:05'
labels:
  - security
  - agents
  - tools
priority: high
dependencies: []
---

## Description

Found and reproduced during TASK-19551 (which closed the same class of hole for
the `fs_*` tools), then re-verified by that task's reviewer. TASK-19551 made
`Tools/local_tool_impls.py::resolve_workspace_path` enforce the sensitive-path
denylist (`Utils/sensitive_paths.py`) for every path a model NAMES, and made the
three enumerating `fs_*` tools (`fs_list`/`fs_glob`/`fs_grep`) filter the
entries they present but the model never named.

The `git_*` tools share that choke point for their path argument, but have no
equivalent output filter — and on three of them `path` is OPTIONAL. When it is
omitted, no candidate path reaches the denylist at all: the seam is explicit at
`Agents/local_tool_provider.py:444-445`, where `path_targets` returns the
repository root and stops:

```
if raw_path is None:
    return (ToolPathTarget(path=repo_root, kind="repository"),)
```

git then enumerates the whole repository on the tool's behalf and the tool
returns its output verbatim. Under the shipped `workspace_root` default (the
app's cwd at startup), an app launched from `$HOME` inside a git repository puts
`~/.ssh/id_rsa` in that enumeration.

Measured on this branch with an isolated `$HOME` that is a git repo containing a
synthetic `.ssh/id_rsa` (probe re-run twice; second run added the clean-tree
case):

| call | leaks |
| --- | --- |
| `git_diff(commit_range="HEAD~1..HEAD")`, no `path` | **file CONTENT**, on a CLEAN worktree |
| `git_diff()`, no `path`, dirty worktree | **file CONTENT** |
| `git_diff(stat=True)`, no `path` | file NAME only |
| `git_status()`, no `path`, dirty worktree | file NAME only |
| `git_log()`, no `path` | nothing — commit metadata only |
| any of the above WITH `path=".ssh/id_rsa"` | nothing; refused "protected path" (TASK-19551) |

Two properties make this worse than the `fs_grep` gap TASK-19551 closed:

1. **No write primitive is required.** The `commit_range` form reads the
   credential out of history, so a read-only agent on a clean checkout is
   enough. An earlier draft of this finding claimed a dirty worktree was
   needed; that is wrong.
2. It is reachable by **prompt injection** from fetched web content, exactly
   like the hole TASK-19551 closed — the model only has to call a read-only,
   `reads`-tagged git tool with no arguments.

Scope note: `git_log` is clean and `git_status` discloses names only. The fix
should not be described or built as "the three git tools leak".

## Acceptance Criteria

- [ ] `git_diff` never returns the content of a path `is_sensitive_path` refuses,
      whether that path is reached via the worktree, the index, or a
      `commit_range`, and whether or not the caller supplied `path`
- [ ] `git_status` never names such a path
- [ ] The behaviour is enforced by construction (e.g. denylisted paths inside the
      repository are excluded from git's own output via pathspec, or the output
      is filtered before it is returned) rather than by asking the model to pass
      `path`
- [ ] A born-red test reproduces the clean-worktree `commit_range` content leak
      and the `stat=True`/`git_status` name leaks, and each is refused after
      the fix
- [ ] A test pins that `git_log` output is unchanged, and that ordinary
      (non-denylisted) diffs, stats and statuses are unchanged — the fix must not
      silently truncate legitimate multi-file diffs
- [ ] `Utils/sensitive_paths.py`'s module docstring and
      `Tools/local_tool_impls.py`'s drop the TASK-19632 exception once it no
      longer applies (both currently state it explicitly)

## Notes

The two obvious implementations both need care, which is why this is its own
task rather than a rider on TASK-19551:

* **Pathspec exclusion** (`git diff -- . ':(exclude)<relpath>'`) keeps git the
  authority and never parses diff text, but the exclusions must be computed
  per call from the resolved denylist, restricted to paths inside the
  repository root, and rendered repo-relative; `run_git`'s argv allowlist
  (`_validate_argv`) has to accept them.
* **Output filtering** means parsing unified diff / porcelain v2 text, and a
  half-parsed diff is worse than none.

`Tests/Tools/test_local_tool_sensitive_paths.py::
test_every_workspace_rooted_function_uses_the_choke_point` already covers
`git_tool_impls` structurally and carries a NOTE that reaching the choke point
proves the path ARGUMENT is checked, not that output is filtered — that note
points here.
