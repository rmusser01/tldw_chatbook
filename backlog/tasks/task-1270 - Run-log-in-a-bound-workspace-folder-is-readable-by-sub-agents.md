---
id: TASK-1270
title: Run log in a bound workspace folder is readable by sub-agents
status: To Do
assignee: []
created_date: '2026-07-28 00:00'
labels: [agents, run-log, security]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent run log picks its directory name based on where the log lands:

- **sandbox fallback** → `.agent-runs` (dotted, so `_is_hidden_within` excludes it from
  `glob_files`/`grep_files`)
- **bound workspace folder** → `agent-runs` (undotted)

The dotting exists to stop a spawned sub-agent — which inherits its parent's allow-list —
from reading the parent's entire run log through the generic file tools. That was
reproduced during review of PR #1066 by extracting a planted `PARENT_SECRET_API_KEY`, and
fixed for the sandbox case.

The undotted workspace name was justified by a premise recorded in the design spec §9.4 and
in `run_log.py`'s own comments: that `glob_files`/`grep_files` glob `_tool_sandbox_root()`
alone and cannot reach workspace folder roots at all.

**That premise is no longer true.** TASK-850 ("Scope glob_files and grep_files to workspace
folder roots") is Done, and both tools now resolve their roots through `allowed_file_roots`:

```
GlobFiles  (class at file_operation_tools.py:897)  → :959   allowed_file_roots(write=False, ...)
GrepFiles  (class at file_operation_tools.py:1433) → :1562  allowed_file_roots(write=False, ...)
```

So when a workspace folder is bound, the run log is written undotted into a root those tools
now search, and a sub-agent can `grep_files` its parent's history — the same disclosure the
dotting was introduced to prevent, reopened for the workspace case by an unrelated change
landing on dev.

The existing isolation tests do not catch this because they only exercise the
sandbox-fallback configuration.

Note the trade-off behind the original undotted choice: the log was meant to be a
user-visible artifact in the user's own project. A dotted directory remains fully visible to
the *user* (it is an ordinary directory, listed by `ls -a` and shown by editors); it is
hidden only from the app's own sandboxed file tools, which is precisely the intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A sub-agent cannot read its parent's run log via `glob_files` or `grep_files` when a workspace folder is bound, proven by a test that plants a distinctive secret and asserts it is not recoverable
- [ ] #2 The same protection holds in the sandbox-fallback configuration (no regression of the existing behaviour)
- [ ] #3 The remedy does not depend on which branch `resolve_log_root` took, so a future change to root resolution cannot silently reopen it
- [ ] #4 The app's own reader (`search_run_log` / `load_records`) still reads the log in every configuration, since it deliberately does not route through `validate_path`
- [ ] #5 The log remains reachable by the user as an ordinary file, and the design spec §9.1/§9.4 and `run_log.py`'s comments are corrected to state the current premise rather than the stale one
- [ ] #6 A regression test pins the invariant against the specific change that caused this — i.e. it fails if `glob_files`/`grep_files` can reach the log directory, regardless of how their root resolution is implemented
<!-- AC:END -->
