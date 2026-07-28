---
id: TASK-1251
title: Workspace folder bound to the sandbox root re-exposes the run log
status: To Do
assignee: []
created_date: '2026-07-28 00:00'
labels: [agents, run-log, security]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent run log is written under a bound read-write workspace folder when one
exists, and under the tool sandbox root otherwise. Those two cases deliberately
use different directory names:

- **workspace folder** → `agent-runs` (undotted, so the log is a visible artifact
  in the user's own project)
- **sandbox fallback** → `.agent-runs` (dotted, so `_is_hidden_within` excludes it
  from `glob_files`/`grep_files`)

The dotting exists for a security reason. The sandbox root is exactly the root
`glob_files` and `grep_files` are rooted at, and sub-agents inherit the parent's
allow-list — so with an undotted name a spawned child could `grep_files` its
parent's entire run log, contradicting `spawn_subagent`'s contract that a child
"sees only the task text you pass". That was reproduced during review by
extracting a planted `PARENT_SECRET_API_KEY` from a parent's log, and is fixed
for the default configuration.

The choice is made from which branch `resolve_log_root` took, deliberately not by
comparing path strings. That leaves one edge case open: **nothing prevents a user
from binding a workspace folder whose resolved path IS the sandbox root**
(`Workspaces/registry_service.py` `add_folder_binding` has no such guard). When
that happens `resolve_log_root` takes the workspace branch, the writer picks the
undotted `agent-runs` name, and the log lands undotted *inside the sandbox root* —
reproducing exactly the disclosure the dotting prevents.

Reproduced during the final re-review of `feat/agent-run-log-spec`. It requires a
deliberate and unusual binding, which is why it was not treated as a merge
blocker, but it silently defeats a security control and should not stay open.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A workspace folder whose resolved path is the sandbox root (or lies inside it) cannot cause the run log to be written under an undotted, glob/grep-reachable name
- [ ] #2 The chosen remedy is stated and justified — either refusing/warning on such a binding at `add_folder_binding`, or having the writer detect the overlap and fall back to the dotted name
- [ ] #3 A sub-agent cannot reach the parent's run log via `glob_files` or `grep_files` in this configuration; the test plants a distinctive secret and asserts it is not recoverable
- [ ] #4 The normal cases are unaffected: a genuine workspace folder outside the sandbox still gets the visible undotted `agent-runs`, and the no-workspace fallback still gets the dotted name
- [ ] #5 The app's own reader (`search_run_log` / `load_records`) still reads the log in every case, since it deliberately does not route through `validate_path`
<!-- AC:END -->
