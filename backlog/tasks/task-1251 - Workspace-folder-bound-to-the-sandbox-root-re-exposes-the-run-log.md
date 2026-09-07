---
id: TASK-1251
title: Workspace folder bound to the sandbox root re-exposes the run log
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 00:00'
updated_date: '2026-07-28 19:42'
labels:
  - agents
  - run-log
  - security
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
- [x] #1 A workspace folder whose resolved path is the sandbox root (or lies inside it) cannot cause the run log to be written under an undotted, glob/grep-reachable name
- [x] #2 The chosen remedy is stated and justified — either refusing/warning on such a binding at `add_folder_binding`, or having the writer detect the overlap and fall back to the dotted name
- [x] #3 A sub-agent cannot reach the parent's run log via `glob_files` or `grep_files` in this configuration; the test plants a distinctive secret and asserts it is not recoverable
- [x] #4 The normal cases are unaffected: a genuine workspace folder outside the sandbox still gets the visible undotted `agent-runs`, and the no-workspace fallback still gets the dotted name
- [x] #5 The app's own reader (`search_run_log` / `load_records`) still reads the log in every case, since it deliberately does not route through `validate_path`
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In RunLogWriter.bind(), after resolving `root`, compute containment against `_tool_sandbox_root()` directly (resolved-path comparison), independent of the `is_sandbox_fallback` branch flag resolve_log_root() reports.
2. OR the chosen root: dot the directory name whenever the flag says fallback OR the containment check says the resolved root is the sandbox root or nested inside it.
3. Fail closed (treat as sandboxed/dot) if the containment check itself raises.
4. Add tests: workspace folder nested inside the sandbox root, workspace folder equal to the sandbox root, and a regression guard that a genuine outside-sandbox workspace folder keeps the undotted name.
5. Add an end-to-end grep_files test mirroring the existing sandbox-fallback disclosure test, planting a secret and confirming it is unreachable through this workspace-inside-sandbox binding too (AC #3).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in tldw_chatbook/Agents/run_log.py: RunLogWriter.bind() now decides the dotted-vs-undotted name on ACTUAL containment (resolved root == or under `_tool_sandbox_root()`), not merely on which branch resolve_log_root() took. The check is skipped (cheap short-circuit) when the fallback flag already says True, and runs otherwise; any exception during the check fails closed (dots the name) rather than risking disclosure. This also covers callers that monkeypatch resolve_log_root() wholesale, since the check reads the already-resolved `root` value directly rather than the side-channel flag.

Tests added: Tests/Agents/test_run_log_writer.py (nested-inside-sandbox, equal-to-sandbox, and outside-sandbox-stays-undotted regression guard) and Tests/Agents/test_run_log_sandbox_isolation.py::test_workspace_folder_inside_the_sandbox_is_dotted_and_hidden_from_grep, which mirrors the existing PARENT_SECRET_API_KEY disclosure test but through a workspace-folder binding nested inside the sandbox root, driving the real GrepFiles tool and confirming the secret is unrecoverable (AC #3). search_run_log/load_records' own reader is unaffected since it globs log_dir directly and never routes through validate_path/_is_hidden_within (AC #5), unchanged by this fix.

Addressed as part of the PR #1066 Qodo review pass (finding F8) rather than as a separate follow-up PR, per reviewer instruction to fix everything before merge.
<!-- SECTION:NOTES:END -->
