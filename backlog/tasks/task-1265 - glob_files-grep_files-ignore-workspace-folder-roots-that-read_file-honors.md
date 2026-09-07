---
id: TASK-1265
title: glob_files/grep_files ignore workspace folder roots that read_file honors
status: Done
assignee: []
created_date: '2026-07-27 00:00'
labels: [agents, tools, security, tech-debt]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The sandboxed file tools disagree about what "the sandbox" means. `read_file`,
`write_file` and `list_directory` resolve against the sandbox root **plus the
run's bound workspace folder roots**:

```python
validate_path_multi(
    file_path,
    allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root()),
)
```

`glob_files` and `grep_files` never call `allowed_file_roots` at all. Both glob
and containment-check against the sandbox root alone:

```python
root = _tool_sandbox_root()
candidates = root.glob(pattern)
...
if not path.is_file() or not is_within(path, root, context=sensitive_ctx):
```

So a file in a bound workspace folder is readable by exact path but cannot be
found or searched. An agent that can read a workspace file has no way to discover
it, and a user who binds a project folder reasonably expects the search tools to
see it — the asymmetry is silent and surprising in both directions.

This was found while designing programmatic run memory
(`Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md`
§9.4), which routes around it rather than fixing it: correcting the asymmetry
widens two shared tools' reach for **every** caller, which is a security-relevant
change that deserves its own review rather than being folded into a feature spec.

Whichever way it is resolved, the two behaviours should agree. Note that
narrowing is also a legitimate outcome: if search tools are deliberately meant to
stay sandbox-only, that intent belongs in the tool descriptions and docstrings,
which currently say only "inside the tool sandbox" without acknowledging that
sibling tools reach further.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A decision is recorded (ADR or task notes) on whether search tools reach workspace folder roots, with the security reasoning stated
- [x] #2 `glob_files` and `grep_files` behave consistently with `read_file`/`list_directory` per that decision — either both reach workspace roots, or the sandbox-only scope is documented as deliberate in each tool's description
- [x] #3 If widened: containment is checked against every allowed root, not just the sandbox root, and a path outside all of them is still refused
- [x] #4 If widened: the existing hidden-component rule (`_is_hidden_within`) and sensitive-path denylist apply to workspace roots exactly as they do to the sandbox root
- [x] #5 If widened: `_MAX_CANDIDATES` and the other traversal bounds are re-evaluated against the larger search space, since a bound workspace folder can be far larger than the sandbox
- [x] #6 If widened: the `"reads"` risk tag and its `ask` permission floor are re-confirmed as still appropriate for the wider reach
- [x] #7 Symlink and mount drift on workspace roots is re-checked per call, matching `allowed_file_roots`' existing ADR-028 behaviour
- [x] #8 Tests cover a file in a bound workspace folder being found or refused per the decision, and a file outside all roots always being refused
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed as SUPERSEDED — not implemented under this id.

TASK-850 ("Scope glob_files and grep_files to workspace folder roots") shipped exactly
this change independently while the run-log branch was in flight. Both tools now resolve
their roots through `allowed_file_roots`:

- `GlobFiles` (file_operation_tools.py:897) -> :959
- `GrepFiles` (file_operation_tools.py:1433) -> :1562

so the asymmetry this task described — `read_file`/`write_file`/`list_directory` honouring
workspace folder roots while the search tools saw only the sandbox — no longer exists.

Consequence worth following: this task was filed while designing the run log, whose §9.4
routed AROUND the asymmetry rather than fixing it, and whose sandbox-vs-workspace directory
naming was justified by the search tools being unable to reach workspace folders. TASK-850
invalidated that premise, which reopened a sub-agent disclosure in the workspace case. That
is tracked as TASK-1270.
<!-- SECTION:NOTES:END -->
