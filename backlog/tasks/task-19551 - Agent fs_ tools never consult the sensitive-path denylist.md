---
id: TASK-19551
title: >-
  Agent fs_* tools never consult the sensitive-path denylist
status: To Do
assignee: []
created_date: '2026-08-21 20:01'
labels:
  - security
  - agents
  - tools
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 1
#1**. CONFIRMED by the lane, and **independently re-confirmed by the review
controller** and again at this branch base.

The `fs_*` agent file-tool family never calls `is_sensitive_path`. The entire
guard on the agent-supplied path is:

```
tldw_chatbook/Tools/local_tool_impls.py:53
    return validate_path(path, workspace_root, allow_hidden=True)
```

`resolve_workspace_path` is the single choke point the seven `fs_*` tools
funnel through (dispatched from `tldw_chatbook/Agents/local_tool_provider.py`).
A `grep is_sensitive_path` across `Tools/` and `Agents/` returns **9 hits in
`Tools/file_operation_tools.py` and 1 in `Agents/run_log.py` — and ZERO in
`local_tool_impls.py`, `patch_tool_impls.py`, or `local_tool_provider.py`**.
`Utils/sensitive_paths.py` names only the five `file_operation_tools`
enforcers as the contract's enforcement points; the `fs_*` family postdates
that contract and never joined it.

Paths the lane proved are accepted without ever asking — **all of them already
in the denylist**: `~/.ssh/id_rsa`, `~/.config/tldw_cli/config.toml`,
`~/.aws/credentials`, `mcp_permissions.json`, `chachanotes.db`.

Two consequences, and the second is the severe one:

1. **Credential read.** Keys are read into the agent transcript, which is then
   sent to a model provider.
2. **One-step permission-gate bypass.** `fs_write`/`fs_patch` can rewrite
   `mcp_permissions.json`, flipping tools from `ask` to `allow`. The gate that
   is supposed to be fail-closed can be disarmed by a single tool call that the
   gate itself permits.

Reachability is worse than "a malicious user": this is reachable **by prompt
injection from fetched web content**, since the agent loop will act on
instructions embedded in pages it retrieves.

Aggravating configuration: `config.py:2952` ships `workspace_root` commented
out, so launching the app from `$HOME` confines the tools to `$HOME` — which
contains every path listed above.

The lane's bottom line for its whole tier applies here exactly: this is a
**seam that never adopted an existing correct primitive**, not a wrong
primitive. The denylist is correct; these tools just do not call it.

## Acceptance Criteria

- [ ] `is_sensitive_path` is consulted inside `resolve_workspace_path`, so that
      all seven `fs_*` tools inherit the check from the one choke point rather
      than each re-implementing it
- [ ] Reading, writing, or patching any denylisted path through an `fs_*` tool
      is refused or routed to an explicit approval, matching the behaviour the
      `file_operation_tools` enforcers already have
- [ ] Specifically pinned by test: `~/.ssh/id_rsa`, `~/.aws/credentials`, the
      app's own `config.toml`, `mcp_permissions.json`, and `chachanotes.db` are
      not silently readable or writable via `fs_read`/`fs_write`/`fs_patch`
- [ ] A regression test pins that `fs_write`/`fs_patch` cannot rewrite the
      permission store — the gate cannot be disarmed by a gated tool
- [ ] `Utils/sensitive_paths.py`'s enforcer list is updated so the contract
      names its real enforcement points, and a test fails if a new tool reaches
      the filesystem without passing through the choke point
- [ ] The shipped `workspace_root` default is reviewed: launching from `$HOME`
      must not silently confine the agent to a root containing the user's
      credentials
