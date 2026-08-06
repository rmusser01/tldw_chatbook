# ADR-032: Local Agent Tool Permission Boundary

Status: Accepted
Date: 2026-08-05
Related Task: [TASK-1338 - Local agent tools phase 1: plumbing + fs_list pilot](../tasks/task-1338%20-%20Local-agent-tools-phase-1-plumbing-fs_list-pilot.md)
Supersedes: N/A

## Decision

Workspace-local file, web, and todo tools join the Console agent runtime as a
first-class `ToolProvider` at the `Agents/tool_catalog.py` seam. Local tools
carry `local:<name>` catalog ids and `fs_`/`web_`/`todo_` tool names, following
the ADR-030 `library_*` naming precedent.

Local tools reuse the MCP permission store under the synthetic server key
`local:__local__` — distinct from the existing `builtin:tldw_chatbook`
no-transport precedent — with no store schema change, since server keys are
opaque. Tool-override → server-default → global-default precedence, session
approvals, the kill switch, and rug-pull `definition_hash` checking all apply
unchanged. Write tools carry `mutates` risk tags so the approval card presents
them correctly.

Approval is fail-closed under the three-mechanism discipline: the approval
callback is cleared first on any refusal path, `invoke()` returns a refusal
without stamping approval when no callback is wired, and `stamp_scope` wraps
sub-agent runs so nested invocations re-check permissions. Refusal strings are
pinned constants from spec §3.3: `LOCAL_DENY_REFUSAL`, `LOCAL_TIMEOUT_REFUSAL`,
and `LOCAL_KILL_SWITCH_REFUSAL`.

All path-taking tools confine to a configurable `[console] workspace_root`
(default: the app cwd at startup), coerced and templated following the
`collapse_large_pastes` precedent in `config.py`. Hidden path components are
allowed under the root via a new `allow_hidden` parameter on `validate_path`;
traversal outside the root is always rejected.

## Context

The Console agent runtime currently offers only calculator/datetime plus MCP
and skill tools. The spec
`Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md` adds
workspace-local file/web/todo tools, which gives model-initiated calls local
filesystem read/write and network access for the first time outside the MCP
boundary. That raises the stakes on approval gating: a misconfigured or
bypassable approval path would let a cloud model read or modify arbitrary local
files without user consent.

The MCP permission store already implements the approval semantics these tools
need — per-tool overrides, session grants, persistent "always allow" with
rug-pull hashing, and a global kill switch — but it is keyed by MCP server id.
Introducing a parallel store would duplicate the audit trail and force a second
approval UX. Registering local tools under one synthetic server key reuses all
of it while keeping local tools clearly distinguishable from any real MCP
server.

Path confinement needs a canonical root the user can inspect and change. The
existing `validate_path` helper rejects hidden components outright, which is
too strict for real workspaces (`.git`, dotfile configs), so the boundary is
root confinement with hidden components explicitly allowed under it.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Self-hosted MCP server consumed via the in-process delegate | JSON-RPC plumbing for local file reads, and the runtime would depend on the MCP lifecycle for basic capabilities. |
| Separate local permission store | Duplicates the audit trail and approval UX, and two stores would drift in precedence, session-grant, and kill-switch behavior. |
| Config-flag-only gating | No interactive approval and no per-tool persistence; the weakest safety story for exactly the tools that need it most. |

## Consequences

### Benefits

- Adding a local tool never touches the runtime loop; registration goes through
  the existing `ToolProvider` seam.
- "Always allow" persists with a `definition_hash` exactly like MCP tools, so
  rug-pull protection covers local tools with no new code.
- The MCP workbench's permission UI can display local tools later without
  migration, since they already live in the same store.
- One approval and audit trail covers MCP and local tools uniformly.

### Accepted trade-offs

- The permission store gains a synthetic server key that is not a real MCP
  server; UI that enumerates servers must special-case or filter it.
- `validate_path` grows an `allow_hidden` parameter whose default must remain
  the current strict behavior for existing callers.
- `workspace_root` defaults to the app cwd, so the confinement boundary moves
  with where the app is launched until the user configures it.

## Links

- [TASK-1338](../tasks/task-1338%20-%20Local-agent-tools-phase-1-plumbing-fs_list-pilot.md)
- [Design specification](../../Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-08-04-local-agent-tools-phase1.md)
- [ADR-030: Direct Local Library Tool Boundary for Console and MCP](030-local-library-agent-tool-boundary.md)
