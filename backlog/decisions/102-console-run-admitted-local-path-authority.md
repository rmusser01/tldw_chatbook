# ADR-102: Bind Console local path tools to run-admitted workspace roots

Status: Accepted
Date: 2026-08-30
Related Task: [TASK-19504](../tasks/task-19504%20-%20Bind-Console-local-path-tools-to-run-admitted-workspace-roots.md)
Design: [Console run-admitted local path authority](../../Docs/superpowers/specs/2026-08-30-task-19504-run-admitted-workspace-roots-design.md)
Supersedes: ADR-069's disabled-session local-provider fallback only

## Decision

The in-app Console will publish structured local filesystem, read-only Git, and
Virtual CLI schemas only when the owning run admits one or more valid local-folder
bindings. A run captures bindings from its own workspace ID, never from the
currently viewed or globally active workspace. The default workspace and a named
workspace with no valid bindings publish none of those path-taking schemas.
Unrelated local web, Watchlists, and todo tools remain available, and built-in
sandbox file tools retain their private Chat scratch authority.

Each admitted root carries its stable binding ID as the model-facing root alias,
canonical-locator fingerprint, captured filesystem identity, and admission-time
access mode. With one admitted root the alias may be omitted for compatibility;
with multiple roots every path call must provide an alias. Alias selection never
re-reads the active workspace.

Before review and again immediately before execution, the provider resolves the
captured binding ID and verifies workspace membership, local-filesystem kind,
ready status, locator fingerprint, filesystem identity, and sufficient current
access. Reads accept `ro` or `rw`; mutations require `rw`. Removal, retargeting,
identity replacement, or access downgrade refuses the call. Once admitted, the
operation uses ADR-101's one-shot pinned executor as its pathname-race boundary.

Project-instruction-enabled sessions retain ADR-069's single selected binding and
its stronger session-state guard. The selected binding is represented through the
same run-root shape so local providers have one authority contract.

Path-tool input schemas include the admitted alias contract. That intentional
schema change flows through the existing permission definition hash, invalidating
stale persistent approvals without a new permission store or migration.

The standalone local MCP server remains outside this in-app authority path and
continues to use its explicit `[console] workspace_root` or standalone process-CWD
fallback. Raw shell remains governed separately by ADR-094 and is not widened by
this decision.

## Context

ADR-069 selected one workspace binding for project-instruction-enabled sessions
but explicitly left disabled sessions on the legacy provider-root behavior.
Subsequent Console work gave each Chat private scratch and retired configured-root
authority in the mounted Console, yet structured `fs_*`, `git_*`, and Virtual CLI
schemas still appear against that scratch when no workspace folder was admitted.
Named workspaces with multiple bindings likewise have no stable per-call root
selection contract.

TASK-19637 and ADR-101 now provide the missing execution primitive: every admitted
operation runs in a fresh worker pinned to the captured root identity. TASK-19504
can therefore remove fallback authority without overstating pathname safety.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Generate one tool name per root | Makes catalog names workspace-specific, multiplies approval rows, and creates more schema churn than one explicit argument. |
| Encode aliases in paths such as `@root/file` | Introduces a second path grammar and makes patch/Git argument parsing ambiguous. |
| Keep scratch-backed structured tools when no binding exists | Preserves hidden authority and contradicts the task's bindingless behavior. Built-in sandbox tools already own private scratch access. |
| Re-resolve the active workspace on every call | A tab or workspace switch could retarget a live run. Authority must derive from the run's captured workspace and binding IDs. |
| Add a new permission subsystem | Existing schema definition hashes already invalidate stale grants; another store would duplicate policy. |

## Consequences

- Dynamic path schemas vary with the admitted alias set and intentionally require
  renewed approval after this upgrade or a later authority-shape change.
- Bindingless runs lose only structured local filesystem/Git and Virtual CLI
  schemas; non-path local tools and built-in scratch tools remain intact.
- Mixed `ro`/`rw` roots can expose mutation schemas when at least one root is
  writable, but a call selecting a read-only alias is refused at call time.
- Registry and filesystem authority are checked at least twice per path call.
- The model sees opaque binding IDs and labels, never raw absolute locators.

## Links

- [ADR-028](028-settings-workspaces-category-and-folder-roots.md)
- [ADR-032](032-local-agent-tool-permission-boundary.md)
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-101](101-one-shot-pinned-workspace-tool-execution.md)
