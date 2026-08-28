# ADR-033: Local Agent Process-Execution Boundary

Status: Accepted
Date: 2026-08-05
Related Task: [TASK-2820 - Local agent tools phase 2: file tools (fs_read/fs_write/fs_edit/fs_glob/fs_grep)](../tasks/task-2820%20-%20Local-agent-tools-phase-2-file-tools-fs_read-fs_write-fs_edit-fs_glob-fs_grep.md)
Supersedes: N/A
Partially Superseded By: [ADR-093: Raw and Virtual CLI Execution Boundaries](093-raw-and-virtual-cli-execution-boundaries.md)

## Decision

Phases 3b-ii/4 of the local-agent-tools effort introduce the first
model-invocable tools that spawn processes and answer the long-deferred
raw-shell question. Three decisions follow, all under the ADR-032 naming,
confinement, and approval discipline.

**Git tools process boundary.** The ported read-only git tools
(`git_status`, `git_diff`, `git_log`, `git_blame`, `git_branches`) spawn
`git` via fixed argv arrays with no shell interpolation, an allowlist of
read-only subcommands and flags, cwd/repo discovery confined to the
workspace root (ADR-032 confinement), a 30 s timeout, and a 1 MB output
cap. They carry NO risk tag: the existing `process` tag in
`HIGH_RISK_TAGS` (`MCP/permission_store.py:69`) is deliberately NOT
applied to this read-only allowlisted set. The risk floor exists to force
fresh approval when powerful capabilities are granted by inheritance; a
fixed read-only argv allowlist with no shell interpolation has a
materially smaller blast radius than arbitrary process execution, and the
tools only return repository state. **Tripwire (binding):** if the
allowlist ever expands past read-only subcommands, the `process` tag MUST
be applied and this ADR revisited.

**No raw shell tool for models — virtual-CLI is the adopted answer.** The
project will not add a raw bash/shell tool for model use. ADR-093 partially
supersedes this decision by authorizing a separately gated, user-invoked raw
CLI command with explicit full OS-user authority; that command bypasses the
model and is not a catalog tool. The adopted model-facing design
(design-only for now, implemented in a future phase) is tldw_server's
governed virtual-CLI model (`run_command_module.py` +
`command_runtime/`): an allowlisted command registry (`ls`, `cat`,
`grep`, `find`, `stat`, …) mapping onto the policy-checked
`fs_*`/`git_*` cores — no host-shell subprocess at all — with
profile-granted commands and output spill-to-disk past ~64 KiB with
preview caps. Reference: tldw_server @
`5605b9d9906322c2e6b5342b48c391ae674d315e`.

**Deferred permission upgrades, recorded.** TTL-bound approval grants
(augmenting permanent `always_allow`), claude-code-style rule syntax
(`Read(/docs/**)`), and `explain-policy` dry-run evaluation UX are
explicitly deferred. They are not phases 3-4 scope; they are recorded
here so the questions do not get re-asked.

## Context

ADR-032 established the local tool permission boundary — synthetic server
key `local:__local__`, fail-closed approval, workspace-root confinement —
for tools that read and write files through Python code paths. Phase 3 of
the design spec `Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md`
(phases 1-2 shipped as PRs #1352/#1358) crosses a new line: the ported
git tools are the first model-invocable tools that spawn a subprocess,
and the same phase had left the raw-shell question open since the
original design.

The re-plan spec
`Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md`
found mature, portable answers in tldw_server's unified MCP surface:
read-only git tools with a subcommand allowlist, and a governed
virtual-CLI that covers the legitimate model use cases for shell-like
interaction without a host-shell attack surface. Because spawning
processes is a security/runtime-boundary decision — and because the
deliberate rejection of a raw bash tool is exactly the kind of decision
future contributors will ask about again — both are recorded here before
the first subprocess-spawning tool lands.

The permission-store risk floor (`HIGH_RISK_TAGS`) floors inherited
allows to `ask` for tags like `mutates` and `process`. Applying it
mechanically to the git tools would make every session re-approve tools
whose entire capability is reading repository state through a fixed argv
allowlist — approval friction without a matching risk. The trade-off is
recorded as a binding tripwire rather than silently accepted.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Raw bash tool with sandboxing | A sound sandboxing story (seatbelt/containerization, network policy, audit) is a project of its own, and the virtual-CLI covers the legitimate model use cases without a host-shell attack surface. |
| No process-spawning tools at all | Read-only git state is high-value for a coding assistant, and the allowlist boundary (fixed argv, read-only subcommands, confinement, timeout, output cap) is tight. |
| Apply the `process` tag to git tools now | Would floor inherited allows to `ask` for tools that only read repo state, adding approval friction without a matching risk; recorded as the binding tripwire instead. |
| Port tldw_server's permission profiles/rules wholesale | Duplicates chatbook's existing permission-store machinery; the deferred-upgrades note captures the valuable parts (TTL grants, rule syntax, dry-run UX) as future work. |

## Consequences

### Benefits

- The process-execution boundary is documented before the first
  subprocess-spawning tool lands, so phase-3b-ii plans and tests pin
  against a stated contract (argv allowlist, confinement, timeout, output
  cap).
- The raw-shell question is answered with a concrete reference design
  (tldw_server's virtual-CLI at a pinned commit), ending the recurring
  "should we add a bash tool" debate.
- The permission-upgrade backlog (TTL grants, rule syntax, dry-run UX) is
  recorded as deliberate deferral, so those questions are not re-asked.

### Accepted trade-offs

- The git tools do not risk-floor: an inherited `allow` from a server or
  global default authorizes them without fresh approval, accepted because
  the read-only argv allowlist bounds the blast radius to repository
  state. The tripwire makes any allowlist expansion a forcing event for
  revisiting this.
- The virtual-CLI is design-only until a future phase; models have no
  shell-like capability in the interim beyond the catalog tools.
- External MCP callers cannot use approval cards, so an `ask` state fails
  closed externally and mutates-tagged tools stay refused by default
  until an operator grants `allow` (re-plan spec §3.1).

## Links

- [TASK-2820](../tasks/task-2820%20-%20Local-agent-tools-phase-2-file-tools-fs_read-fs_write-fs_edit-fs_glob-fs_grep.md)
- [Re-plan specification](../../Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md)
- [Original design specification](../../Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md)
- [ADR-032: Local Agent Tool Permission Boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-093: Raw and Virtual CLI Execution Boundaries](093-raw-and-virtual-cli-execution-boundaries.md)
