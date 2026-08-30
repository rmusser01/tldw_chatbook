# TASK-19504: Console run-admitted workspace roots design

## Goal

Remove hidden in-app Console fallback authority for structured local path tools.
Every published `fs_*`, read-only `git_*`, or Virtual CLI operation must select a
folder binding admitted for the owning run, while unrelated local tools and
built-in private-scratch tools remain available.

## Authority snapshot

At run admission the controller reads the owning session and its workspace ID.
Project-instruction-enabled sessions contribute their already validated ADR-069
selection. Disabled named workspaces contribute every currently valid
local-filesystem binding returned for that workspace. The default workspace and a
named workspace with no valid folder bindings contribute none.

Each immutable admitted root contains:

- `workspace_id` and `binding_id`;
- `alias`, equal to the stable opaque binding ID;
- resolved root and canonical-locator fingerprint;
- captured root/ancestor filesystem identity;
- admission-time `allow_write`;
- a call-time guard that re-reads that binding ID and compares every field above.

Invalid, missing, symlinked, retargeted, identity-replaced, or non-ready bindings
are excluded at admission. The guard also rejects removal and `rw` to `ro`
downgrades. A read accepts either current access mode. A mutation requires current
`rw`; no admission-time state can widen later registry state.

## Local provider catalog

The existing `LocalToolProvider` remains the single provider and approval owner.
It gains an immutable alias-to-authority map and routes only the existing path
specs through the selected authority's existing `WorkspaceToolExecutor`.

- Zero admitted roots: omit `_PATH_AUTHORITY_LOCAL_NAMES`; keep web, Watchlists,
  todo, and other non-path specs.
- One root: add optional `root_alias` with a one-value enum; omission selects the
  sole root.
- Multiple roots: add required `root_alias` with the admitted aliases as its enum.
- Mutation specs appear only when at least one admitted root is writable. Selecting
  a read-only root for a mutation fails before execution.

The alias argument is retained in approval arguments but removed before invoking
the existing root-bound handler. `path_targets`, approval preflight, and execution
all use the same selector. The schema itself therefore participates in the
existing definition-hash approval guard.

## Virtual CLI

`VirtualCliProvider` receives the same read-authority map and schema rule. It
selects one existing `VirtualCliRegistry`/pinned executor per call. Because every
Virtual CLI command is read-only, both `ro` and `rw` roots are eligible. With no
roots, the provider is not composed.

Raw shell is unchanged and remains under ADR-094; TASK-19504 neither grants it a
workspace binding nor claims its process boundary.

## Controller composition

Provider composition receives the captured roots instead of a scratch/config/CWD
fallback. Project-instruction state supplies exactly one selected root. Otherwise
the controller captures the owning named workspace's valid bindings once at run
admission. It never consults the active workspace after capture.

The non-path local provider may retain private scratch only as an internal service
construction input; no path spec or handler is registered against it. Built-in
file tools continue to receive Chat scratch plus their existing binding policy.
Standalone MCP construction remains unchanged.

## Upgrade communication

The Console tool guide will state that structured path tools now require admitted
workspace folders and that existing persistent approvals may prompt once again
because their schemas changed. No permission rows are manually deleted.

## Error handling and privacy

All authority failures are fail-closed and use bounded, path-free refusal copy.
Diagnostics identify only codes and opaque binding IDs. Absolute locators never
enter model schemas, approval metadata, or generic logs.

## Verification

Focused tests will prove:

- zero/one/multiple-root catalog and alias schemas;
- owning-workspace capture independent of active workspace;
- read/write enforcement for mixed roots;
- removal, retarget, identity replacement, and downgrade revocation;
- preserved ADR-069 single-selection behavior;
- preservation of web, Watchlists, todo, and built-in scratch tools;
- Console config/CWD independence and unchanged standalone MCP behavior;
- definition-hash invalidation from the schema change;
- Virtual CLI parity and existing ADR-101 pinned execution.

The task will run targeted provider/controller/workspace tests, changed-file static
checks, diff checks, and a final independent review. A full repository sweep is
outside the default verification scope unless explicitly requested.
