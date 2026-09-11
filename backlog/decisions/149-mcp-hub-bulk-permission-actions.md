# ADR-149: MCP Hub Bulk Permission Actions

Status: Proposed
Date: 2026-09-11
Related Tasks: MCP Hub UX program 2026-09-11 (Wave F / O1)
Spec: Docs/superpowers/specs/2026-09-11-mcp-hub-bulk-permission-actions-design.md
Related: ADR-081 (prompt reduction — persistent permission changes must go through `set_tool_state`; no telemetry-driven or model-initiated auto-approval), ADR-090 (advisory surfaces never alter verdicts), ADR-079 (profile-scoped policy)

## Decision

Add two keyboard-scoped bulk operations to the Permissions matrix, both
posting a single message that the workbench executes as N ordinary,
individually profile-scoped `set_tool_state` writes:

1. **`shift+space`** — apply the cursor row's next cycled state to every
   *visible* (post-filter) tool row of that row's server.
2. **`C`** — clear every visible tool override for that server (back to
   Inherit).

The visible filter is explicitly the bulk's scope selector. Raw-shell
tool rows are skipped and the skip is named in the single transient
echo. No new store API, no batch endpoint, no selection mode: the
mutation surface widens only at the UI gesture layer while every write
keeps the exact semantics (profile context capture, definition-hash
fingerprinting, kill-switch subordination, fail-closed errors) of a
single Space press.

## Why

The matrix scales to dozens of rows per server but every change is one
keystroke per row, and the correct existing bulk lever (the server-
default rung) is undiscoverable. "Allow this server's read-only tools"
and "revert this server to its default" are the two genuinely missing
operations; both are safe to express as repeated single writes, and the
filter already gives users a natural scoping gesture.

## Alternatives considered

- **Teach the server-default rung only** (tooltips/legend): fixes
  discoverability, not the two missing operations; adopted as a
  companion, not the fix.
- **Marked-row multi-select** (mark rows, then apply): the heaviest UI —
  a modal marking state in a currently stateless matrix — for a marginal
  remainder over filter-scoped keys; rejected as YAGNI.
- **A dedicated bulk store API** (one call, one profile-context check):
  marginally faster, but creates a second mutation path to harden and
  audit; rejected while N is bounded by visible rows (tens).

## Consequences

- Two new Permissions-mode bindings and one message pair; footer hints
  and legend copy extend (test-pinned strings move with them).
- Per-row execution-log entries are unchanged — audit granularity does
  not decrease, and a partial failure stops the batch with the service's
  own error surfaced (each write is independent and idempotent, so
  partial application is safe).
- Bulk "allow" inherits all existing floors: risk-tagged tools still
  resolve through the high-risk floor at read time, and rug-pull hash
  semantics still downgrade stale allows. Bulk changes what is *stored*,
  never what is *resolved*.
- Future bulk surfaces (cross-server, profile-wide) must extend this ADR
  rather than add parallel mechanisms.
