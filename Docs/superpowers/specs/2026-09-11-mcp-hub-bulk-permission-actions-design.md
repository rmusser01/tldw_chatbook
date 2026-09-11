# MCP Hub Bulk Permission Actions — Design

Date: 2026-09-11
Status: Draft (awaiting review)
Review basis: MCP screen UX review 2026-09-11 (Opportunity O1)
ADR: backlog/decisions/149-mcp-hub-bulk-permission-actions.md
Related: ADR-081 (prompt reduction; persistent changes must go through `set_tool_state`), ADR-090 (advisory lanes never alter verdicts), raw-shell two-state projection (mcp_workbench `_project_raw_shell_store_state`)

## Problem

The Permissions matrix scales to dozens of rows per server, but every
change is one Space-press per row. The server-default rung is the correct
bulk lever and already exists — yet nothing on the screen teaches that,
and two bulk operations are genuinely missing:

1. **Apply a state to many tool rows at once** ("allow every read-only
   tool on this server").
2. **Clear all tool overrides for a server** (revert to inheriting the
   server default — today only reachable by pressing Space on each
   override row until it returns to Inherit).

## Goals

- Both operations complete in ≤2 keystrokes from the matrix.
- Every write goes through the existing typed, profile-scoped
  `set_tool_state` path — no new store API, no bulk-write bypass
  (ADR-081 posture intact: no auto-approval, no model-initiated writes).
- One confirmation echo + one resync per bulk action (not N toasts).
- Raw-shell tools are never bulk-mutated (their two-state projection
  makes a generic 4-rung bulk write a lie).

## Non-goals

- No free-form multi-row selection mode (considered, rejected as the
  heaviest UI for the marginal remainder over the two keys below).
- No cross-server bulk actions.
- No bulk changes to server defaults or the global row (single-key
  already).

## Approaches considered

**(a) Two scoped keys on the matrix (recommended).**
- `shift+space` on any row of server S: applies that row's *cycled next
  state* to every visible tool row of S. (Cycling a row with shift means
  "this state, for all of them.") Concretely: resolve the same
  `cycle_ui_state` next-state the plain Space press would produce, then
  write it to each visible tool row of S.
- `C` on any row of server S: sets every tool row of S back to Inherit
  (`set_tool_state(..., None)`), i.e. "clear this server's overrides."
Both operate on the **currently visible (post-filter) tool rows of the
cursor row's server only** — the filter becomes the bulk's scope
selector, which is the power-user mental model the filter already
serves.
Trade-off: two new bindings to document (footer hint + legend), but no
new widgets or modes.

**(b) Teach the existing server-default rung only.** Tooltip on
server-default rows + legend line. Trade-off: zero new surface, fixes
discoverability but not the two missing operations; adopted as a *part*
of (a) regardless (the tooltip ships with it).

**(c) Marked-row selection.** Space marks rows, a later key applies.
Trade-off: a modal marking state in a matrix that currently has none —
state to restore, bindings to guard, and it duplicates what the filter
already does for scoping. Rejected (YAGNI).

## Design (approach a)

**Interaction contract**
- Both keys are bindings on `MCPPermissionsMode` (like `space`), posting
  one new message each (`BulkStateRequested`, `BulkClearRequested`)
  carrying the cursor row's `(server_key, profile_context)` — the
  workbench remains the single writer.
- Both keys operate on the cursor row's **server scope**: any tool row
  or that server's default row. The **global row is excluded** — it owns
  no server, and both keys no-op against it (the existing hint Static,
  not a toast, says so). No-op likewise when the server has zero
  *visible* tool rows.
- Scope is the **currently visible (post-filter) tool rows** of that
  server — the filter is the bulk's selector, which is the power-user
  mental model the filter already serves. Consequence, made explicit: a
  filter that hides some rows means `C` clears only the *visible*
  overrides; the echo wording pins this (`"{label}: {N} visible
  overrides cleared"`), the tooltip teaches the full-clear recipe
  ("clear the filter, then C"), and the unfiltered preview's override
  count continues to report any remainder honestly.
- Footer hint (Permissions mode only): `shift+space bulk set · C clear
  overrides`. Legend gains one clause: "shift+space applies the next
  state to the server's visible tools; C clears its visible overrides."
- Echo shape (pinned copy): `"{server_label}: {N} tools → {label}"` /
  the clear wording above — same transient-echo mechanism Space uses
  today.
- Wave-B interplay (cycle reorder lands first): "the row's next cycled
  state" is computed with the same `cycle_ui_state` the plain press
  uses, so after the reorder the first `shift+space` from Inherit
  applies **Ask** to the visible set — the bulk path inherits the
  safety ordering for free and can never bulk-apply Allow as a first
  press from Inherit.

**Write path** (`mcp_workbench`)
- Iterate visible tool rows of the server; skip raw-shell rows (and say
  so in the echo when any were skipped: `· 1 skipped (raw shell)`).
- Every write through `_call_profile_scoped(service.set_tool_state, ...)`
  under the SAME validated `PermissionProfileContext` as a single press;
  on the first failure, stop and surface the service's own error text
  (no partial-success silence) — partial writes are safe because each is
  an independent, idempotent store write.
- One `_sync_permissions_mode(echo=...)` after the batch; `update_states`
  fans out to Tools mode exactly as a single press does.
- Built-in rows participate (hash-free), gated tools cannot appear by
  construction, and an "allow" bulk on a risk-tagged set still leaves
  floors to do their job at resolution time — the matrix re-renders the
  effective states, which is the honest display.

**Safety rails**
- Kill switch, rug-pull hash semantics, and profile stale-write guards
  are unchanged and authoritative; a bulk "allow" writes the same
  definition-hash-fingerprinted state N single presses would.
- No bulk action is offered when the matrix shows zero visible tool rows
  for the cursor's server (no-op with the existing hint Static, not a
  toast).

**Testing**
- Store-level: none (no store change).
- Workbench tests: batch posts exactly N profile-scoped `set_tool_state`
  calls; raw-shell skip + echo suffix; partial-failure toast carries the
  service message; single resync (assert one `update_matrix` echo).
- Canvas tests: binding posts the message with the cursor row's identity;
  footer/legend copy; no-op on empty visible set.

## Open questions (for reviewer)

1. Should bulk writes record a single aggregated execution-log entry, or
   is per-row logging (current behavior per write) acceptable? (Design:
   per-row, unchanged — audit granularity should not decrease.)
2. Should `shift+space` on a server-default row apply the server
   default's next state, or the *global* cycle? (Design: the row's own
   next state, identical to what a plain Space on that row would
   produce — one rule, no special cases.)
