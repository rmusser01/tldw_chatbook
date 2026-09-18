# ADR-170: Suppress table redraw highlights when they are published

Status: Accepted
Date: 2026-09-18
Task: TASK-32796
Related: ADR-161

## Context

The shared DataTableClickSelectMixin forwards focused row/cell highlights to
existing selection handlers so clicks and arrow keys update inspectors. Its
repopulation flag is released after refresh, but table messages can reach the
pane later. MCP server navigation clears tool/finding details and then its own
redraw selects rows again, reopening obsolete details. Findings also lacked an
explicit repopulation boundary.

## Decision

Make the shared repopulating_table(table) helper a synchronous context manager.
Wrap each programmatic clear/add/cursor-restore block in Textual's existing
table.prevent(RowHighlighted, CellHighlighted) context. Suppression occurs at
message publication, before queue delivery can outlive a timing flag. Reset the
helper's gesture dedup state for the rebuilt table, so the next real activation
is accepted. Include table identity in dedup keys: refreshing hidden Audit
Findings must not reset a pending gesture in Executions. Apply the same context
to external drill cursor moves whose callers already populate the inspector. Do not suppress RowSelected or hold this context across
an await. Preserve ordinary click/arrow selection and one-shot Enter dedup.

Update all existing callers (MCP Tools, Permissions, Servers, Audit executions
and Voice Cloning) and the missing Audit Findings boundary together. This
changes the helper contract, not table ownership, runtime permissions, provider
behavior or persisted state. Refresh must preserve focus and retained row
identity where the owning component already promises it.

## Alternatives

- Add another refresh/pause: still guesses when another message queue drained.
- Blur tables during refresh: discards the user's keyboard position.
- Filter only inside the inspector: leaves programmatic selection messages
  active in other consumers and cannot distinguish their cause after delivery.
- Replace Textual tables with a new framework: unnecessary; prevent() already
  provides the publication boundary needed by this repair.
