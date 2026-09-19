# ADR-169: Shared ownership of MCP local configuration saves

Status: Accepted
Date: 2026-09-18
Task: TASK-32793
Extends: ADR-168; ADR-033 settings commit models remain authoritative.

## Context

The local-tools master setting is exposed in Tools and the built-in Servers
detail. Both dispatch exclusive screen workers around threaded configuration
writes. A held older off-write can finish after a newer on-write; cancelling its
observer does not cancel the thread. Read-only refresh restores the old cached
label while a choice is pending. A file replacement followed by failed cache
publication is incorrectly described as a failed save.

## Decision

Extend the existing app-owned MCP root-save coordinator to exactly two local
configuration keys: workspace_root and local_tools_enabled. Retain one FIFO
write queue, with bounded latest request/outcome and committed-value receipts
per key. Both MCP master controls call the same admission path synchronously.
Activation captures the displayed target and configuration before event delivery.
Admission captures
the originating configuration identity before yielding; the atomic configuration
writer checks that identity under its existing lock. Cancellation only cancels
observation. Normal shutdown fences admission and drains admitted writes before
dependent teardown, as specified by ADR-168.

Root remains explicitly saved, preserving its mounted draft and exact revision.
Master remains instant-apply, allowing a newer explicit reversal while a save is
pending. Render the latest pending choice with a Saving receipt across refresh
and screen recreation. On failure, restore persisted/cached truth with an error;
on committed-file/cache-publication failure show a saved-to-file warning. Scope
receipts and partial-cache fallback values to configuration path, runtime
generation and file revision. A successful no-op cannot falsely clear an
unrepaired cache warning. Root and master receipts remain independent; master projection never refreshes
the editable root. Pending master choices only project control state; saved off
explanations are hidden until the write completes. For a known sibling-only mutation,
carry an unchanged committed receipt to the new fence only when its prior fence
matches the locked pre-write identity. A second partial publication preserves
the warning; a successful publication of the whole config may clear it. Never
replace a newer pending or failed receipt. Unknown external edits remain fenced.

Publish the persistence outcome before best-effort catalog refresh. Presentation
failure must not turn a committed setting into a reported save failure. Queued
callbacks consult the current owner/state and may not replace a newer choice or
another configuration's controls. No permission, tool exposure, runtime authority,
storage schema, durable job ledger or hard-termination guarantee is added.

## Alternatives

- Keep independent exclusive workers: cancellation cannot recall writes and the
  two controls could still race each other.
- Disable the controls during writes: removes immediate reversal and still does
  not establish ownership across navigation or shutdown.
- Copy a second save coordinator: repeats the same FIFO, shutdown and scoped
  receipt contract next to the existing owner. Two bounded per-key receipts share
  that machinery instead.
- General settings framework: unnecessary. Other settings keep their existing
  commit models; this owner admits only the two existing local MCP controls.

The design and targeted evidence are tracked in TASK-32793 and
Docs/superpowers/plans/2026-09-18-mcp-master-settings.md.
