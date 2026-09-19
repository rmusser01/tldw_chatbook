# ADR-171: Console conversation review and attention presentation

Date: 2026-09-18

Status: Accepted — written design approved; pre-implementation review corrections recorded 2026-09-18.

Related tasks:

- [TASK-32826: unread and attention actions](../tasks/task-32826%20-%20Add-manual-unread-and-meaningful-conversation-attention-actions.md)
- [TASK-32827: consistent conversation rows](../tasks/task-32827%20-%20Align-Console-conversation-rows-with-workspace-rows.md)
- [TASK-32828: Inspector modal](../tasks/task-32828%20-%20Redesign-the-Conversation-Inspector-modal-for-context-and-costs.md)

Amends presentation in ADR-083. Preserves ADR-010 local marks, ADR-085 receipt
authority, ADR-069 including its September 7 capture amendment, ADR-097 trace
disclosure, ADR-031 keyboard conventions and ADR-150 design tokens.

## Context

Console Conversations and workspace chat rows use different density and action
controls. Separate appearance and menu buttons add clutter. Abstract state
glyphs do not make attention requirements obvious. The Conversation Inspector
modal nests Next Send navigation and makes context/cost inspection harder than
its underlying data requires. The user explicitly excludes the Inspector sidebar.

## Decision

1. Store manual unread as a durable profile-local conversation mark under
   ADR-010, independent of syncable conversation metadata and operational
   unseen-result receipts. It clears on explicit Mark as read or a successful
   deliberate revisit, never repaint, background work or automatic restoration.
   Fence acknowledgement against a newer mark and changed target identity via
   serialized compare-and-clear. A process-lifetime generation suffices for UI
   callbacks; timestamp equality alone is not a concurrency guard. Batch reads
   must cover the requested conversation IDs, not only the latest 100 marks.
2. Give Conversations and workspace chat rows a consistent compact visual and
   keyboard contract while retaining their existing ownership, ordering and
   bounded projections. Preserve subagent/progress information.
3. Use one right-hand conversation menu control whose representative icon
   reflects attention priority or the custom/default icon. Its action is always
   opening the menu. Move appearance editing into that menu. State semantics
   derive from authoritative models, not rendered glyphs; text and ASCII
   fallbacks make each state understandable.
4. Approval/blocked/failed attention outranks running and manual unread. Keep
   stopped/cancelled outcomes and successful unseen results honest and subject
   to their existing acknowledgement rules. A manual mark never rewrites or
   clears an operational receipt. The spec defines complete icon precedence.
   Coarse background-unseen evidence without an outcome uses a neutral activity
   indicator, never a success check. Typed semantic projection data preserves
   concurrent states and hidden-row aggregation without a new receipt owner.
5. Organize the existing **Conversation Inspector modal** into Context,
   Usage & cost, and Exchange history. Entry points open relevant views.
   Context uses section/detail navigation; usage uses turn/detail navigation;
   narrow layouts provide a Back route. Preserve accounting, capture, export,
   redaction and explicit Next Send disclosure authorities. Preserve historical
   captured instruction access under ADR-069/097, and invalidate Safe/Full bodies
   across every trace-bearing view. Usage labels state actual estimate/coverage
   scope; call detail is never added twice to aggregate costs.
6. Do not redesign the Inspector sidebar or globally change its glyphs as a
   side effect. Shared code edits must preserve that surface's behavior.

## Alternatives considered

- Reuse operational unseen receipts for manual unread: rejected because a
  user reminder must not fabricate outcomes or acknowledge failures.
- Store unread in synced conversation status: rejected because read state is
  local organization, not workflow status or workspace ownership.
- Clear unread whenever the current chat renders: rejected because this would
  immediately undo marking the currently open conversation unread.
- Separate appearance, attention and action buttons: rejected because the
  approved design requires a cleaner row with one stable menu control.
- Let the icon click change behavior with status: rejected because changing
  from menu to approval/open action would make the same target unpredictable.
- Add an Inspector overview or separate screen: rejected because direct
  task-specific entry in the existing modal serves context and cost inspection.

## Consequences

The local marks service remains the manual-read storage owner; receipt services
remain the operational authority. A pure attention presentation mapping and
consistent row controls are shared across the two conversation lists. Context,
usage, capture and export services retain their contracts; no new storage schema,
provider call, dependency, sync contract or Inspector-sidebar redesign is intended.

Targeted persistence/race tests, mounted navigation and production-CSS rendering
are required. Meaningful symbols must be qualified in supported terminal fonts
and ASCII mode. The written specification has been approved; its source-backed
review corrections are documented there. Implementation plans/notes must link
this ADR. No implementation or live UX qualification is claimed by acceptance.

## Links

- [Approved design specification](../../Docs/superpowers/specs/2026-09-18-console-conversation-review-and-attention-design.md)
- [Local marks](010-console-conversation-local-marks.md)
- [Workspace row ownership](083-console-edge-rails-and-workspace-tree-ownership.md)
- [Activity receipts](085-console-activity-receipts-and-switcher-ownership.md)
