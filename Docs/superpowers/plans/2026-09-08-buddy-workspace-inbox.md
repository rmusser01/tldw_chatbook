# Buddy workspace inbox

Task: TASK-32083. Approved spec: ../specs/2026-09-08-console-buddy-management-design.md.

ADR required: yes; existing ADR applies.
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: explicit workspace scope projects the existing Console execution and receipt owners.

1. Test a pure inbox projection using real session/run/receipt values. Include only
   local sessions currently in the bound workspace; sessionless unread results
   require current workspace membership. A repurposed live slot cannot admit a
   receipt from another conversation. Do not manufacture completion records.
   Extend the existing ordinary/queued outcome publisher to retain unread results
   when Console is hidden even if its selected session has not changed.
2. Add an app-owned coordinator that reads workspace membership off the UI loop,
   reads live state and immutable receipt snapshots, and keeps result IDs frozen.
   Opening and refreshing are read-only. Mark seen acknowledges only the selected
   result after verifying its current owner; it never answers a decision.
3. Add a native modal with Needs you / Running / Results, stable keyboard selection,
   a scrollable list and fixed Open / Mark seen / Close actions. Refresh content
   without grabbing focus. No microphone control is rendered.
4. Open the shared exact-target conversation modal with allow_voice=False from
   every workspace row. Retain the inbox beneath it and the original destination
   beneath both; closing does not touch execution or Console selection.
5. Verify projection, stale/removed targets, receipt acknowledgement races, and
   mounted normal/compact layouts. Update task evidence and user documentation.

Integration seam: BuddyInboxEntry carries key, title, group, summary, exact
BuddyBinding, and immutable receipt IDs. Speech may read entries but must never
acknowledge them. The manager chooses the binding; neither modal retargets itself.
