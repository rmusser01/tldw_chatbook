# Conversation Buddy quick interaction

Task: TASK-32082. Approved design:
`Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md`.

ADR required: no
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: implement the approved projection/command surface over Console ownership.

1. Expose `open_buddy_conversation(app, binding, *, allow_voice=True)` through an
   app-owned coordinator. Resolve the exact local binding on every command. Keep
   drafts by explicit conversation, transferring them when the same durable owner
   gains a canonical binding; never switch ambient Console selection for a reply.
   Restore explicitly opened saved inbox rows through shared full-tree hydration
   with activate=False, checking local durable identity/profile after each I/O gap.
   A cold Home opener may bootstrap the existing launch runtime; inbox snapshots
   remain provider/bridge-free, and a failed bootstrap retains Open Console recovery.
2. Add a narrow composer-preserving manual submission option. Carry its state through
   prepared/durable continuation, suppress Console composer-clearing callbacks and
   preserve its draft. Reject unseen staged attachments, one-shot prefill or Library
   evidence visibly; slash commands and @ reference routing require Console. All
   ordinary admission, tools and provider rules still apply.
3. Build a bounded native modal with name, transcript, activity, existing question and
   approval cards, text, Close and Open in Console. Per-round decisions require the
   exact bound session and current round. Reuse the existing resolver, never grant
   permissions in the modal. Register only visible decision-view claims so finite
   answerable-time budgets remain accurate without changing Console selection.
   Unsupported worktree decisions keep their existing Open Console review and do
   not acquire an answerable-time claim merely by showing that explanatory copy.
   Retain decision availability only for an explicitly opened exact live session
   and binding revision. Cold Buddy targets can park new decisions after close;
   wake-only no-view sessions retain their existing fail-closed admission.
4. Reuse ConsoleStreamingDictationSession and its existing availability/config/PCM
   guards. Capture only after an explicit click; finish into an editable draft and
   require Send. Discard on close/suspend, stale owner or generation change. Workspace
   row callers pass allow_voice=False and get no microphone control.
   Use shared Buddy speech controls and hold playback until microphone cleanup.
5. Write failing mounted two-session send/approval tests before implementation. Prove
   unrelated drafts/attachments remain untouched; closing retains accepted work;
   stale/deleted owner commands fail closed; drafts/focus restore; microphone closes
   and late results cannot modify a new view. Run targeted UI/controller tests and
   changed-code lint/format checks only. No full sweep or real-provider claim.
