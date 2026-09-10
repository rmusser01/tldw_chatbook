# ADR-147: Reversible conversation archive and exact resume

Date: 2026-09-10
Status: Accepted

## Context

The archive UX review found that workspace archive could preserve chats but Library's Open in Console staged source context instead of resuming the selected conversation. Chat-level archive, transcript review, bulk recovery and nearby workspace restoration were missing. The user approved addressing all nine review findings.

## Decision

- Conversation archive is a local, durable lifecycle flag independent of deletion, conversation workflow state, and workspace archive. Add an indexed archive column through the next ChaChaNotes migration. Existing records start active. Archive never deletes messages or changes workspace/branch identity.
- Local conversation queries accept `archive_scope` (`active`, `archived`, `all`), defaulting to active. Filtering occurs before counts/pagination. Exact-ID reads and text-only preview remain available for archived conversations. Remote contracts are unchanged.
- Mutations are version checked and return actual changed identities with new versions. Undo targets only those changes and refuses stale intervening changes. Batch failures are reported honestly.
- Console blocks archiving conversations with live, queued, or unsaved work; archive does not cancel agents. Workspace archive similarly refuses live/queued activity within the workspace. Saved idle tabs may be closed only without losing draft state; otherwise keep them and visibly explain the state.
- Library owns durable browse/search, a bounded read-only transcript preview, and individual/bulk Archive, Restore and Undo. Active/Archived/All refer to conversation state, with workspace archive separately labeled. Restore-only preserves active context. Restore and resume states any workspace-wide restoration impact before acting.
- A distinct typed Console resume handoff carries only the persisted local conversation ID. Console claims it, reuses an existing session or hydrates the original conversation/active branch, preserves unrelated drafts and acknowledges only after successful activation. Source reuse stays on the existing source-context handoff under the explicit Use as source action.
- Workspace recovery is reachable from Console's switcher and Settings. Restore accepts an optional replacement name atomically, solving name reuse without asking users to rename an inaccessible archived record. Restore does not auto-activate; Switch is explicit.
- Archive feedback includes a persistent Undo/View archive route. Textual keybindings follow ADR-031, default workspace grouping follows ADR-027, and recovery receipts follow ADR-055's principles.

## Alternatives

- Reusing deleted or workflow state was rejected: archive must not imply trash or overwrite resolved/backlog state.
- Source-staging as resume was rejected: it changes conversation identity and meaning.
- Hiding rows only in memory was rejected: archival must survive restart and paginate correctly.
- Automatic cancellation during archive was rejected: organization must not silently terminate work.

## Consequences

The database and service own lifecycle truth; widgets do not maintain competing archive lists. Cross-screen navigation must preserve original conversation ownership. Targeted migration, real SQLite, mounted UI, branch/draft and compact keyboard checks are required.

Tasks: TASK-32273, TASK-32274, TASK-32275, TASK-32276.
