# 027 - Default-workspace conversations live in the Chats section

Date: 2026-07-26
Status: accepted
Relates to: TASK-723 (workspace-settings UX review, finding m3)

## Context

Every persisted Console conversation carries a workspace identity - chats
started in the built-in Default workspace are stored with
`workspace_id=workspace-default, scope_type=workspace`, exactly like chats in
user-created workspaces. The Console conversation browser, however, files
Default-workspace rows under the plain **Chats** section
(`conversation_browser_state._belongs_to_chats`), while user-created
workspaces get named groups under **Workspaces**. The workspace switcher
presents Default as a selectable workspace like any other.

The 2026-07-26 UX review flagged the mixed metaphor: a user who chats in
Default and later adopts workspaces may look for their old chats under a
"Default" group that does not exist.

## Decision

**Keep the current grouping.** Default-workspace conversations belong in
Chats; only user-created workspaces get named groups. Rationale:

1. Everyday chatting must not demand workspace vocabulary. The Default
   workspace exists so the data model is uniform, not so users think about
   it. A "Default" group under Workspaces would force the workspace concept
   onto users who never opted into it.
2. A Workspaces section that always contains at least one group (Default)
   makes the empty state impossible - "No workspace conversations." is the
   honest signal that the user has not created workspaces yet.
3. The storage identity (`workspace-default`) stays an implementation
   detail; separation guarantees are unaffected (verified live in the
   review's scenario 20).

To keep the surfaces from contradicting each other, the switcher's Default
row is annotated "Default (everyday chats)" so the switcher and the browser
tell the same story: Default = your ordinary chats; named workspaces =
grouped contexts. The Default workspace is also protected from rename and
archive (ADR'd with TASK-714) so the anchor stays stable.

## Consequences

- Browser grouping, switcher copy, and the empty-state copy agree.
- Users searching for "where did my old chats go" after adopting workspaces
  find them where they always were: Chats.
- Any future change that surfaces Default as a named group must revisit this
  record and the empty-state semantics together.
