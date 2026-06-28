# ADR-010: Console Conversation Local Marks

Status: Accepted
Date: 2026-06-27
Related Task: N/A
Supersedes: N/A

## Decision

Console conversation stars will be stored as durable local-only conversation marks in a dedicated local marks table, separate from normalized conversation metadata, workspace memberships, Sync v2 payloads, server payloads, and chat metadata mirror reports.

## Context

Console needs a quick-access `Starred` conversation section above workspace and unscoped conversation groups. The user wants stars to survive app restarts, but explicitly does not want them treated as sync items.

The existing `conversations` table contains durable chat metadata that is already involved in local conversation reads, future sync/mirror discussions, and server conversation contracts. Adding a `starred` column directly to `conversations` would make future sync and mirror code more likely to accidentally treat stars as portable conversation metadata.

The existing Console workspace design also keeps active workspace context separate from broad browsing. Stars are user organization metadata for this Chatbook instance, not workspace ownership, not handoff eligibility, and not server collaboration state.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Store stars in app config | App config would make stars durable but fragile, hard to validate against deleted conversations, and detached from conversation identity. |
| Add `starred` to `conversations` | This would be easy to query but would blur local-only UI organization with syncable conversation metadata. |
| Store stars as workspace memberships | Starred is a cross-workspace quick-access mark, not workspace ownership or eligibility. |
| Defer stars and only group conversations by workspace | This misses the quick-access requirement and would require another pass through the same Console rail. |
| Sync stars as user preferences | The user explicitly said stars are local metadata and not intended to be a sync item. |

## Consequences

Implementation should introduce a local service, tentatively `ConversationLocalMarksService`, backed by a table such as `conversation_local_marks(conversation_id, mark_type, created_at, updated_at)`.

The initial mark type is `starred`. The table may support future local-only marks, but new marks must remain local unless a future ADR changes that boundary.

Conversation marks must not be included in:

- normalized conversation row contracts
- Sync v2 chat metadata outbox records
- chat metadata mirror reports
- server conversation create/update/list payloads
- workspace memberships or handoff manifests

Console and Library may read marks as local enrichment if needed, but they must treat marks as optional local state. If mark storage is unavailable, normal conversation browsing and resume must still work.

Deleted or missing conversations must not break the `Starred` browser section. Implementations may omit orphan marks from render output and may clean them up opportunistically.

If Chatbook later decides to sync conversation stars, that must be a new explicit ADR that revisits privacy, conflict, merge, and server contract semantics.

## Links

- [Console grouped conversation browser design](../../Docs/superpowers/specs/2026-06-27-console-grouped-conversation-browser-design.md)
- [ADR-005: Console Workspace Server-Readiness Boundary](005-console-workspace-server-readiness.md)
