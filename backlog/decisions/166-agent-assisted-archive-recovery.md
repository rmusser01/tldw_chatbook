# ADR-166: Agent-assisted local archive discovery and recovery

Date: 2026-09-17

Status: Proposed — design choices approved in conversation; written specification awaiting user review.

Review revision: 2026-09-19 — chronology and concurrency separated; event-date matching, runtime admission, continuation and replay contracts tightened.

Amends: [ADR-030](030-local-library-agent-tool-boundary.md) and [ADR-079](079-console-library-conversation-authority.md) for a narrow Console archive-recovery capability available in both retrieval modes when Assistant Library access is Allowed. Preserves [ADR-147](147-conversation-archive-and-exact-resume.md)'s local archive and exact-resume boundaries. Uses [ADR-150's chat-creation runtime pattern](150-agent-chat-fork-and-spawn.md) without remembered confirmation.

Task: [TASK-32772](../tasks/task-32772%20-%20Design-agent-assisted-archived-conversation-recovery.md)

Spec: [Agent archive recovery design](../../Docs/superpowers/specs/2026-09-17-agent-archive-recovery-design.md)

## Context

Library already searches/restores archived conversations and resumes their original identities. A user who does not remember a chat's name or workspace should also be able to ask a Console agent for recent archive candidates and choose one for recovery. An archived workspace can hide a chat whose own archive flag is false. Existing last-modified timestamps cannot reliably identify when either kind of archival occurred.

General Library tools follow a Direct-versus-RAG selector and share descriptors with external MCP. Neither a RAG-only index search nor an unconditional new Library mutation provides the intended local inventory and interactive confirmation contract.

## Decision

1. Add two reserved, authenticated Console capabilities: `search_archived_conversations` and `request_conversation_restore`. V1 is an interactive user-originated main-agent feature for ordinary local Console sessions, targeting one local saved conversation at a time. External MCP, headless, temporary, subagent and unattended scheduled/goal-wake execution do not gain this capability. Use a dedicated authenticated recovery seam alongside the selected Library provider; preserve the existing Library registry's exact-class, exact-name and single-provider checks.
2. Honor ADR-079's per-turn Assistant Library authority and model-destination disclosure. Blocked remains blocked. When Allowed, the dedicated recovery capability is available in both Direct and RAG-only modes without an embedding index. This is an explicit archive-only exception; general Library tool exposure and RAG source selection remain unchanged. Recovery searches all local workspaces by default, independently of current retrieval item filters, and labels its scope.
3. Search durable local state with the predicate “non-deleted and (chat archived or workspace archived).” Apply eligibility/filtering before pagination, return bounded text-safe results, and distinguish chat, workspace and combined archive states. Missing workspace data is not Default. A failed workspace read cannot masquerade as complete empty results.
4. Add nullable local `archived_at` timestamps to both stores through new migrations. Existing archives retain unknown times. Set observed wall-clock time atomically on archive, clear on restore, and preserve archive-only sync isolation and version semantics. Do not manufacture unique dates from last-modified timestamps. Workspace records gain a monotonic local `archive_revision` incremented on every real archive/restore transition, including existing UI and Undo writers; conversations retain their existing version token. Without date filters, order by the latest known currently applicable archive timestamp. With filters, match either applicable event and order by the latest event in the interval; label its basis and incomplete chronology and expose a route to undated matches. These timestamps describe current archive episodes, not a historical event ledger.
5. Runtime-owned immutable result references bind source session, result set and target snapshot. Tool requests select a returned reference; model-authored or imported prose/JSON never authorizes or creates live UI actions. Bound retention, require refresh after expiration/restart, and revalidate lifecycle state before confirmation and at the write boundary. Keyset continuation is bound to filters/session and is explicitly a live view, not a frozen snapshot; stale workspace inputs require refresh and exhausted query budgets report incomplete search.
6. Both manual and agent-initiated restoration use one shared coordinator. Agent restoration requires one-shot, per-operation UI confirmation of exact chat/workspace scope and any replacement name; no remembered or persisted Allow bypass. Manual controls remain explicit user actions and do not send data to an agent. Existing manual recovery is available even when agent Library access is Blocked.
7. Offer only meaningful actions: restore chat for chat-only archival; choose chat-only or both for combined archival; restore workspace for workspace-only archival; report already available otherwise. Explain workspace-wide visibility and preserve independently archived chats. Restoring a chat alone inside an archived workspace does not make its workspace active or silently allow resume.
8. Reuse shared lifecycle storage services, with conversation version checks and transaction-level workspace archive-revision/state/name checks. Restore workspace first when both are approved, then the captured conversation. The stores are not an atomic distributed transaction. Preserve actual completed writes, report partial outcomes and retry only remaining work with fresh confirmation; never compensate by silently rearchiving or moving data. Reobserve ownership and availability after writes, including workspace-only recovery, so a concurrent move/deletion/rearchive cannot become a false claim that the selected chat is accessible.
9. Runtime ownership outlives a waiting tool or mounted view once a write begins. Deduplicate by session/run/call identity and argument digest, bridge both entry points to the same pending operation, report cancellation according to actual committed scope and stop unstarted later writes. Receipt eviction cannot revive a consumed approval; late callbacks after owner teardown are rejected. Shutdown settles started storage before database closure. Old references and approvals do not replay after restart. No general persistent action journal is introduced for this reversible operation.
10. Restoration and activation are separate. Preserve the current chat/draft/workspace; offer explicit Open through ADR-147's original-ID handoff. Do not automatically send, resume agents, copy conversation history or move ownership. Preview reuses the existing bounded read-only Library reader.

## Alternatives considered

- **RAG for recovery:** rejected because index availability/completeness cannot establish recent local archive inventory.
- **General Library descriptor mutation:** rejected because Console interaction and its authority must not accidentally become an external MCP contract.
- **Direct-mode-only recovery:** rejected because recovering an archived chat should not require changing unrelated retrieval settings after Assistant Library access is already Allowed.
- **Last-modified as archive time:** rejected because later unrelated changes reorder history and older values do not prove archive chronology.
- **Archive time as workspace concurrency token:** rejected because clocks repeat or move backward; advancing time past unrelated modification timestamps can falsify chronology. A local integer revision separates stale-write protection from display time.
- **UI-only Library handoff:** retained as fallback, but insufficient for findings and conversational selection within chat.
- **Call restore-and-resume for every action:** rejected because restoration must preserve current context and opening is a separate choice.
- **Automatic workspace restoration or chat movement:** rejected because the user must see the broader effect and original ownership must remain intact.
- **Remembered approval or cross-store rollback:** rejected because exact current consent and honest partial completion are simpler and safer than widening future authority or overwriting concurrent work.

## Consequences

Implementation spans two schema migrations (including a workspace archive revision), a narrow Console tool capability, a shared recovery coordinator and structured result/confirmation UI. It must preserve existing local-only archive sync semantics and ADR-079's authenticated authority, including queued-turn and destination behavior. Search/filter query qualification, lifecycle race tests, mounted interaction tests and a disposable-data live recovery check are required; the full test suite is not authorized by this design.

The written spec defines limits, tool arguments, lifecycle choices and error outcomes. It must be reviewed before implementation planning. Existing ADRs remain unchanged while this amendment is proposed; acceptance should add supersession/amendment metadata identifying only the affected clauses, without rewriting their original decisions.
