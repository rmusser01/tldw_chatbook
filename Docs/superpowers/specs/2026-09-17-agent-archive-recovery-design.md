# Agent-assisted archived conversation recovery

Date: 2026-09-17

Status: Written design awaiting user review. Product choices and review corrections approved in conversation; implementation has not started.

Task: [TASK-32772](../../../backlog/tasks/task-32772%20-%20Design-agent-assisted-archived-conversation-recovery.md)

Decision: [ADR-166](../../../backlog/decisions/166-agent-assisted-archive-recovery.md)

## 1. Purpose and approved experience

A user accidentally archives a conversation and remembers neither its title nor its workspace. They can ask the current Console agent to find recently archived conversations, identify the missing chat from bounded results, and choose to restore it. Both a result's Restore button and a follow-up such as “restore the second one” lead to the same exact-target confirmation.

Search covers local saved conversations across Default, active workspaces and archived workspaces. It includes a conversation when the chat itself is archived or its workspace is archived. Those are separate lifecycle states, visibly labeled. Soft-deleted conversations are excluded; this is not Trash recovery.

Restoration preserves conversation identity, messages, branches and workspace membership. It never sends a message or switches the current conversation. Opening the original chat is a separate user action through the existing exact-resume flow.

Approved refinements:

- Offer chat-only and chat-plus-workspace recovery when both are archived.
- Record real archive times for both conversations and workspaces; historical times remain unknown.
- Honor Assistant Library access. When Allowed, dedicated recovery works in both Direct and RAG-only modes without an embedding index.
- Bind actions to stable result identities and captured lifecycle state.
- Report partial writes, stale selections, cancellation and already-restored state honestly.
- Keep v1 local, single-target, main-agent and interactive. No bulk, remote, external MCP, subagent or unattended recovery capability.

## 2. Architecture and alternatives

Use two focused Console tools, `search_archived_conversations` and `request_conversation_restore`, over a shared recovery service. The service composes the existing conversation archive operations and workspace registry. The runtime owns result references, confirmation requests and operation completion; widgets render this state and dispatch typed actions.

Search uses database queries and the existing lexical-search semantics. It does not use RAG, embeddings or the general file tools. Restoration uses the Console runtime's injected-callable/confirmation seam, following the existing chat-creation pattern, without inheriting its remembered-approval option.

Both tool entry points are registered as authenticated, reserved Console capabilities. They are not added to the shared Library descriptor list that also exposes tools through MCP. The main agent can discover them through the normal tool catalog; skills and MCP providers cannot impersonate their names when the built-in capabilities are unavailable.

Alternatives considered:

| Approach | Decision |
| --- | --- |
| Dedicated recovery tools backed by existing services | Chosen: clear purpose, narrow archive-only exception to RAG mode, explicit interactive confirmation. |
| Add archive parameters and restoration to general Library tools | Rejected for this scope: widens the general retrieval contract and risks exposing a Console mutation through shared MCP descriptors. |
| Agent only opens Library search | Existing fallback, but insufficient as the main flow: the requested findings and conversational recovery should be available in chat. |
| Use the current restore-and-resume handler directly | Rejected: it also navigates. Reuse/extract lifecycle work while leaving activation in the existing resume path. |

## 3. Authority and data exposure

Follow ADR-079's app-issued, immutable authority captured at actual turn execution, including queued turns and resolved model-destination disclosure. The recovery provider must validate this authority, not trust a tool source string. All applicable tool kill switches, persona restrictions and runtime eligibility checks continue to apply.

- **Assistant access Blocked:** neither recovery tool is available to the agent. Existing manual Library recovery remains available. The UI explains the block and links to the existing Library-access control; it never silently enables access.
- **Assistant access Allowed:** advertise the narrow recovery capability in Direct and RAG-only modes. The general Library Direct-versus-RAG selection is otherwise unchanged.
- **Temporary sessions:** existing exact-name/read-only restrictions remain authoritative. No recovery mutation is admitted by adding a broad provider whitelist. V1 exposes these interactive recovery tools from ordinary local Console sessions, not temporary/headless sessions.
- **Agent restoration:** an app-owned, one-shot confirmation for the selected target and scope is mandatory every time. Persistent Allow or a session-level remembered decision cannot substitute for it. No UI, cancellation or timeout means no new write.
- **Manual result controls:** Preview and Restore are explicit user actions, like existing Library actions. A later Blocked setting prevents agent calls on subsequent turns, but does not prohibit the user from manually reviewing/restoring local data. Manual actions do not send new transcript content to a model or restart an agent.

Cross-workspace recovery is an explicit exception to the current chat's retrieval item filters: it searches the local recovery inventory, not that chat's selected RAG sources. Library access is still required for agent search. The result header states “All local workspaces” or the explicitly requested workspace filter.

Titles, workspace names, snippets and archived messages are untrusted data. Escape terminal markup/control characters and teach the model to treat returned content as evidence, never instructions. Existing destination disclosure applies when snippets go to a cloud/private-network model. Do not put excerpts or transcript bodies in diagnostic logs, activity metadata or recovery receipts. Normal tool-result history follows existing retention policy; it is not advertised as local-only.

## 4. Search contract and chronology

`search_archived_conversations` accepts:

| Argument | Contract |
| --- | --- |
| `query` | Optional literal lexical text, default empty; maximum 512 characters. Empty means list recovery candidates. |
| `workspace_id` | Optional validated stable local workspace identity; omission searches all workspaces including Default. Display names are not mutation identities. |
| `archived_after`, `archived_before` | Optional timezone-qualified timestamps, inclusive lower/exclusive upper bounds, normalized to UTC. Invalid or reversed ranges are refused. |
| `include_unknown_archive_times` | Boolean, default false. With a date filter, also include otherwise eligible rows with an unknown applicable archive event, labeled as possible rather than proven date matches. Without a date filter, unknown-time rows are always included. |
| `limit` | Default 10, maximum 20, positive integer. |
| `cursor` | Opaque continuation bound to the normalized filters and order. Mismatched or expired cursors require a fresh search. |

The default list has no arbitrary “last seven days” cutoff: it returns the newest known archive events first. For date phrases such as “today,” use the user's local timezone to resolve the interval and show that interval in the result summary.

For each candidate, expose `conversation_archived_at` only while its chat is archived and `workspace_archived_at` only while its workspace is archived. Its ordering/filter key is the latest known timestamp among those currently applicable archive states. Return the basis (`chat`, `workspace` or `both`) and retain both dates when available. If an applicable state has an unknown time, mark chronology as incomplete; never claim that a known date proves when the other event happened.

Rows with no known applicable archive time sort after dated rows, with deterministic last-modified/identity tie breakers explicitly labeled as such, not as archive dates. Date filters match the known ordering key. When `include_unknown_archive_times` is true, append otherwise excluded rows with unknown applicable events after the proven matches; label those rows “Archive time unknown; may match this date range.” The response separately reports whether text/workspace-eligible candidates have undated applicable states and offers “Include unknown archive times”; undated events must not be silently represented as dated matches.

Text matching covers titles, keywords and non-deleted user/visible-assistant message text using existing parameterized literal/FTS boundaries. Do not search system prompts, hidden reasoning or tool payloads through this recovery capability. Results are distinct conversations, ordered by archive recency even when a text query is supplied. A nonblank query gets a bounded matching excerpt where possible; an empty query gets a bounded recent user/assistant excerpt. If a match comes from an inactive branch, label that fact and retain its message identity for preview; restoration still preserves the original active branch. Do not expose image BLOBs or full transcripts in discovery snippets.

Each result contains a stable conversation identity, a result reference, title, workspace label, chat/workspace archive flags, relevant timestamps, message-match context and permitted next actions. Capture conversation version and the workspace lifecycle snapshot internally for later revalidation. Missing workspace metadata must be distinguished from Default; retain a chat-archived candidate with an unavailable-workspace label, but refuse a workspace restoration whose identity cannot be resolved.

Cap snippets at 320 characters and each serialized tool response at 32 KiB, with explicit truncation and continuation indicators. Byte fitting cannot discard entries and then advance the cursor past them. Fetch only bounded text projections for the returned page; never materialize every matching transcript and trim afterward.

Eligibility, workspace and date filtering must occur before pagination. Workspace-only archives must not be filtered out by querying only `conversations.archived = 1`, nor discovered by enriching an already-limited active-conversation page. Capture workspace lifecycle inputs for the query and perform bounded storage-side candidate selection. The two databases are not a globally atomic snapshot: results describe observations at search time and every mutation is revalidated. Missing/failed workspace enumeration is an explicit incomplete-search error, not a successful empty result.

## 5. Stable results and presentation

Render a native result card from validated structured tool output, with numbered rows, lifecycle labels, Preview and the applicable recovery action. Agent prose can summarize candidates but cannot manufacture actionable links, flags or confirmation controls. Use the governing design tokens, existing tool-result/card patterns, compact layouts and keyboard conventions; no new screen is required.

The runtime stores immutable result sets keyed by source session and result-set identity. Every returned row has an opaque reference that resolves to that captured row. A later search, page or list refresh creates a new set without renumbering an earlier card. `request_conversation_restore` accepts a result reference, not an ordinal or an arbitrary conversation ID. The model resolves “the second one” against the referenced card; if multiple cards make the request ambiguous, it asks which one.

Retain at most 10 result pages per source session for 30 minutes from creation; evict oldest first. Closing the source session or restarting the app expires its references and cursors. A selected row is pinned in a separately bounded pending-confirmation record while the card is awaiting a decision, so ordinary page eviction cannot change its target. At most one recovery confirmation is active per source session; a different concurrent request reports `busy` and leaves the existing card intact. An unanswered confirmation expires five minutes after creation, including when the source chat is in the background; expiry releases its pinned row and requires a fresh request.

Old transcript cards may remain visible, but expired actions offer Refresh rather than silently resolving the old ordinal to new results. A manual refresh revalidates stored filters through the manual action path and creates new references. Preview opens the existing bounded, read-only Library conversation reader by exact identity; it does not restore, stage source context or send text to the model.

## 6. Confirmation and restoration

`request_conversation_restore` accepts `result_ref` and optional `scope` (`chat_only`, `workspace_only`, `chat_and_workspace`). A supplied scope selects a valid choice in the confirmation; it never grants authority. An omitted scope presents the valid choices. Refuse an incompatible scope before writing.

| Chat state | Workspace state | Confirmation choices |
| --- | --- | --- |
| Archived | Active or Default | Restore chat |
| Archived | Archived | Restore chat only; Restore chat and workspace |
| Active | Archived | Restore workspace |
| Active | Active or Default | Already available; offer Open, with no write |

The result button and agent call use the same confirmation coordinator and recovery operation. The card shows the current title, workspace, intended effect and any approved replacement workspace name. For chat-only recovery under an archived workspace, say that the workspace remains archived and Open will require its restoration. For workspace recovery, say that the workspace and its other active chats become visible; independently archived chats stay archived.

If a workspace name has been reused, collect a replacement using the existing Restore as interaction. Bind that name to the final confirmation and check name uniqueness in the same workspace transaction as restoration. An edited name invalidates the previous confirmation; never automatically choose or overwrite another workspace.

Before showing confirmation, compare the result with current durable state. A changed version, changed ownership, changed workspace lifecycle or missing/deleted target requires refresh. A chat already available can be reported without mutation. Do not silently upgrade an old chat-only request to a workspace restore.

At confirmation submission, revalidate again and consume one-shot authorization for the exact target, scope and optional name. Conversation restoration keeps the existing version-checked mutation. Workspace restoration must check its captured lifecycle identity inside the write transaction, using the workspace identity, archive state, archive-event timestamp and relevant name snapshot; a preflight read alone is insufficient. This also rejects archive→restore→archive cycles that return to the same boolean state. Already-applied outcomes are recognized by inspecting current state, not by claiming that this request performed them.

For a two-part operation, restore the workspace first, then restore the conversation using its captured version. If either preflight fails, do not begin. Because the stores are separate, a concurrent edit can still cause the second write to fail. Preserve and report the actual completed scope; never automatically rearchive a workspace or move a conversation to compensate. Retry obtains fresh state and confirmation for any remaining mutation.

## 7. Operation ownership, results and navigation

Confirmation waiting uses the established interruptible runtime pattern. The main agent may wait for the user's decision and receive a structured outcome; a manual result button can also operate after the agent turn has finished. Neither path depends on a still-running model call for storage or UI completion.

Before a write starts, Deny, cancellation, expired confirmation, source-session closure or unavailable UI means no mutation. Once the coordinator admits the approved operation and starts storage, it owns completion independently of the awaiting tool or screen. A late cancellation stops further unstarted writes and reports whichever writes actually committed. It does not promise rollback.

Deduplicate submissions by app-owned operation identity. Double clicks, duplicate confirmation events and retries carrying the same native tool-call identity return that operation's pending/final outcome. A new tool-call identity requires a new confirmation and cannot consume an earlier approval. Keep at most 20 completed in-memory receipts per source session, with per-resource outcomes and sanitized errors; retain active operations until settlement and evict completed receipts oldest first. Receipts are presentation/deduplication data, not permission. If the source view closes while storage finishes, retain the app-owned completion and surface a notification; do not reopen either conversation automatically.

Structured outcomes distinguish `restored`, `already_available`, `declined`, `expired`, `stale`, `not_found`, `name_conflict`, `unavailable`, `partial` and `storage_error`. Include the actual chat/workspace outcomes and valid next action. Avoid a generic success result when only one part succeeded. Denial ends that request; do not reopen the same confirmation automatically.

No durable permission or replay token is created. After a process restart, old references cannot execute. A new search observes durable state, including a workspace that was restored before interruption; new confirmation covers only remaining work.

After success, Open conversation is an explicit user action through ADR-147's original-ID resume handoff. It rechecks current lifecycle state, reuses an existing session where possible, preserves the original active branch and unrelated drafts, and retains existing navigation ownership checks. If the workspace is still archived, Open offers its explicit recovery instead of silently restoring it. Restoration itself never activates a workspace, resumes agents or sends queued work.

## 8. Persistence changes

Add nullable `archived_at` fields to the conversation store and workspace store through their next available migrations; do not edit historical migrations or assume the current schema version from old guidance. Existing archives migrate with NULL timestamps. Existing active records also start NULL.

Set the timestamp atomically on each successful active→archived transition, leave it unchanged for no-op repeats, and clear it on archived→active. Use a UTC clock value advanced beyond the entity's previous modification timestamp when necessary so a rapid archive cycle cannot reuse the lifecycle timestamp. Restore and Undo paths must obey the same rule. Add indexes appropriate to archive eligibility/order and verify query plans on representative data.

Conversation timestamps follow the existing local-only archive contract: no new sync payload field, no archive-only sync event, and no overwrite by shared-payload synchronization. Update the current schema's trigger definitions through the new migration as needed, preserving version advancement and retention of the latest shared payload under ADR-147. Workspace archive timestamps remain local lifecycle state. Do not backfill from last-modified or created timestamps.

Use durable `conversations.workspace_id` ownership, including Default normalization, rather than relying on a potentially lagging workspace membership projection. Workspace metadata enrichment is batched/off-loop under the existing in-memory SQLite ownership exception. Lifecycle failures never rewrite ownership to Default to make recovery succeed.

## 9. Integration boundaries

| Existing area | Responsibility in this change |
| --- | --- |
| `DB/ChaChaNotes_DB.py`, migrations | Conversation timestamp, archive query projections/order, version-checked writes and sync invariants. |
| `Workspaces/models.py`, `registry_service.py`, WorkspaceDB migrations | Workspace timestamp and transaction-level conditional restoration. |
| `Chat/chat_conversation_service.py`, `conversation_archive_actions.py` | Shared storage lifecycle operations and bounded recovery service composition. |
| Agent catalog, agent runtime and `Chat/console_agent_bridge.py` | Authenticated discovery, authority, primary-run eligibility and injected interactive calls. |
| Console runtime/store and view-hook declarations | Result/confirmation ownership, exact completion delivery, cancellation and receipts. |
| Console tool-result/confirmation widgets | Structured result cards and keyboard-accessible user actions composed from design tokens. |
| Existing Library reader and Console archive/resume modules | Read-only preview, Restore as, and explicit exact conversation opening. |

Keep recovery query/operation logic out of the already-large screen and bridge modules. Add a focused service/coordinator where needed; do not introduce a general action framework or a second archive inventory. Any added view hook must be declared in `CONSOLE_VIEW_HOOK_SLOTS` and covered by the existing slot-set test.

## 10. Verification and documentation

Run targeted tests only. Implementation qualification must prove:

1. Real SQLite fresh/upgraded databases preserve archive state, versions, NULL historical times and sync-log invariants; rapid archive cycles create distinct lifecycle stamps.
2. All four chat/workspace state combinations, Default, missing workspace metadata, deleted records, duplicate titles, keyword/message matches and empty queries produce correct discovery/actions.
3. Filtering precedes pagination; workspace-only candidates are found; date boundaries, local timezone conversion, incomplete chronology, deterministic ties and response-byte fitting are honest.
4. Allowed versus Blocked and Direct versus RAG-only behave as specified. No embeddings load, spoofed providers, subagent mutation, temporary mutation or shared MCP exposure.
5. Both a real result-button event and a conversational tool request reach the production confirmation coordinator and change only the captured resource identities.
6. Wrong-session/expired references, changing result sets, stale versions, rehoming, deleted targets, workspace archive cycles and name collisions never restore an unintended target.
7. Duplicate submissions, Deny, cancellation before admission, cancellation after a committed workspace write, UI navigation and source-session closure produce truthful completion/partial receipts without replay.
8. Restore-only preserves current workspace/session/draft; explicit Open resumes the original identity and active branch and preserves unrelated drafts.
9. Mounted compact/wide UI tests exercise actual keyboard dispatch, result rendering, feedback and Restore as. Run view-hook slot coverage and design-token/CSS checks for modified surfaces.

Validate affected lint/format checks and perform a live app check with disposable local data: archive a saved chat, find it from another workspace, restore it through each entry point, and verify the original conversation opens. Automated fakes do not replace this interaction check. No provider message should be sent merely by restoring/opening.

Update the Console tool, Library-access and sessions/workspaces user guides with the RAG-only recovery exception, unknown-time behavior and recovery choices. Document that “not found” refers to the searched local archive scope: a closed active chat or remote-only chat is not evidence of deletion. Offer the existing manual Library route rather than silently broadening the agent's search.

## 11. ADR check and handoff

ADR required: yes

ADR path: `backlog/decisions/166-agent-assisted-archive-recovery.md`

Reason: new Console agent contracts, a narrow amendment to Library authority/retrieval selection, archive chronology in two stores, and confirmation/operation ownership across UI and runtime.

This document is a design, not an implementation plan. After user review of the written spec, invoke writing-plans to define atomic implementation tasks and targeted checks. No application code or schema migration is implemented by this design task.
