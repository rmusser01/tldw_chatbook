# Agent-assisted archived conversation recovery

Date: 2026-09-17

Status: Written design awaiting user review. Product choices and review corrections approved in conversation; implementation has not started.

Reviewed again: 2026-09-19. Corrected event-date matching, workspace lifecycle concurrency, capability registration, continuation behavior and receipt/result ownership before implementation planning.

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

The existing `register_builtin_library_provider` accepts exactly the Direct or RAG provider class, validates its complete name set, and permits only one such provider per registry. Do not route a second recovery provider through that method or broaden its class/name checks. Add an explicit, narrowly authenticated recovery registration/injected-runtime seam that coexists with the selected Library provider. Bind its authority to the exact owning session, run and the two reserved names, deriving Allowed from the same captured Library policy. Admission and execution both enforce interactive user-originated primary-run eligibility; being a primary agent alone does not admit scheduled or automatic goal/wake runs. Missing actor provenance fails closed.

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
- **Manual result controls:** Preview and Restore are explicit user actions, like existing Library actions. A later Blocked setting prevents agent calls on subsequent turns, but does not prohibit the user from manually reviewing/restoring local data. Manual actions do not inject newly fetched excerpts into model history or restart an agent. If an active agent is awaiting confirmation for the same operation, the user's decision settles that exact request and its bounded outcome returns normally; it does not create a second operation.

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

For each candidate, expose `conversation_archived_at` only while its chat is archived and `workspace_archived_at` only while its workspace is archived. Without date filters, order by the latest known timestamp among those currently applicable archive states. With date filters, a candidate matches when either applicable event lies in the requested interval; order proven matches by their latest event inside that interval. For example, a chat archived Monday inside a workspace archived Friday must appear in a Monday search, labeled as a chat-date match. Return the matching event basis (`chat`, `workspace` or `both`) and retain both dates when available. If an applicable state has an unknown time, mark chronology as incomplete; never claim that a known date proves when the other event happened. These fields describe the current archive episode, not a historical event ledger: a restored chat's former archive date is not searchable.

Rows with no known applicable archive time sort after dated rows, with deterministic last-modified/identity tie breakers explicitly labeled as such, not as archive dates. When `include_unknown_archive_times` is true, append otherwise excluded rows with unknown applicable events after the proven matches; label those rows “Archive time unknown; may match this date range.” A row with a proven in-range event appears only once even if another applicable event is undated. The response separately reports whether text/workspace-eligible candidates have undated applicable states and offers “Include unknown archive times”; undated events must not be silently represented as dated matches.

Text matching covers titles, keywords and non-deleted user/visible-assistant message text using existing parameterized literal/FTS boundaries. Do not search system prompts, hidden reasoning or tool payloads through this recovery capability. Both match predicates and snippet projections must enforce this restriction; hiding a snippet after unrestricted FTS matching would still leak that excluded content matched. Results are distinct conversations, ordered by archive recency even when a text query is supplied. A nonblank query gets a bounded matching excerpt where possible; an empty query gets a bounded recent user/assistant excerpt. If a match comes from an inactive branch, label that fact and retain its message identity as match provenance; restoration still preserves the original active branch. Preview opens the existing reader without promising an unsupported branch/message jump. Do not expose image BLOBs or full transcripts in discovery snippets.

Each result contains stable conversation and workspace identities, a result reference, title, workspace label, chat/workspace archive flags, relevant timestamps, message-match context and permitted next actions. Returning workspace identity makes the optional follow-up workspace filter usable without guessing names. Capture conversation version and the workspace lifecycle snapshot internally for later revalidation. Missing workspace metadata must be distinguished from Default; retain a chat-archived candidate with an unavailable-workspace label and Preview, but offer no recovery mutation until its ownership can be resolved. A deleted or unavailable explicit workspace filter returns `unavailable`, not an empty success or fallback to all workspaces.

Cap snippets at 320 characters and each serialized tool response at 32 KiB, with explicit truncation and continuation indicators. Byte fitting cannot discard entries and then advance the cursor past them. Fetch only bounded text projections for the returned page; never materialize every matching transcript and trim afterward.

Use keyset continuation over the complete deterministic sort tuple, including known/unknown match group and conversation identity, rather than offsets into a changing list. Bind cursors to the originating session, normalized filters and search expiry. Continuation may reobserve changing local state; it is not a frozen archive snapshot or an exact point-in-time total. If the captured workspace lifecycle inputs change, return `stale` and offer Refresh instead of mixing different workspace inventories. New/reordered conversations ahead of the cursor require Refresh; immutable old cards keep their identities. Do not claim a complete empty archive when a time-bounded or resource-limited search stops early. Apply an interruptible five-second storage budget and return `search_incomplete` with refinement/retry guidance if exceeded; UI work stays off-loop. The page/byte limits bound materialization, not the cost of a full text scan.

Eligibility, workspace and date filtering must occur before pagination. Workspace-only archives must not be filtered out by querying only `conversations.archived = 1`, nor discovered by enriching an already-limited active-conversation page. Capture workspace lifecycle inputs for the query and perform bounded storage-side candidate selection. The two databases are not a globally atomic snapshot: results describe observations at search time and every mutation is revalidated. Missing/failed workspace enumeration is an explicit incomplete-search error, not a successful empty result.

## 5. Stable results and presentation

Render a native result card from validated structured tool output, with numbered rows, lifecycle labels, Preview and the applicable recovery action. Agent prose can summarize candidates but cannot manufacture actionable links, flags or confirmation controls. Use the governing design tokens, existing tool-result/card patterns, compact layouts and keyboard conventions; no new screen is required.

The runtime stores immutable result sets keyed by source session and result-set identity. Every returned row has an opaque reference that resolves to that captured row. A later search, page or list refresh creates a new set without renumbering an earlier card. `request_conversation_restore` accepts a result reference, not an ordinal or an arbitrary conversation ID. The model resolves “the second one” against the referenced card; if multiple cards make the request ambiguous, it asks which one.

Retain at most 10 result pages per source session for 30 minutes from creation; evict oldest first. Closing the source session or restarting the app expires its references and cursors. A selected row is pinned in a separately bounded pending-confirmation record while the card is awaiting a decision, so ordinary page eviction cannot change its target. At most one recovery confirmation is active per source session; a different concurrent request reports `busy` and leaves the existing card intact. An unanswered confirmation expires five minutes after creation, including when the source chat is in the background; expiry releases its pinned row and requires a fresh request.

Old transcript cards may remain visible, but expired actions offer Refresh rather than silently resolving the old ordinal to new results. A manual refresh revalidates stored filters through the manual action path and creates new references. Preview opens the existing bounded, read-only Library conversation reader by exact identity; it does not restore, stage source context or send text to the model.

Manual refresh results are local UI state, not a new model tool result. If the user then refers conversationally to that refreshed card, the agent must perform an authorized fresh search before selecting a target it has not observed. Imported, copied or model-authored text resembling a recovery result must never rehydrate live result references or mutation controls. After restart, offer the existing manual Library search route unless app-owned provenance can validate the old card's filters; never parse arbitrary transcript JSON into actionable authority.

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

At confirmation submission, revalidate again and consume one-shot authorization for the exact target, scope and optional name. Conversation restoration keeps the existing version-checked mutation. Workspace restoration must check its captured lifecycle identity inside the write transaction, using workspace identity, archive state, the monotonic `archive_revision` defined below and the relevant name snapshot; a preflight read alone is insufficient. A timestamp is display data, never a concurrency token. This rejects archive→restore→archive cycles even when the clock repeats or moves backward. Already-applied outcomes are recognized by inspecting current state, not by claiming that this request performed them.

For a two-part operation, restore the workspace first, then restore the conversation using its captured version. If either preflight fails, do not begin. Because the stores are separate, a concurrent edit can still cause the second write to fail. Preserve and report the actual completed scope; never automatically rearchive a workspace or move a conversation to compensate. Retry obtains fresh state and confirmation for any remaining mutation.

The same cross-store limitation applies to workspace-only recovery: the selected conversation can be moved or deleted after its last preflight even though the confirmed workspace restore succeeds. Re-read target ownership/lifecycle after writes and distinguish what this operation changed from the current observed availability. Never claim the selected chat is accessible if its workspace was rearchived, it moved, or it disappeared concurrently; report the completed workspace effect and a stale/unavailable target. This is an explicit concurrency limit, not a promise of atomic recovery across databases.

## 7. Operation ownership, results and navigation

Confirmation waiting uses the established interruptible runtime pattern. The main agent may wait for the user's decision and receive a structured outcome; a manual result button can also operate after the agent turn has finished. Neither path depends on a still-running model call for storage or UI completion.

Before a write starts, Deny, cancellation, expired confirmation, source-session closure or unavailable UI means no mutation. Once the coordinator admits the approved operation and starts storage, it owns completion independently of the awaiting tool or screen. A late cancellation stops further unstarted writes and reports whichever writes actually committed. It does not promise rollback.

Deduplicate submissions by app-owned operation identity. Bind agent requests to `(source session, run ID, native tool-call ID, normalized argument digest)` and manual requests to an app-generated action identity; call IDs alone may repeat across turns. Duplicate events for the same operation return its pending/final outcome; reused call IDs with changed arguments are refused. The same card/target cannot start a second operation while one is pending or executing, including through the other entry point. A new tool-call identity requires a new confirmation and cannot consume an earlier approval. Keep at most 20 completed in-memory presentation receipts per source session, with per-resource outcomes and sanitized errors; retain active operations until settlement and evict completed receipts oldest first. Consumed confirmation objects never become usable through receipt eviction. A replay whose outcome is no longer retained reports `expired` and requires a new request/confirmation; it never silently executes again. Existing run-owned call settlement tracks identities until run teardown; late callbacks after teardown are rejected. Receipts are presentation data, not permission. If the source view closes while storage finishes, retain the app-owned completion and surface a notification; do not reopen either conversation automatically. Release closed-session state after active operations settle; app shutdown joins started storage before closing its database handles.

Structured outcomes distinguish `restored`, `already_available`, `declined`, `expired`, `busy`, `cancelled`, `stale`, `not_found`, `name_conflict`, `unavailable`, `search_incomplete`, `partial` and `storage_error`. Include the actual chat/workspace outcomes and valid next action. Avoid a generic success result when only one part succeeded. Denial ends that request; do not reopen the same confirmation automatically.

No durable permission or replay token is created. After a process restart, old references cannot execute. A new search observes durable state, including a workspace that was restored before interruption; new confirmation covers only remaining work.

After success, Open conversation is an explicit user action through ADR-147's original-ID resume handoff. It rechecks current lifecycle state, reuses an existing session where possible, preserves the original active branch and unrelated drafts, and retains existing navigation ownership checks. If the workspace is still archived, Open offers its explicit recovery instead of silently restoring it. Restoration itself never activates a workspace, resumes agents or sends queued work.

## 8. Persistence changes

Add nullable `archived_at` fields to the conversation store and workspace store through their next available migrations; do not edit historical migrations or assume the current schema version from old guidance. Existing archives migrate with NULL timestamps. Existing active records also start NULL.

Set the timestamp to the observed UTC wall clock atomically on each successful active→archived transition, leave it unchanged for no-op repeats, and clear it on archived→active. Do not advance it beyond last-modified to manufacture uniqueness: imported/future timestamps and clock rollback can otherwise produce misleading archive dates. Chronology is limited by the device clock; optimistic concurrency is independent of it.

Add a local nonnegative integer `archive_revision` to workspace records, seeded to 0 during the same migration and incremented atomically on every actual archive or restore transition, including Undo and existing UI paths. A workspace archive write must condition on the expected prior state/revision so concurrent requests cannot overwrite the episode timestamp. Keep the existing conversation version as its concurrency token; do not add a redundant conversation counter. All writers of workspace archive state must preserve the revision invariant. This small counter replaces the fragile plan to use timestamps as episode identities. Add indexes appropriate to archive eligibility/order and verify query plans on representative data.

Conversation timestamps follow the existing local-only archive contract: no new sync payload field, no archive-only sync event, and no overwrite by shared-payload synchronization. Update the current schema's trigger definitions through the new migration as needed, preserving version advancement and retention of the latest shared payload under ADR-147. Workspace archive timestamps remain local lifecycle state. Do not backfill from last-modified or created timestamps.

Use durable `conversations.workspace_id` ownership, including Default normalization, rather than relying on a potentially lagging workspace membership projection. Workspace metadata enrichment is batched/off-loop under the existing in-memory SQLite ownership exception. Lifecycle failures never rewrite ownership to Default to make recovery succeed.

## 9. Integration boundaries

| Existing area | Responsibility in this change |
| --- | --- |
| `DB/ChaChaNotes_DB.py`, migrations | Conversation timestamp, archive query projections/order, version-checked writes and sync invariants. |
| `Workspaces/models.py`, `registry_service.py`, WorkspaceDB migrations | Workspace timestamp, archive revision and transaction-level conditional lifecycle changes. |
| `Chat/chat_conversation_service.py`, `conversation_archive_actions.py` | Shared storage lifecycle operations and bounded recovery service composition. |
| Agent catalog, agent runtime and `Chat/console_agent_bridge.py` | Authenticated discovery, authority, primary-run eligibility and injected interactive calls. |
| Console runtime/store and view-hook declarations | Result/confirmation ownership, exact completion delivery, cancellation and receipts. |
| Console tool-result/confirmation widgets | Structured result cards and keyboard-accessible user actions composed from design tokens. |
| Existing Library reader and Console archive/resume modules | Read-only preview, Restore as, and explicit exact conversation opening. |

Keep recovery query/operation logic out of the already-large screen and bridge modules. Add a focused service/coordinator where needed; do not introduce a general action framework or a second archive inventory. Any added view hook must be declared in `CONSOLE_VIEW_HOOK_SLOTS` and covered by the existing slot-set test.

## 10. Verification and documentation

Run targeted tests only. Implementation qualification must prove:

1. Real SQLite fresh/upgraded databases preserve archive state, versions, NULL historical times and sync-log invariants; repeated/backward clocks and future modification timestamps do not bypass workspace archive revisions or falsify event times.
2. All four chat/workspace state combinations, Default, missing workspace metadata, deleted records, duplicate titles, keyword/message matches and empty queries produce correct discovery/actions.
3. Filtering precedes pagination; workspace-only candidates are found; either applicable event can satisfy a date range, with honest unknown-time handling, timezone boundaries, ties and byte fitting. Continuation under conversation changes, stale workspace inputs and storage-budget exhaustion never claims a frozen/complete snapshot.
4. Allowed versus Blocked and Direct versus RAG-only behave as specified, including the real registry's existing exact-class/single-provider checks. No embeddings load, spoofed providers, scheduled/goal-wake runs, subagent mutation, temporary mutation or shared MCP exposure.
5. Both a real result-button event and a conversational tool request reach the production confirmation coordinator and change only the captured resource identities.
6. Wrong-session/expired references, changing result sets, stale versions, rehoming, deleted targets, workspace archive cycles and name collisions never restore an unintended target.
7. Duplicate submissions across both entry points, reused native call IDs across runs, changed arguments, receipt eviction, Deny, cancellation before admission, cancellation after a committed workspace write, UI navigation, source-session closure and app shutdown produce truthful completion/partial receipts without replay. A selected chat changing during workspace-only recovery does not become a false success.
8. Restore-only preserves current workspace/session/draft; explicit Open resumes the original identity and active branch and preserves unrelated drafts.
9. Mounted compact/wide UI tests exercise actual keyboard dispatch, result rendering, feedback and Restore as. Run view-hook slot coverage and design-token/CSS checks for modified surfaces.
10. Excluded system/reasoning/tool text cannot influence discovery matching or snippets; imported fake cards cannot create authority. Manual refresh remains outside model history and a conversational follow-up cannot target an unseen renumbered row.

Validate affected lint/format checks and perform a live app check with disposable local data: archive a saved chat, find it from another workspace, restore it through each entry point, and verify the original conversation opens. Automated fakes do not replace this interaction check. No provider message should be sent merely by restoring/opening.

Update the Console tool, Library-access and sessions/workspaces user guides with the RAG-only recovery exception, unknown-time behavior and recovery choices. Document that “not found” refers to the searched local archive scope: a closed active chat or remote-only chat is not evidence of deletion. Offer the existing manual Library route rather than silently broadening the agent's search.

## 11. ADR check and handoff

ADR required: yes

ADR path: `backlog/decisions/166-agent-assisted-archive-recovery.md`

Reason: new Console agent contracts, a narrow amendment to Library authority/retrieval selection, archive chronology in two stores, and confirmation/operation ownership across UI and runtime.

This document is a design, not an implementation plan. After user review of the written spec, invoke writing-plans to define atomic implementation tasks and targeted checks. No application code or schema migration is implemented by this design task.
