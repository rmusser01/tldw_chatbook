# Console Library controls: per-conversation retrieval and assistant access

**Status:** Approved design, written for review on 2026-08-22

**Task:** [TASK-19900](../../../backlog/tasks/task-19900%20-%20Make-Console-Library-controls-explicit-per-conversation.md)

**Decision:** [ADR-079](../../../backlog/decisions/079-console-library-conversation-authority.md)

**PR:** [#1933](https://github.com/rmusser01/tldw_chatbook/pull/1933)

## Summary

Console can consult a user's Library in three ways, owned by three different
actors:

| Mechanism | Actor | Authority |
| --- | --- | --- |
| **Search Library** | User | Explicit action, always available |
| **Automatic retrieval** | Application | Per-conversation Never / Automatic |
| **Library tools** | Assistant | Per-conversation Blocked / Allowed |

The current interface blurs those mechanisms. A status chip reports staged
evidence, a global switch retrieves before every send, and the assistant gets
a built-in Library provider whenever agent tools run. The global
`direct_library_tools` value is also easy to misread: `false` does not disable
Library access; ADR-030 defines it as the RAG fallback.

This design gives automatic retrieval and assistant access independent,
device-local per-conversation controls. Manual search remains reachable in all
four combinations. The existing global Direct/RAG selector stays a selector.
Assistant-initiated reads become reviewable as minimized local activity, never
as staged evidence for a future prompt.

## Goals

- Make it obvious whether the application will retrieve before the next send
  and whether the assistant may initiate Library reads during the turn.
- Preserve established behavior for conversations that predate the upgrade,
  while shipping privacy-tight defaults for new local work.
- Keep a device's model-access policy out of synchronized conversation data.
- Ensure a blocked or unreadable policy cannot be bypassed through built-in,
  Skill, MCP, queued-turn, or subagent composition.
- Give automatic retrieval a truthful pre-dispatch failure flow rather than
  silently sending an ungrounded request.
- Let users review assistant Library operations without retaining duplicate
  source bodies or feeding activity back to the model.
- Separate policy editing, manual search, staged/cited evidence, and activity
  review so each surface has one job.

## Non-goals

- A per-call approval card for every built-in Library read. The conversation
  policy is the authorization boundary; MCP and other local tools keep their
  existing independent approval systems.
- Per-conversation selection among Direct and RAG. The existing global
  `direct_library_tools` selector remains authoritative when access is Allowed.
- Configurable automatic-retrieval source categories. Automatic retrieval has
  one fixed, predictable category set in this release.
- Synchronizing or exporting the policy as part of a Chatbook/conversation.
- Replacing canonical CitationTrace work from ADR-024.
- Reworking Library retrieval ranking, indexing, or source-resolution policy.
- Removing the dead legacy `get_rag_context_for_chat` helper.

## Delivery decomposition

The design is one coherent authority model, but implementation is too broad
for one atomic PR. Delivery is dependency-ordered so each review has one
primary failure surface:

| Task | Deliverable |
| --- | --- |
| TASK-19900.1 | Device-local policy schema, initialization, CAS, and session/promotion lifecycle |
| TASK-19900.2 | Immutable turn authority, provider absence/selection, and permanent name reservation |
| TASK-19900.3 | Fixed-category automatic retrieval and pause/recovery send gate |
| TASK-19900.4 | Two-axis chip, policy/search split, source density, responsive/focus behavior |
| TASK-19900.5 | Minimized activity capture, projection, Inspector, and export redaction |
| TASK-19900.6 | Documentation, integrated targeted gates, mutation checks, and live qualification |

No child task is marked In Progress until the written spec and detailed
implementation plan are approved. Dependencies point only backward; storage
lands before runtime/UI, and qualification lands after every product slice.

## 1. User model

### 1.1 Three mechanisms, not one mode

Manual search is not a state. It is an explicit user action and remains
available whether automatic retrieval is Never or Automatic and whether the
assistant is Blocked or Allowed.

The two policy axes produce four useful states:

| Automatic retrieval | Assistant access | What happens |
| --- | --- | --- |
| Never | Blocked | Only the user can run Search Library. |
| Automatic | Blocked | The app prepares evidence before eligible sends; the assistant cannot initiate Library reads. |
| Never | Allowed | No pre-send retrieval; the assistant may use the selected Direct/RAG provider during its turn. |
| Automatic | Allowed | The app may stage evidence before dispatch and the assistant may make additional Library reads during the turn. |

This is deliberately not an Off/Manual/Auto enum. Such an enum cannot express
the middle two combinations and makes “Manual” sound unavailable in other
states.

### 1.2 Shipped and global defaults

Shipped defaults for a newly created local Console session are:

- Automatic retrieval: **Never**.
- Assistant Library access: **Blocked**.

Canonical Settings may change the defaults used for future local sessions:

- Existing `[chat_defaults].rag_auto_retrieve_on_send`, shipped `false`, is
  relabeled and treated as the new-session automatic-retrieval default.
- New `[console].assistant_library_access_default`, shipped `false`, is the
  new-session assistant-access default.

The session captures both once when it is created. Later Settings changes do
not rewrite an open session or persisted conversation. The existing
`[console].direct_library_tools` setting is different: it is a live global
Direct-versus-RAG selector captured into each Allowed turn, not a policy
default and not an enable switch.

### 1.3 Existing and externally arrived conversations

- A conversation already present when the database upgrades is initialized
  once to its prior effective behavior: current global automatic retrieval
  and assistant access Allowed.
- A conversation inserted later by sync/import has no device-local authority.
  A missing row therefore means Never and Blocked, irrespective of global
  defaults.
- Opening a missing-row synced conversation does not insert a policy row until
  the user explicitly saves policy on this device. Safe defaults can remain a
  read-only effective value.

## 2. Ownership and components

The implementation adds narrow units instead of teaching the existing RAG
modal, conversation metadata helper, and tool registry to share ownership.

### 2.1 Policy model and repository

`Chat/console_library_policy.py` owns immutable values and validation:

- `ConsoleConversationLibraryPolicy`
- `ConsoleLibraryPolicySnapshot`
- `ConsoleLibraryPolicyDefaults`
- policy normalization and safe missing/error outcomes

`Chat/console_library_policy_repository.py` owns database reads, legacy
initialization, insert/update CAS, and deletion behavior. It has no Textual or
provider imports. Reads distinguish:

- row found and valid;
- row absent, effective safe defaults;
- legacy initialization still pending;
- unreadable/corrupt/error, effective safe defaults plus an explicit error.

The repository never returns a valid-looking Allowed policy from an exception.

### 2.2 Session holder

`ConsoleChatSession` receives a dedicated policy holder containing:

- effective values;
- persisted revision when one exists;
- whether the user explicitly staged a choice;
- legacy/error state;
- whether a durable save is pending.

A new local session captures global defaults into this holder. A policy edit
on an empty tab does not create a conversation, but it makes the session
non-untouched so navigation/close behavior cannot silently discard an explicit
choice. Merely holding untouched shipped defaults does not force persistence.

### 2.3 Turn snapshot

`ConsoleTurnExecutionContext` gains a detached Library section containing:

- automatic-retrieval policy;
- assistant-access policy;
- policy revision/source state;
- selected Direct/RAG mode;
- fixed automatic source categories;
- relevant item-scope snapshot;
- provider/model egress identity;
- activity attempt/run identifiers.

The snapshot is captured at actual execution. It is the only runtime input for
automatic admission and built-in Library-provider composition.

### 2.4 Activity projection

Assistant activity reuses `message_trajectory_metadata` for storage but not
`derive_trajectory` for presentation. A pure
`Chat/library_activity.py` projection filters and groups `library_activity`
events by durable turn, branch, attempt, and actor. It has no Textual or DB
dependency.

`derive_trajectory` must explicitly treat `library_activity` as sidecar-only:
it is neither a message-owned row nor an ordinary nested trajectory record.
This prevents the event from displacing an anchor's timing or appearing twice.

## 3. Device-local persistence

### 3.1 Table

The current ChaChaNotes schema is v44. Implementation is expected to add a
v44→v45 migration after re-checking the schema head immediately before coding.

```sql
CREATE TABLE console_conversation_library_policy (
    conversation_id TEXT PRIMARY KEY
        REFERENCES conversations(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    schema_version INTEGER NOT NULL DEFAULT 1
        CHECK(schema_version > 0),
    auto_retrieve_on_send INTEGER NOT NULL DEFAULT 0
        CHECK(auto_retrieve_on_send IN (0, 1)),
    assistant_library_access INTEGER NOT NULL DEFAULT 0
        CHECK(assistant_library_access IN (0, 1)),
    policy_revision INTEGER NOT NULL DEFAULT 1
        CHECK(policy_revision > 0),
    legacy_inherit_pending INTEGER NOT NULL DEFAULT 0
        CHECK(legacy_inherit_pending IN (0, 1)),
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

The table is local-only:

- no sync columns or triggers;
- no conversation-metadata mirror;
- no search/FTS/index projection;
- no Chatbook export/import mapping;
- cascade deletion with the local conversation row.

### 3.2 One-time legacy initialization

The migration itself must not read TOML or application config. Inside the
migration transaction it inserts one marked row for each conversation that
already exists:

- `assistant_library_access = 1`, matching the previously always-composed
  Library provider;
- `auto_retrieve_on_send = 0` as a placeholder;
- `legacy_inherit_pending = 1`.

Before the Console becomes interactive, an idempotent application initializer:

1. reads the current global automatic-retrieval value once;
2. updates only rows where `legacy_inherit_pending = 1`;
3. writes that automatic value, retains assistant Allowed, clears the marker,
   bumps the revision, and updates the timestamp in one transaction.

A crash before completion leaves marked rows for retry. A conversation synced
or imported after the migration transaction is not marked and can never be
captured by a later broad backfill. If initialization fails, Console opens in
an installed restricted state: manual Search Library remains usable, while
automatic retrieval and assistant access are Never/Blocked and policy UI
reports unavailable until Retry succeeds.

### 3.3 Reads and writes

Policy edits use optimistic compare-and-swap:

- update `WHERE conversation_id = ? AND policy_revision = ?`;
- create a previously absent local row with a conditional insert whose unique
  conflict is reported as a revision conflict, never silently overwritten;
- increment revision only on success;
- distinguish missing row, stale revision, corrupt row, and database error;
- never publish the candidate in session state before the durable write
  succeeds.

The policy modal remains open on failure. A stale revision offers Reload and
Compare/Retry; a missing conversation reports that it no longer exists. No
path displays Saved until the repository confirms the committed revision.

### 3.4 First persistence and temporary sessions

An empty durable-capable tab stages policy in memory. On first conversation
creation for a **new local session**, the holder's captured policy is inserted
in the same transaction as the conversation row even when the user did not
edit it; otherwise later resume would mistake the row for externally arrived
missing policy and lose the captured global defaults. Opening an already
persisted synced/imported conversation with a missing policy row remains
write-free until an explicit policy save. Failure rolls back first conversation
creation rather than creating a row that can briefly execute under
missing-policy semantics.

Ephemeral sessions retain policy and assistant activity in memory. Promotion
is one transaction covering:

- the conversation row;
- the policy row;
- promoted messages and active lineage;
- queued `library_activity` rows.

Any failure rolls the complete promotion back and restores the session's
ephemeral identity, policy holder, messages, activity, and retryability. It
must not leave a non-ephemeral session pointing at a partial durable bundle.

## 4. Runtime policy

### 4.1 Snapshot timing

The immutable snapshot is captured when a turn begins executing:

- an immediate send captures after send admission;
- a queued send captures after dequeue, so it sees changes made before it
  actually runs;
- a running turn never observes later policy, selector, provider, scope, or
  Settings changes;
- every subagent spawned by the turn inherits the same snapshot rather than
  resolving the conversation again.

The primary and subagents cannot independently widen Library authority. A
subagent's own narrower allow-list may still remove Library tools.

### 4.2 Four combinations

At execution, the snapshot drives two separate gates:

1. `auto_retrieve_on_send` controls only the pre-dispatch retrieval stage.
2. `assistant_library_access` controls only built-in Library-provider
   composition.

Neither gate infers from the other. Manual search is outside both gates.

### 4.3 Built-in provider composition

When assistant access is Blocked, `_library_provider_for_context` returns no
provider. No Library schema or callable is visible to the model, primary
agent, or subagents.

When Allowed:

- `direct_library_tools = true` composes `LibraryToolProvider`, with ADR-030's
  18 list/get/search tools over Media, Notes, Prompts, Skills, Conversations,
  and Collections.
- `direct_library_tools = false` composes `LibraryRagToolProvider`, exactly
  `search_library_rag`, over Notes, Media, and Conversations.

The selector is captured into the turn. Changing Settings during a run affects
the next executed turn.

### 4.4 Permanent name reservation

Create one canonical immutable reserved-name set containing the union of:

- all 18 direct `library_*` names;
- `search_library_rag`.

Skill and MCP collision filtering uses this set in all conversations and both
selector modes, even when no Library provider is registered. Registration
order or a dormant policy can never allow a third-party tool to claim a
built-in identity.

This reservation does not grant access and does not suppress unrelated MCP
Library tools with different names outside the existing ADR-030 overlap list.
MCP policy remains owned by its server/profile principal. Workspace/file tools
remain owned by `local:__local__` under ADR-032.

### 4.5 Provider egress disclosure

Allowed means Library results may enter the active model's request history.
The policy modal states the resolved provider/model destination and whether it
is local or cloud. If an Allowed conversation changes from local inference to
a cloud provider, Console immediately updates the expanded runtime line and
shows a persistent, non-blocking inline disclosure before the next send. No
new acknowledgment is required. The policy is preserved; the provider change
is not silently interpreted as consent revocation or renewal.

## 5. Manual Search Library

Manual search remains available from the composer/actions regardless of both
policy axes and regardless of assistant tool availability.

The manual surface owns:

- query;
- source-category toggles supported by the current Library Search/RAG seam;
- Search action;
- current item-scope summary;
- result/recovery state.

The query is always prefilled with the exact composer draft. Remove
`_console_draft_looks_like_rag_query`; no heuristic decides whether user text
looks like a query. The user may edit the search query without changing the
draft.

Source toggles are labeled **This search only**. They affect that manual run
and may be retained as harmless search-surface state, but they never become
automatic-retrieval policy. Manual results continue through the existing
staged-evidence bundle and are visible as `Sources — next send`.

## 6. Automatic pre-send retrieval

### 6.1 Admission

Automatic retrieval runs only when all are true:

- the executing turn snapshot says Automatic;
- the send is eligible plain user text under the current send contract;
- no explicit evidence bundle is already staged for this send;
- policy and RAG readiness are readable enough to execute;
- the send has not chosen the one-shot bypass.

Commands, approvals, regenerations, and other existing excluded send kinds
remain excluded. Explicit staged evidence wins because it expresses a more
specific user choice and avoids duplicate retrieval/cost.

### 6.2 Query and source categories

The executed draft is the query. The category set is fixed:

- Notes
- Media
- Conversations

Automatic retrieval never reads the manual search modal's source toggles.

The existing resolved conversation/workspace item scope still narrows exact
Note and Media eligibility. Under current scope semantics, an active item
scope excludes Conversations. The UI discloses this as runtime scope detail;
it does not mutate the fixed category policy.

### 6.3 Pre-dispatch lifecycle

The send enters a visible `Preparing Library context…` state before provider
dispatch. The user may Cancel, which stops the pending turn and keeps the
draft available.

Outcomes:

| Outcome | Behavior |
| --- | --- |
| Evidence found | Stage the bundle into the same exact request, disclose the count, then dispatch. |
| Zero matches | Dispatch without evidence and retain `0 matches · sent without Library evidence` on the sent turn. |
| Timeout or service failure | Pause before dispatch; show Retry, Send once without Library, and Cancel. |
| User chooses Retry | Re-run preparation against the same queued draft and immutable turn policy, with a new attempt ID. |
| User chooses Send once without Library | Dispatch this turn without evidence; retain bypass disclosure; do not change policy. |
| User chooses Cancel | Dispatch nothing; preserve the draft and policy. |

Retry does not silently refresh policy or provider selection. If the user wants
a later policy change to apply, Cancel and send again as a new turn.

## 7. Assistant Library activity

### 7.1 What is recorded

Each completed or provider-level refused built-in Library operation creates a
versioned, bounded payload in a `library_activity` sidecar row. A
conversation-policy Blocked state registers no provider and therefore creates
no synthetic activity event:

```json
{
  "version": 1,
  "attempt_id": "opaque",
  "run_id": "opaque",
  "actor": {
    "kind": "primary|subagent",
    "run_id": "opaque",
    "parent_run_id": "opaque-or-null"
  },
  "library_provider": "direct|rag",
  "operation": "library_search_notes",
  "status": "succeeded|empty|blocked|failed",
  "result_count": 3,
  "query_preview": "bounded text or null",
  "source_refs": [
    {"type": "note", "id": "opaque", "title": "bounded title"}
  ],
  "error_code": null,
  "error_summary": null
}
```

Bounds are service-owned and applied before persistence:

- fixed maximum query/error/title characters;
- fixed maximum reference count;
- opaque canonical IDs only;
- no bodies, snippets, excerpts, embeddings, binary data, filesystem paths,
  credentials, provider request bodies, or arbitrary exception strings.

The sidecar's top-level message/conversation/turn/seq fields provide durable
ownership and ordering. Model/provider fields describe the model turn; the
payload's `library_provider` describes Direct versus RAG.

### 7.2 Capture boundary

Capture occurs at the trusted built-in Library-provider result seam, before
provider-result truncation and before the result reaches the model. This is
the only point that sees authoritative operation identity and structured
source references without parsing lossy marker text.

The event first enters a thread-safe, turn-owned memory sink. If minimization
or sink admission fails, the Library result is withheld and the tool receives
a bounded failure; authorization cannot proceed without review capture.

Durable persistence may occur with the turn's other sidecar writes. A transient
database failure retries. Exhaustion does not retroactively fail an otherwise
completed model turn, but the Inspector shows a persistent **Library activity
not saved** warning with Retry. Logs contain only event IDs, sizes, statuses,
and error categories.

### 7.3 Anchoring, branches, and temporary work

- Anchor each event to the durable user/system turn opener, not an ephemeral
  tool marker or whichever assistant message happens to finish last.
- `turn_id`, `attempt_id`, `run_id`, actor kind, and parent-run identity
  distinguish retries and subagents.
- The selected-turn projection follows the active message lineage. Activity
  from another branch remains stored and appears only when that branch/turn is
  selected through the appropriate historical view.
- Ephemeral sessions keep activity in memory and persist it as part of atomic
  promotion.

### 7.4 Presentation and export

Activity never becomes:

- staged evidence;
- a Sources entry;
- next-send context;
- a system/user message;
- provider history;
- synchronized data.

The generic trajectory ledger does not render it as another tool row. The
Selected turn Inspector uses the separate projection and shows operation,
actor, mode, status, count, time, and bounded references.

ADR-067 exports raw sidecar rows in an explicit trajectory export. Its default
redaction path must recognize `library_activity` and replace query/source
details with bounded operation/status/count previews. Full activity details
are included only in the existing explicit full-export opt-in. Import treats
unknown/new fields as inert data and never executes source references.

## 8. Console interface

### 8.1 Fixed two-axis status chip

The status strip uses exactly one fixed-order policy chip:

```text
Library · Auto off · Agent blocked
Library · Auto on  · Agent blocked
Library · Auto off · Agent allowed
Library · Auto on  · Agent allowed
```

Spacing may normalize to the renderer, but noun and axis order never change.
Do not mix `Never`, `Automatic`, `Off`, `On`, `Blocked`, and `Allowed` between
chip variants.

Exceptional unreadable state is explicit:

```text
Library: blocked · policy unavailable
```

Runtime readiness, item scope, Direct/RAG selection, provider destination, and
staged source counts are not extra chip axes. They appear in expanded details
and their existing dedicated surfaces.

### 8.2 Library Access modal

Activating the policy chip opens a policy-only modal. It contains:

- status line: `This conversation · stored locally · not synced`;
- Automatic retrieval radio rows: Never / Automatic;
- Assistant Library access radio rows: Blocked / Allowed;
- resolved Direct/RAG explanation when Allowed;
- category disclosure: Direct covers all six Library categories; RAG covers
  Notes, Media, and Conversations;
- current model destination and local/cloud disclosure;
- explicit Save and Cancel;
- persistent Saving, Saved, Conflict, Unavailable, and Error feedback near
  the controls.

Use text-valued radio rows, not unlabeled switches. Save is disabled until the
draft differs from its loaded revision and while a save is in flight. A
conflict offers Reload and Compare/Retry. Read failure disables policy editing
until Retry; it does not fabricate Never/Blocked as if successfully loaded.

For an unpersisted session, the status line says
`Temporary session · applies until close or saved`. Save commits to the holder,
not the database, and the modal says so.

Escape, backdrop, and Cancel use ADR-031 safe dismissal. A clean modal closes.
A dirty modal asks whether to discard; it never saves or discards merely
because Escape/backdrop was pressed. Focus starts on the first radio group,
returns to the opening chip after close, and moves to the error/recovery status
when a save fails.

### 8.3 Search Library modal

The existing combined modal becomes a search-only surface:

- exact composer draft prefill;
- editable query;
- manual source-category toggles labeled `This search only`;
- current item-scope summary;
- Search Library and Cancel actions;
- retrieval status/recovery.

It contains no standing policy switch. Running a search never changes either
conversation policy axis.

### 8.4 Sources and Selected turn

The source tray uses one primary row per source:

```text
✓ Q3 planning notes                         note
✓ turbine-maintenance-log                  media
⚠ vendor contract                         note
```

Activation/expansion reveals snippet, authority, freshness, and source action.
Do not spend three or four visible rows per source in the ten-row tray.

Terminology is fixed:

- `Sources — next send` for staged evidence;
- `Cited sources (N)` for the selected sent answer;
- `Library activity (N actions)` for assistant operations.

The conversation Inspector gains a **Selected turn** group containing Cited
sources and Library activity. A message-level activity affordance selects that
turn and focuses the activity subsection. The empty state is explicit:
`No Library activity for this turn.` Activity is not added as another
top-level rail peer.

### 8.5 Responsive and literal rendering

Replace the modal's fixed width with a viewport-fitting bound and a bounded
scroll body. Keep actions visible/pinned when content overflows. At narrow
widths, radio rows, disclosures, and actions stack without horizontal
clipping. Tests mount the production screen hierarchy and stylesheet bundle
and assert painted containment, not only declared dimensions.

All dynamic query, title, error, source, activity, and recovery text renders
with `markup=False` or the equivalent literal-text boundary. The adjacent
`console_staged_context.py` recovery sink is included when this flow touches
it so a Library title such as `[red]` cannot become formatting.

## 9. Failure behavior

| Failure | User-visible behavior | Runtime behavior |
| --- | --- | --- |
| Legacy initializer fails | Policy unavailable with Retry | Never/Blocked; no built-in provider |
| Policy read fails/corrupt row | `Library: blocked · policy unavailable` | Never/Blocked for the attempted turn |
| Policy save fails | Modal stays open; Error + Retry | Prior committed revision remains active |
| Policy revision conflict | Conflict + Reload/Compare/Retry | No candidate publication |
| Conversation deleted during save | “Conversation no longer exists” | No row recreation |
| Auto retrieval timeout/failure | Pause with Retry / Send once without / Cancel | No provider dispatch until user decides |
| Auto retrieval zero matches | Persistent 0-match disclosure on turn | Dispatch without evidence |
| User cancels preparation | Draft preserved | No provider dispatch |
| Library provider blocked | Tool absent | No schema/call/result |
| Activity memory capture fails | Bounded tool failure | Result withheld from model |
| Activity durable save exhausts retries | Inspector “not saved” warning + Retry | Completed turn remains completed |
| Provider changes local→cloud while Allowed | New destination re-disclosed | Policy preserved; later turn uses new snapshot |

## 10. Security and privacy invariants

- Manual search authority never grants automatic or assistant authority.
- Missing, pending, corrupt, or unreadable policy never grants access.
- The trusted built-in namespace is reserved independently of catalog
  availability.
- Primary and subagents share one immutable maximum authority; a child cannot
  re-read globals to widen it.
- Assistant access governs only built-in Library providers. MCP and local
  tools remain under their independent permission principals and disclosure.
- Conversation policy is device-local and excluded from sync, import/export,
  metadata mirrors, FTS, diagnostics, and provider payloads.
- Activity is local-only, minimized before storage, literal-rendered, and
  default-redacted on explicit trajectory export.
- Automatic retrieval uses a fixed category set and does not retain a manual
  query/filter as standing hidden policy.
- “Send once without Library” is request-scoped, visible, and non-persistent.
- Persistent logs never contain Library queries, titles, IDs, source bodies,
  excerpts, or tool results.

## 11. Verification strategy

The repository requires targeted verification unless the owner separately
opts into a full sweep.

### 11.1 Migration and repository

- Historical v44 fixture proves the table is absent before migration and
  structurally complete afterward.
- Migration marks only preexisting conversation IDs.
- A conversation inserted after migration is never legacy-initialized.
- Initialization snapshots global auto exactly once and is crash/retry
  idempotent.
- Existing conversations become prior-auto + Allowed; new/missing rows resolve
  Never + Blocked.
- Corrupt rows/read failures are explicit and fail closed.
- CAS success, stale revision, missing conversation, and concurrent writers.
- First conversation creation and policy insert are atomic.
- Ephemeral promotion success and injected failures at every write boundary
  prove complete rollback/restoration.
- Conversation deletion cascades policy/activity rows as defined.

### 11.2 Runtime and catalog

- All four policy combinations.
- Allowed Direct exposes exactly the 18 ADR-030 tools.
- Allowed RAG exposes exactly `search_library_rag`.
- Blocked exposes neither provider.
- Every 19-name collision is rejected in Blocked, Direct, and RAG modes for
  Skills and MCP.
- Policy/selector/provider changes during a turn do not affect that turn.
- Queued turns capture at execution.
- Primary and subagents share the snapshot; child narrowing still works.
- Read/init failure never constructs a Library provider.

Authorization/name-reservation tests are mutation-checked: removing the gate
or static reservation must turn the test red.

### 11.3 Automatic retrieval

- Eligible plain sends, excluded send kinds, and explicit staged-evidence skip.
- Exact draft query and fixed Notes/Media/Conversations categories.
- Item scope narrows Notes/Media and excludes Conversations when active.
- Success stages the bundle into the exact dispatched request.
- Cancel prevents dispatch and preserves the draft.
- Timeout/failure pauses; Retry, one-shot bypass, and Cancel have distinct
  outcomes and do not change policy.
- Zero matches dispatches without evidence and leaves persistent disclosure.
- No path silently falls through from failure to provider dispatch.

### 11.4 Activity

- Direct and RAG success/empty/blocked/failure events.
- Capture occurs before model delivery and before result truncation.
- Query/title/reference/error bounds and forbidden-content scans.
- Primary/subagent/run/attempt attribution under concurrency.
- Durable turn-opener anchoring, active-branch selection, and ephemeral
  promotion.
- Capture failure withholds result; durable failure retries and shows warning.
- `derive_trajectory` timing/message ownership is unchanged with activity rows
  present; separate projection returns them once.
- No activity in staged context, prompt construction, provider history, sync,
  ordinary logs, or Sources.
- Default trajectory export redacts activity details; full opt-in retains the
  bounded event.

### 11.5 Interface and live verification

- Exact chip grammar for four states plus unavailable state.
- Policy modal Save/Cancel, dirty Escape/backdrop, conflict, error, temp-state,
  provider disclosure, disabled reasons, and focus restoration.
- Search modal exact draft prefill, no heuristic, and “This search only”
  filters.
- One painted row per source with expandable details.
- Selected-turn Cited sources/Library activity grouping and focus handoff.
- Literal dynamic text, including Rich/Textual markup-shaped titles/errors.
- Narrow and standard viewports under the production hierarchy and complete
  stylesheet bundle; action containment and rendered-frame assertions.
- Scratch-profile real Console walkthrough covering policy persistence,
  automatic success/zero/failure recovery, Blocked/Allowed tool catalog,
  Direct/RAG selection, activity review, provider change disclosure, restart,
  and deletion.

## 12. Rollout and rollback

Rollout is fail-closed. The startup initializer completes or exposes an
unavailable state before Console can advertise Library authority. No recurring
backfill exists after the one-time marked set is cleared.

Rollback disables policy editing, automatic retrieval, and built-in Library
composition while retaining the local table and sidecar rows. Manual Search
Library remains available. Older application versions must not be asked to
down-migrate or reinterpret the v45 database; use the normal migration backup
and downgrade procedure.

## ADR check

**ADR required:** yes

**ADR path:** `backlog/decisions/079-console-library-conversation-authority.md`

**Reason:** the design changes local schema/migration, sync and ownership,
assistant permission/runtime composition, privacy retention, and long-lived
Console control/review structure.
