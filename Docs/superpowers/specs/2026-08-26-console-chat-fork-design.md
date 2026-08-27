# Console chat fork from a message

**Date:** 2026-08-26

**Status:** Owner-approved design; implementation planning not started

**Task:** Not yet assigned

**ADR:** [ADR-092](../../../backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md)

**Baseline:** `de580f20ba9c3a521d4c336668c2c7946d67c614`

## Summary

Add a visible **Fork** action to each eligible selected Console message, immediately
before **Regenerate**. The action opens a small confirmation dialog with an editable
title defaulted to `Forked from <original title>`. Confirming creates a new Console
chat whose history is the source chat's active lineage through that message,
inclusive, opens the fork as the active Console tab, and leaves the source chat open
and unchanged.

Forking is a projection into a new chat, not a sibling variant, a second live view of
the same conversation, or a generic deep copy of runtime state. The projection
captures the visible message variants and durable user-visible sidecars, assigns new
identities, preserves durable lineage where one exists, and recreates local authority
for the new live session. It never copies scratch files, approvals, tool grants, live
runs, provider continuation, or recovery state.

Durable sources produce durable forks. Temporary sources produce temporary forks. A
temporary fork that is saved later becomes an independent durable root; saving it
does not persist or mutate its temporary source.

## ADR check

**ADR required:** yes

**ADR path:**
`backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md`

**Reason:** the feature establishes a long-lived cross-module copy contract for
conversation storage, message identity, variants and sidecars, temporary-to-durable
ownership, and security-sensitive Console authority.

No schema migration is expected. The current conversation schema already has
`root_id`, `parent_conversation_id`, and `forked_from_message_id`; the persistence seam
must expose and populate them correctly.

## User promise

When the user forks at a message:

- the new chat contains exactly the visible conversation path through that message;
- messages after the boundary and branches not on that path are absent;
- the source chat's active leaf, selected variants, persistence, title, workspace, and
  live state do not change;
- the fork has an independent identity and may diverge immediately;
- included attachments, citations, generated images, visible thinking, and other
  supported message content remain honest after reload;
- ephemeral video bytes and local filesystem authority are not implied to have been
  copied; and
- failure never leaves a partially presented fork or falsely claims that a committed
  fork was rolled back.

## Goals

- Make branching from an earlier message a fast, understandable per-message action.
- Preserve the exact active lineage and visible variant boundary the user selected.
- Support user and assistant targets, durable and temporary chats, text and image
  variants, sent attachments, citations, generated media, and stopped partial replies.
- Give the fork fresh conversation, message, turn, variant, and sidecar identities.
- Preserve durable ancestry without sharing mutable message ownership.
- Copy declarative chat configuration while recreating runtime and security authority.
- Commit all durable fork data atomically before publishing the new live tab.
- Keep the original chat byte-for-byte and behaviorally unchanged by the fork action.

## Non-goals

- Forking an agent run, its activity ledger, tool calls, approvals, queue, or recovery
  checkpoint.
- Copying private provider continuation or making a copied assistant answer resumable.
- Copying scratch files, external folder access, permission decisions, or resolved
  project-instruction bodies.
- Persisting a temporary source as a side effect of forking or later saving its fork.
- Automatically generating a response when the boundary is a user message.
- Copying later turns, off-path sibling branches, or unselected variants.
- Preserving cost/usage accounting as though copied messages were new provider calls.
- Adding a general conversation-template, merge, or cross-workspace import system.
- Changing the existing workspace-membership action currently named
  `fork_conversation_into_workspace`; that operation is a link, not this clone.

## Terminology

- **Source session:** the live Console tab on which the action was opened.
- **Source conversation:** the durable conversation behind the source session, when
  one exists.
- **Boundary message:** the selected stable USER or ASSISTANT tree node, included as
  the last message of the fork.
- **Active lineage:** the unique parent chain from the source tree root to the
  boundary. It is derived from canonical tree maps, not the rendered transcript.
- **Visible variant:** the message generation or sibling currently presented in the
  source without changing which variant the source persists.
- **Fork snapshot:** the immutable, fully validated projection from which the new
  session and any durable rows are built.
- **Declarative configuration:** values describing how a future turn should be
  composed, such as provider/model selection and context policy.
- **Authority:** live permission or resource ownership, such as scratch leases,
  approvals, file bindings after validation, and recovery ownership.

## User experience

### Action placement and eligibility

The selected-message action row adds a text-labelled `Fork` action immediately before
`Regenerate`. Its tooltip and dialog title use the fuller label **Fork chat**. The
transcript adds the conflict-free `f` selected-message binding and the action guide
adds `f Fork`; the screen's discoverability copy must list it with `c`, `e`, and `r`.

The action appears for real USER and ASSISTANT tree nodes. It does not appear on
render-derived system/status rows, TOOL/activity markers, original-attempt previews,
or other display-only transcript rows. A real node is enabled when its content is a
stable fork boundary:

- complete USER messages are eligible;
- complete ASSISTANT messages are eligible;
- stopped ASSISTANT messages with non-empty visible content are eligible and the
  dialog labels the boundary as a partial response;
- pending or streaming targets are disabled until they settle;
- failed or cancelled targets without meaningful visible content are disabled; and
- deleted, stale, or no-longer-on-the-captured-path targets are disabled.

If a row is temporarily ineligible, selecting the row shows the reason in visible
action-help text and pressing `f` repeats that reason. The design does not assume that
a disabled Textual button itself can receive focus. Color and hover-only tooltip text
are not the sole explanation. A run occurring after an otherwise stable earlier
boundary does not disable that earlier boundary.

### Confirmation dialog

Opening the action captures an initial source fence and shows a compact modal:

```text
Fork chat

Copy this chat through:
Assistant response · 8 messages · visible variant 2 of 3

Includes sent attachments and citations.
Runs, tool history, drafts, and scratch files are not copied.
Starts with fresh scratch and permissions; temporary video files are not copied.

Name
[Forked from Research notes                              ]

                                      [Cancel] [Fork chat]
```

For a user boundary, the summary says `User message` and makes clear that no reply is
generated automatically. For a stopped response, it says `Partial assistant response`.
The message count is the number of USER/ASSISTANT nodes in the projected lineage.
Variant detail is shown only when it adds information. Attachment/citation and video
facts are conditional: the modal does not claim that content is present when the path
has none, and it shows the video warning whenever the copied path contains a video.

The title field is focused with its contents selected so typing immediately replaces
the default. Enter confirms and Escape cancels. The default is `Forked from
<source display title>`; a missing title uses `Untitled chat`. The resulting value
passes the canonical conversation-title normalization: trim surrounding whitespace,
collapse line breaks/control characters into safe single-line text, reject blank
input, and enforce the existing title length bound. The prefixed default truncates at
that same bound without cutting malformed text.

The confirm button changes to `Forking…` and becomes disabled while one submission is
in flight. Cancel is also disabled after commit begins. Repeated Enter, clicking twice,
or an event replay cannot create a second fork for the same submission token.

### Success and failure

On success, the new fork is registered as a Console session, appears in the same Chats
or Workspace rail location, and becomes the active Console tab. The original tab
stays open and retains its exact state.

Failure before durable commit leaves the modal open, preserves the user's title, and
shows a bounded actionable error with Retry. No fork appears. A validation conflict
explains that the source or visible variant changed and asks the user to close and
reopen the dialog rather than silently copying a different boundary.

Durable commit and tab activation cannot be one transaction. If commit succeeds but
session publication or activation fails, the error says that the fork was created and
names it, then offers `Open fork` or directs the user to the Chats/Workspace rail. It
must not say the operation failed or create another conversation on Retry.

For a temporary fork, store registration is its publication boundary. If activation
fails after registration, the registered tab remains discoverable and can be opened;
if registration itself fails, no temporary fork remains.

## Canonical fork snapshot

### Open fence

Opening the dialog captures only a lightweight fence, not a mutable object graph:

- source session ID and session generation/revision;
- source conversation ID and version when durable;
- boundary native message ID and persisted message ID when available;
- boundary message content/version token;
- active-lineage identity through the boundary;
- selected visible variant ID/index and generation-envelope version; and
- source durability mode and displayed title.

The title remains user-editable independently of this fence.

### Confirm-time validation

Confirming re-resolves the source by the captured session ID, never by whichever tab is
currently active. Under the store's mutation serialization it verifies that the
session, boundary, path, message version, and visible variant still match the fence.
It then creates one immutable `ConsoleChatForkSnapshot` containing normalized copy
records and configuration values. Persistence and live publication consume only this
snapshot; they do not reread mutable source state midway through the operation.

The lineage comes from the store's native `nodes` and `parents` ownership maps.
`messages_for_session()` is not a valid source because it includes spliced TOOL and
activity presentation markers. Variant resolution reads the current visible generation
without calling `create_sibling`, `keep`, variant persistence, or any method that
changes the source active leaf.

Snapshot construction is fail-closed. If any in-scope payload cannot be cloned
honestly—for example, a sent attachment is corrupt or an image payload required by the
visible variant is unavailable—the operation stops before publication and identifies
the unsupported item. It never creates a fork with silently missing content.

### Identity remapping

Every fork-owned mutable record receives a fresh ID. The snapshot contains explicit
maps for:

- native message IDs;
- persisted message IDs, for durable forks;
- turn IDs, preserving grouping among records that shared a source turn;
- generation/variant IDs;
- attachment and generation-metadata sidecar IDs; and
- citation or other copied message-owned sidecar IDs.

Parent links are rewritten in their proper namespace. No native parent is stored as a
persisted parent and no source message ID becomes the fork's mutable owner. Cross-
references to another copied message are remapped. References to immutable external
records, such as a Library source identity, may be retained. Every other relationship
is cleared with truthful degraded metadata or causes precommit validation failure if
it is necessary to render the copied content. The fork contains no dangling source-
message reference.

Fresh IDs need not be artificially different across the new native and persisted
namespaces if an existing canonical creation path intentionally uses one UUID for both,
but no ID from the source is reused as fork ownership.

## Copy contract

The projection is an allowlist. New Console message sidecars are excluded until their
owner and remapping behavior are deliberately added to this contract and tested.

### Included content

| Source data | Fork behavior |
| --- | --- |
| USER/ASSISTANT active lineage | Copy root through the selected boundary, inclusive, with fresh ownership and remapped parents. |
| Visible text variant | Copy the currently displayed answer/content directly, including a session-only selected sibling, without keeping or changing it in the source. |
| Visible generated-image variant | Copy the selected renderable image payload and its bounded generation provenance with fresh attachment/sidecar ownership. Do not copy unselected images. |
| Stopped partial assistant text | Copy the non-empty visible partial with terminal `stopped` semantics; do not make it resumable. |
| Sent attachments | Copy each attached item's ordered logical association. Use fresh sidecar IDs; an immutable content-addressed blob may be shared only if its store contract makes sharing ownership-safe. Never retain a source filesystem path as authority. |
| Citations and source notices | Copy the stable, user-visible citation/source projection from durable rows or current settled message state and remap message-owned IDs. Stable external Library/source IDs may remain references; copied snippets remain bounded by their existing storage contract. |
| Displayable thinking | Copy the supported visible-thinking envelope belonging to the selected assistant generation, with fresh generation ownership. Preserve proprietary text-free evidence as text-free. |
| Generation provenance | Copy the selected variant's bounded provider/model/request provenance needed to explain or regenerate the visible content. |
| Role identity | Copy character/persona identity, persona memory mode, human display override, and system prompt as declarative configuration. |
| Future-turn model configuration | Copy selected provider, model, compatible generation parameters, agent/runtime mode, source/RAG selector, context-window/compaction policy values, and speech preferences. |
| Conversation scope | Preserve Default Chats versus the same named Workspace ID and the same durable item/source scope configuration. Workspace membership is projected after durable commit through the existing idempotent seam. |
| Library policy | Seed a new conversation policy owner with the source's captured effective `auto_retrieve_on_send` and `assistant_library_access` values. Start its own revision history. |
| Project-instruction selection | Copy only declarative selected binding/configuration and stable locator fingerprint inputs. The fork performs a fresh validation and preflight before future use. |

### Excluded or recreated state

| Source data | Fork behavior |
| --- | --- |
| Messages after the boundary | Exclude. |
| Off-path branches and unselected variants | Exclude. The fork begins with one selected path, not a hidden copy of the source tree. |
| Rendered system/status/TOOL/activity rows | Exclude; they are presentation or run history, not conversation tree ownership. |
| Draft, prefill, staged files/evidence | Exclude composer text, one-shot prefill, unsent/staged attachments, staged evidence, and pending paste/drop state. The fork composer starts empty. |
| Runs and queues | Exclude active/pending runs, prompt queue entries, wakeups, fleet/subagent state, agent todos, progress, run logs, and cancellation state. |
| Tool and review state | Exclude tool calls/results as activity records, change-review state, feedback votes, annotations, and original-attempt previews. |
| Provider continuation | Exclude `provider_continuation_json`, mandatory tool-resume state, response cursors, and provider-side continuation tokens. |
| Dispatch recovery | Exclude turn preparations, dispatch checkpoints, recovery owners, retry state, and optimistic echoes not already committed as stable messages. |
| Usage/cost accounting | Exclude original request usage, token/cost ledger entries, and billing aggregates. Copied messages did not incur new provider calls. |
| Derived context | Exclude compaction summaries, memory records, retrieval results, caches, serialized provider history, and prebuilt prompt bodies. Recompute from fresh IDs and copied policy. |
| Privacy/capture overrides | Exclude next-send and per-conversation exchange-capture overrides. The new conversation resolves the canonical current global/default capture policy. |
| Local authority | Allocate fresh private scratch lazily. Do not copy scratch locators/files/leases, approvals, tool grants, selected permission decisions, or resolved project-instruction bodies. |
| Transient UI | Exclude scroll/focus/selection state, expansion state, toasts, modal state, cached render rows, undo stacks, and speech playback. |

### Generated video

ADR-044 stores generated-video bytes in a message-ID-keyed ephemeral store. A fork
therefore never copies or aliases those bytes. It copies the visible human-readable
video marker and bounded regeneration metadata into the new message, remapping any
in-snapshot image-to-video source reference and clearing every path, URL, store key,
or source message owner. The fork renders the standard named tombstone with a
regenerate action even when the source video still plays. The dialog discloses this
before confirmation.

This behavior is not a partial-copy error: video ephemerality is the modality's
canonical durable contract. A later explicit `Save a copy…` remains the only way to
export video bytes.

### Selected generation envelope

The fork copies the user-displayable parts of the selected generation atomically:
visible answer, selected image attachment if any, displayable thinking/proprietary
evidence, and bounded generation provenance. It deliberately strips private
continuation and original usage accounting. If the existing envelope cannot express
that projection without violating its version contract, the implementation must add a
versioned fork projector; it must not shallow-copy or mutate the envelope in place.

## Configuration and authority

Forking preserves intent but grants no authority.

### Workspace and filesystem

A fork retains the source conversation's Default Chats or named Workspace association.
It does not clone a Workspace. The new live session receives a fresh scratch owner
whose unpredictably named private directory is allocated lazily under ADR-082. Two
simultaneous live fork/source tabs never share scratch, even when they belong to the
same durable Workspace.

Explicit Workspace folder bindings remain properties of that Workspace, but the fork
must re-resolve their current existence, read/write mode, and fingerprint before local
tools see them. It never inherits a scratch snapshot, lease, selected filesystem root,
or successful preflight object from the source.

### Project instructions

The fork may retain the declarative selected project binding and fingerprint inputs.
ADR-069 still requires fresh startup/lazy resolution, current fingerprint validation,
and normal preflight before any instruction body influences a provider request.
Resolved instruction bodies are not copied into session metadata, history, logs, or the
fork snapshot. Instructions remain untrusted context and never become tool permission.

### Library, RAG, and memory

The fork captures the source's effective declarative Library policy values and writes
them into a fresh policy owner/revision when the fork becomes durable. It copies RAG
selector and item-scope configuration, but not retrieved chunks, evidence staging,
memory summaries, compaction records, or cache entries. Those derived records are
branch-valid against message identity under ADR-052 and must be recomputed.

### Permissions, capture, and one-shots

Permission decisions, MCP approvals, local-tool grants, attachment upload approvals,
and one-shot settings are never copied. The fork applies the current canonical global,
Workspace, provider, and per-principal rules when a future action is attempted.

Exchange-capture policy is privacy governance rather than conversational context.
Neither a next-send override nor a per-conversation Safe/Full override is copied. The
fork resolves the current global/default capture detail and exposes that effective
state through the existing Inspector before the next send.

## Durable and temporary ownership

### Durable source

A durable fork is one SQLite transaction containing the new conversation, every copied
message in lineage order, all included message-owned sidecars, the fork's active leaf,
and required per-conversation policy rows. It uses:

- `new_conversation.id`: a preallocated fresh UUID;
- `new_conversation.root_id`: the source conversation's canonical `root_id`;
- `new_conversation.parent_conversation_id`: the source conversation ID; and
- `new_conversation.forked_from_message_id`: the original persisted boundary message
  ID.

All copied messages have fresh persisted IDs. Their parent links refer only to other
new messages. The active leaf is the copied boundary. The fork's first later message
continues from that new leaf.

The source conversation and messages are read and version-checked but never updated.
The operation must not call `set_conversation_active_leaf` for the source, persist its
visible variant, rename it, or modify its workspace membership.

Workspace registry membership is a separate idempotent postcommit projection, matching
the existing Console persistence boundary. The durable `workspace_id` remains
authoritative if registry projection is temporarily degraded, and restore/reconcile
must use the committed fork ID rather than create another conversation.

### Temporary source

An explicitly temporary source creates an explicitly temporary fork. The fork exists
only in the live Console store, receives fresh session/message/turn/variant identities,
starts with fresh scratch and no run state, and records at most process-local display
provenance back to the source session/boundary. It writes no durable conversation,
message, policy, or ancestry row.

If the temporary fork is later promoted/saved, it becomes an independent durable root:

- `root_id` is its new conversation ID;
- `parent_conversation_id` is null; and
- `forked_from_message_id` is null.

The temporary source remains temporary and unchanged. Process-local provenance is not
converted into a foreign-key claim.

An ordinary persistable session that has not yet acquired durable IDs follows the same
ancestry rule: its fork keeps the source's durability mode, but when either session is
first persisted it is an independent root because no durable source identity existed
at fork time.

## Commit, publication, and idempotency

### Durable operation

The flow is:

1. Validate the dialog fence and build the immutable snapshot.
2. Preallocate every fork-owned ID and bind that immutable write set to the modal's
   submission token.
3. Write the entire durable projection in one ChaChaNotes transaction.
4. Treat a retry whose preallocated conversation/message/sidecar IDs already contain
   the matching canonical write set as the same completed operation; any partial or
   mismatched collision is an error.
5. Project Workspace membership idempotently after commit.
6. Hydrate/register the new live session from the committed projection.
7. Activate the new tab.

The preallocated ID set is the durable idempotency key for an ambiguous postcommit
result. The UI retains it and the immutable expected records until the operation is
resolved; no new metadata field is needed. A retry queries those exact IDs and never
generates a second set after persistence might have committed.

No live fork is published before the database transaction succeeds. Transaction
failure rolls back the new conversation, messages, sidecars, active leaf, and policy
rows together. A postcommit Workspace link, hydration, mount, or activation failure is
a degraded success, not a rollback.

### Temporary operation

The snapshot is converted to a complete detached session before registration. Store
registration publishes it under one new session ID. Activation follows. Replaying the
same modal submission uses the same session ID and resolves to the already registered
fork rather than appending another one.

## Component ownership

Implementation should preserve the Console module ownership ratchet:

- `Chat/console_chat_fork.py` owns frozen snapshot/copy records, eligibility, copy
  allowlists, identity remapping, and pure projection helpers.
- `ConsoleChatStore` supplies serialized canonical tree/variant reads, source-fence
  validation, and detached-session registration. It does not persist by deep-copying
  its internal dictionaries.
- `ChatPersistenceService` owns the atomic durable write and extends the
  conversation-creation seam to accept existing lineage columns. The existing
  `fork_conversation_into_workspace` method remains a differently named membership
  link and must not be reused as the clone implementation.
- `UI/Console_Modules/session.py` owns dialog orchestration, source-session targeting,
  in-flight state, publication, activation, and partial-success recovery.
- `UI/Console_Modules/message.py` dispatches the selected-message request through a
  named session-controller callable wired in `UI/Console_Modules/wiring.py`; it does
  not absorb persistence or tab logic.
- `Chat/console_message_actions.py` owns action ordering and eligibility presentation.
- `Widgets/Console/console_transcript.py` renders the action, tooltip, key binding, and
  action guide.
- A focused Console modal widget owns layout, title input, validation presentation,
  focus, and the confirm/cancel result contract.
- `UI/Screens/chat_screen.py` receives wiring/delegation changes only. New fork
  business logic must not be added to the screen monolith.

Names may follow current neighboring conventions during implementation, but these
ownership boundaries are normative.

## Error and recovery matrix

| Failure point | Required result |
| --- | --- |
| Source session/boundary vanished before confirm | Keep dialog open; explain that the source changed; no mutation. |
| Active path/message/variant version changed | Fail the captured operation; require reopening for a new exact preview. |
| Title invalid | Inline validation; no snapshot persistence. |
| Attachment/image/citation projection unsupported or corrupt | Fail closed before commit and name the affected content class; never omit silently. |
| Project binding no longer validates | The fork may still be created with declarative association, but future use is unavailable until normal preflight succeeds; no stale body or permission is copied. |
| SQLite transaction fails | Roll back the complete fork; preserve dialog/title; Retry uses the same safe submission identity only after confirming no commit. |
| Transaction result is ambiguous | Resolve the preallocated conversation/message/sidecar ID set before Retry; reuse the complete matching fork, retry only when none exist, and quarantine any partial or mismatched collision. |
| Workspace registry projection fails after commit | Report created-with-degraded-membership; keep durable `workspace_id`; reconcile idempotently. |
| Live hydration/registration fails after commit | Report that the named fork exists; offer rail/open recovery; do not duplicate it. |
| Tab activation fails after registration | Leave the registered fork discoverable and offer `Open fork`. |
| Source closes after commit | Fork remains valid and independent. |

Diagnostics may include opaque source/fork IDs and content classes, but not message
bodies, project-instruction bodies, attachment bytes, secrets, or local paths.

## Security and privacy invariants

- Forking does not widen Workspace, filesystem, Library, MCP, tool, provider, or upload
  authority.
- Every local filesystem capability is re-resolved for the new session; scratch is
  always fresh.
- Project-instruction bodies are fetched and preflighted afresh and never placed in
  metadata or logs by the fork operation.
- Provider credentials, continuation tokens, endpoint secrets, request captures, and
  permission decisions are excluded by construction.
- Titles are plain, bounded, single-line text and are escaped by existing presentation
  surfaces.
- Attachment and generated-image copying uses canonical size/type validation and does
  not follow arbitrary source paths.
- Ephemeral video paths and bytes are neither copied nor aliased.
- The snapshot allowlist rejects unknown sidecar types rather than serializing arbitrary
  object state.
- The source is read under version fences and receives no writes.

## Testing and verification

Implementation uses targeted tests only unless the owner separately requests a full
suite.

### Pure/domain tests

- Active lineage through USER and ASSISTANT boundaries is exact and inclusive.
- Off-path branches, later messages, display-only TOOL markers, and unselected variants
  are absent.
- Visible session-only variants copy without changing source selection or active leaf.
- New message/turn/variant/sidecar IDs and parent remaps contain no source ownership.
- Stable external source references remain valid; internal references remap; dangling
  message references fail.
- Complete, stopped-partial, streaming, pending, failed-empty, deleted, and stale
  targets have the specified eligibility/reason behavior.
- Unknown sidecar types fail closed.
- Source objects and persistence rows are unchanged after success and every failure.

### Persistence tests with real in-memory SQLite

- Durable fork conversation ancestry uses existing root/parent/boundary columns.
- Messages, attachments, citations, selected generation data, active leaf, and policy
  rows commit atomically with fresh IDs.
- Injected failure at every write stage leaves no partial fork.
- Ambiguous retry with the preallocated ID set returns the one complete matching fork
  and rejects a partial or mismatched collision.
- The fork reloads with the same visible path and can accept a divergent next turn.
- Workspace projection failure preserves the committed fork and reconciles without a
  duplicate.
- Derived memory, capture overrides, usage, continuation, checkpoints, and feedback do
  not appear in copied rows.

### Variant and media tests

- Text sibling selection, generated-image selection, attachments, citations, and
  displayable thinking copy only the visible supported envelope.
- Generated video always reopens as a named tombstone with regeneration metadata and
  no copied source path/store key/bytes.
- In-snapshot media references remap; missing required image/attachment payloads fail
  before commit.
- Proprietary thinking evidence remains content-free; private continuation is absent.

### Authority and temporary tests

- Source and fork receive different scratch roots and lease generations.
- Approvals/tool grants do not appear in the fork.
- Same Workspace association is retained while folder/project instruction authority is
  freshly validated.
- Library policy values are seeded into a fresh owner/revision; derived retrieval and
  compaction state is absent.
- Temporary-to-temporary fork writes no DB rows, diverges independently, and later
  promotion creates a durable root with null parent/fork columns.
- Saving a temporary fork does not persist its temporary source.

### UI tests

- `Fork` renders immediately before `Regenerate` for eligible selected messages.
- `f`, button click, tooltip, guide, screen help, action IDs, and controller dispatch
  remain synchronized.
- Modal title selection, typing, Enter, Escape, blank/overlong validation, focus order,
  double-submit fencing, and narrow-terminal layout work in Textual Pilot.
- Modal summary distinguishes USER, ASSISTANT, partial, variants, attachments, and
  ephemeral-video behavior.
- Precommit errors preserve the title; postcommit activation errors identify the
  already-created fork and recover without duplication.
- Success opens the fork as active while the original tab remains open and unchanged.

Run production-shaped Pilot checks with the consolidated Console stylesheet at
`120x35`, `100x30`, and `80x24`. Add one live local TUI journey that creates a short
chat, forks from a middle USER and ASSISTANT message, renames one fork, switches among
all tabs, restarts for durable reload, and confirms the source did not change. It need
not contact a provider; deterministic seeded conversation data is stronger evidence
for the clone boundary.

## Documentation and rollout

- Update the Console message-action guide and user guide with the `Fork` action, `f`
  shortcut, copy boundary, temporary behavior, and video caveat.
- Document that a durable fork appears in the same Chats/Workspace section and that a
  temporary fork remains temporary.
- Keep persisted conversation formats backward compatible; old rows require no
  backfill.
- Add no new core dependency.
- Treat the projection allowlist as versioned application behavior. A future sidecar
  must opt in deliberately rather than becoming copyable by accident.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Reuse `create_sibling` or Regenerate | It creates a message variant inside and mutates the source conversation; the requested object is a new independent chat. |
| Open a second tab for the same conversation ID | Both tabs would share durable message ownership and active-leaf changes, so the original would not remain untouched. |
| Generic `deepcopy` of `ConsoleChatSession` | Copies transient caches, run state, authority, stale object references, and mixed ID namespaces without a durable contract. |
| Copy rendered `messages_for_session()` output | That view contains spliced TOOL/activity rows and may omit canonical tree relationships. |
| Persist/Keep the visible variant in the source first | Violates the source-untouched promise and makes forking depend on a separate mutation. |
| Copy only plain text | Loses attachments, citations, images, visible thinking, configuration, and the user's selected variant. |
| Copy the complete hidden branch tree | Surprises users, retains unselected alternatives, and makes the chosen message an unclear boundary. |
| Copy scratch, approvals, continuation, or checkpoints | Transfers live authority/recovery ownership and can leak files or resume an operation under a different chat identity. |
| Copy ephemeral video bytes | Contradicts ADR-044's message-keyed ephemeral storage and explicit save-only escape hatch. |
| Auto-persist a temporary source before forking | Mutates the original and changes its privacy/lifecycle without consent. |
| Best-effort partial fork | Creates an apparently complete chat with silently missing user data. The safe full projection must fail before commit instead. |

## Related decisions

- [ADR-033: Application session state ownership](../../../backlog/decisions/033-application-session-state-ownership.md)
- [ADR-044: Ephemeral generated-video storage](../../../backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md)
- [ADR-052: Conversation memory and compaction](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md)
- [ADR-069: Project-instruction local state and preflight](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md)
- [ADR-079: Per-conversation Library authority](../../../backlog/decisions/079-console-library-conversation-authority.md)
- [ADR-082: Per-chat private scratch](../../../backlog/decisions/082-console-per-chat-private-scratch-space.md)
- [ADR-089: Full semantic capture policy](../../../backlog/decisions/089-console-full-semantic-capture-policy.md)
- [ADR-090: Displayable thinking ownership](../../../backlog/decisions/090-console-thinking-block-ownership-and-replay.md)
