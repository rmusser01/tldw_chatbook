# Console chat fork from a message

**Date:** 2026-08-26

**Status:** Owner-approved direction; critique amendments incorporated; awaiting final
written-spec approval

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
captures the visible message variants and supported user-visible sidecars, assigns new
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
- included attachments, citations, generated images, and other
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
- Give the fork fresh conversation, message, turn, and variant identities, and rebuild
  copied sidecars under those new owners.
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

This delivery also replaces the overloaded USER/ASSISTANT action row with one stable
primary row and a labelled `More…` menu:

- the primary row keeps Copy, Speak/Stop when available, Edit, Fork when eligible,
  Regenerate/Retry when available, Continue when available, and `More…`, in that
  order;
- `More…` contains Save as…, Helpful, Not helpful, and Delete when those actions are
  available;
- image/video view, variant, playback, and save-copy controls live with their media
  card rather than expanding the generic message-action row; and
- compact TOOL/display-only rows with one or two specialized actions keep those
  actions direct and never show Fork.

Opening `More…` binds the opener's native message ID and the available action IDs,
then moves focus to its first available item. Up/Down traverse, Enter first closes the
menu and then invokes the existing action or confirmation flow with that captured
message ID; dispatch never rereads the current selection. Escape or click-away closes
the menu. A selection change, message removal, or transcript recomposition also closes
it without dispatch. Focus returns to `More…` only when that exact opener still
exists; otherwise it returns to the selected transcript row, or to the composer when
selection was cleared. A captured target that became unavailable fails with the normal
visible action reason rather than acting on a different message. The menu gains no
screen-wide shortcut: direct high-frequency actions retain `c`, `e`, `f`, and `r`.
The selected-row guide names only the direct actions plus `More…`; the open menu names
its own contents. Existing Edit & resend copy changes from “forks a new branch” to
“creates a new response branch in this chat,” reserving **Fork chat** for this new-chat
operation.

The action appears for real USER and ASSISTANT tree nodes. It does not appear on
render-derived system/status rows, TOOL/activity markers, original-attempt previews,
or other display-only transcript rows. A real node is enabled when its content is a
stable fork boundary:

- complete USER messages are eligible;
- complete ASSISTANT messages are eligible;
- stopped ASSISTANT messages with non-empty visible content are eligible and the
  dialog labels the boundary as a partial response;
- failed ASSISTANT messages with non-empty visible content are eligible and the
  dialog labels the boundary as a failed partial response;
- pending or streaming targets are disabled until they settle;
- failed targets without meaningful visible content and every discarded target are
  disabled; and
- when the source conversation is durable, every node in the copied lineage must have
  a persisted message ID. A settled boundary left unsaved by a persistence failure is
  disabled with `This message has not been saved yet. Try Fork again after it is
  saved.` Forking never persists the source implicitly; and
- deleted, stale, or no-longer-on-the-captured-path targets fail confirmation because
  such rows are normally no longer renderable.

If a row is temporarily ineligible, selecting the row shows the reason in visible
action-help text and pressing `f` repeats that reason. The design does not assume that
a disabled Textual button itself can receive focus. Color and hover-only tooltip text
are not the sole explanation. A run occurring after an otherwise stable earlier
boundary does not disable that earlier boundary.

### Confirmation dialog

Opening the action captures an initial source fence and shows a compact modal:

```text
Fork chat

Through Assistant 8: “The retrieval results suggest…”
8 messages · showing response 2 of 3
Creates: Saved chat · Research Workspace

Includes sent attachments and cited source details.
Starts with new private working files; file and tool access will be requested again.

Name
[Forked from Research notes                              ]

                                      [Cancel] [Fork chat]
```

The excerpt is a plain-text, whitespace-collapsed, markup-escaped excerpt from the
actual visible boundary, limited to two terminal lines and capped by cell width with an
ellipsis. It is preceded by the user-visible speaker and ordinal. For a user boundary,
the summary says `Through User <n>` and adds `No reply will be generated`. Stopped and
failed non-empty boundaries say `Partial response` and `Failed partial response`.

The result row is always present. A durable fork says `Creates: Saved chat · <Chats or
Workspace name>`. A temporary fork says `Creates: Temporary chat · Save later to keep
it`, followed by `Saving this fork will not save the original chat.` The message count
is the number of USER/ASSISTANT nodes in the projected lineage. Response-variant detail
appears only when it adds information.

Attachment and citation facts are conditional. A durable fork with verified governed
provenance says `Includes sent attachments and cited source details.` A temporary fork
says `Citation markers remain in the message text; source inspector details are not
copied.` The common exclusions live behind a short `What is not copied` disclosure
opened from the modal; its body names runs, tool history, drafts, staged files,
temporary working files, prior permissions, and (for temporary forks) inspectable
citation provenance. When the copied path contains a generated video, the warning
stays in the main flow:
`This video will appear as unavailable in the fork. Save a copy first if you need the
file.` No other caveat competes with the title field.

The title field is focused with its contents selected so typing immediately replaces
the default. Enter confirms and Escape cancels. The default is `Forked from
<source display title>`; a missing title uses `Untitled chat — fork`.

Fork titles reuse `derive_console_session_title` rather than introduce another parser.
A new `CONSOLE_FORK_TITLE_MAX_LENGTH = 60` caps the title input and the final normalized
value. The helper collapses whitespace to a single line and appends its existing ASCII
ellipsis when truncation is necessary; the caller rejects a blank result. The same
normalized value is used by the tab, conversation row, Workspace projection, retry
recovery, and success copy.

Repeated Enter, clicking twice, or an event replay cannot create a second fork for the
same modal submission. The complete modal state contract is:

| State | Visible controls and focus | Enter | Escape / backdrop |
| --- | --- | --- | --- |
| `editing` | Editable selected title; Cancel and Fork chat; title focused | Validate and begin | Cancel. A primary-button backdrop request has the same result. |
| `validating` | Title remains focused; Fork chat disabled; `Checking fork…` | No-op | Invalidate this submission and cancel. A late validation completion is a no-op. |
| `committing` | Title disabled; `Forking…`; a focusable status row receives focus | No-op | Do not dismiss. Replace the status once with `Fork creation is finishing and can no longer be cancelled.` |
| `precommit-error` | Preserved editable title; Cancel and Retry; Retry focused | Retry | Cancel; nothing was created. |
| `stale-source` | Read-only title; Close; Close focused | Close | Close. Copy says `This chat changed. Close and choose Fork again.` |
| `created-not-opened` | Read-only title; Close and Open fork; Open fork focused | Open fork | Close the dialog without deleting the created fork. |

The backdrop is never a confirmation. Descendant disclosures receive Escape first.
Every state change updates one text status row; color is supplementary. Normal success
closes the modal, activates the fork, and announces `Fork created and opened. The
original chat is still open.` A temporary success includes `Temporary` in that message
and uses the existing temporary-chat rail treatment rather than adding a new badge.

The controller assigns each validation attempt a monotonically increasing submission
generation. Escape/backdrop cancellation during `validating` invalidates that
generation before dismissing. The async validator must compare the captured generation
on the app loop immediately before the single transition to `committing`; a mismatch
returns without writing or publishing anything. Once that transition is published,
the operation is no longer cancellable. This is the single-shot async cancellation
boundary required by ADR-031.

### Success and failure

On success, the new fork is registered as a Console session, appears in the same Chats
or Workspace rail location, and becomes the active Console tab. The original tab
stays open and retains its exact state. The success notice repeats whether the fork is
saved or temporary.

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
- active-lineage identity through the boundary;
- for every lineage message, its native ID, content/version token, and selected text
  sibling ID plus stable content token when a text sibling is displayed;
- for every displayed generated-image choice, the screen-owned tuple
  `(native_message_id, selected_position, browse_revision)` plus a deterministic
  fingerprint of the selected attachment bytes/type/name and canonical generation
  metadata; and
- source durability mode and displayed title.

The title remains user-editable independently of this fence.

### Confirm-time validation

Confirming re-resolves the source by the captured session ID, never by whichever tab is
currently active. Under the store's mutation serialization it verifies that the
session, boundary, path, message version, and visible variant still match the fence.
Every lineage entry is revalidated; a variant change on an earlier copied message is
as stale as a change at the boundary.

For a durable source, confirmation also requires a persisted message ID for every
copied lineage node, including the boundary used by
`forked_from_message_id`. A missing ID produces the same unsaved-boundary error and no
write; the fork operation never repairs or persists the source as a side effect.

Text selection and generated-image selection have different owners. The store
revalidates the canonical tree, message content tokens, and selected text siblings.
The session controller captures and revalidates the screen-owned generated-image
position and monotonic browse revision on the Textual app loop. It also recomputes the
attachment/metadata fingerprint so payload replacement cannot masquerade as the same
position. The app-loop revalidation and transition to `committing` are one serialized
state change; any later browse request cannot alter the committed snapshot. No
generation-envelope ID or version is invented for positional metadata.

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
- real text generation/variant IDs when present; and
- other independently identified mutable copied records.

Attachment and generation-metadata rows have no independent IDs. They are rebuilt
under each fresh message owner while preserving their checked `position` ordering.
Immutable governed citation traces and payloads are not fork-owned and keep their
canonical identities; only an eligible durable message-owner link is fresh.

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
| Visible generated-image variant | Copy the selected renderable image payload and its bounded generation provenance under the fresh message owner. Do not copy unselected images. |
| Partial assistant text | Copy non-empty `stopped` or `failed` visible content with the same terminal state; do not make it resumable. |
| Sent attachments | Rebuild each attached item's ordered `(message_id, position)` association under the fresh message owner. An immutable content-addressed blob may be shared only if its store contract makes sharing ownership-safe. Never retain a source filesystem path as authority. |
| Citations and source notices | Inline markers and source-footer text already in the selected message body copy as text. For a durable source with an active governed trace, a durable fork preserves the immutable trace/payload identities and creates only a fresh message-owner link after current revocation, namespace, message-body, revision, and fingerprint checks pass. Canonically revoked/unavailable provenance stays unavailable and receives no active owner link. A body mismatch or failed active-trace check aborts rather than attaching the trace to different text. A temporary fork copies no governed trace, payload, or owner state; its inspector is unavailable, and later promotion does not reconstruct the omitted association. Transient settled-message presentation state is never treated as citation authority. |
| Generation provenance | Copy the selected variant's bounded provider/model/request provenance needed to explain or regenerate the visible content. |
| Role identity | Copy character/persona identity, persona memory mode, human display override, and system prompt as declarative configuration. |
| Future-turn model configuration | Copy selected provider, model, compatible generation parameters, agent/runtime mode, source/RAG selector, context-window/compaction policy values, and speech preferences. |
| Conversation scope | Preserve Default Chats versus the same named Workspace ID and the same durable item/source scope configuration. Workspace membership is projected after durable commit through the existing idempotent seam. |
| Library policy | Seed a new conversation policy owner from the source's effective runtime `auto_retrieve` and `assistant_access` values (persisted as `auto_retrieve_on_send` and `assistant_library_access`). Start its own revision history. |
| Project-instruction selection | Copy only `project_instructions_enabled`, the declarative binding ID, and locator fingerprint. Set `project_instruction_notice_key` to null, then perform fresh validation, notice, and preflight before future use. |

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
| Local authority | Allocate fresh private scratch lazily. Do not copy scratch locators/files/leases, approvals, tool grants, selected permission decisions, or resolved project-instruction bodies. |
| Transient UI | Exclude scroll/focus/selection state, expansion state, toasts, modal state, cached render rows, undo stacks, and speech playback. |

### Generated video

ADR-044 stores generated-video bytes in a message-ID-keyed ephemeral store. A fork
therefore never copies or aliases those bytes. It copies the visible human-readable
video marker and bounded regeneration metadata into the new message. An image-to-video
`source_image_message_id` is remapped only when that image message is in the snapshot;
otherwise it is cleared. Every path, URL, store key, and old message owner is cleared.
The fork renders the standard named tombstone with a
regenerate action even when the source video still plays. The dialog discloses this
before confirmation.

This behavior is not a partial-copy error: video ephemerality is the modality's
canonical durable contract. A later explicit `Save a copy…` remains the only way to
export video bytes.

### Selected generation envelope

The fork copies the user-displayable parts of the selected generation atomically:
visible answer, selected image attachment if any, and bounded generation provenance.
It deliberately strips private continuation and original usage accounting. Any future
message-owned envelope remains excluded by the allowlist until its own accepted
decision defines fork behavior. If the current generation envelope cannot express the
selected projection without violating its version contract, the fork fails before
commit rather than inventing a second generic envelope framework.

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
The fork always clears `project_instruction_notice_key`; fresh validation may therefore
show the current locator/destination notice even when the source had already accepted
one.

### Library, RAG, and memory

The fork captures the source's effective declarative Library policy values and writes
them into a fresh policy owner/revision when the fork becomes durable. It copies RAG
selector and item-scope configuration, but not retrieved chunks, evidence staging,
memory summaries, compaction records, or cache entries. Those derived records are
branch-valid against message identity under ADR-052 and must be recomputed.

### Permissions and one-shots

Permission decisions, MCP approvals, local-tool grants, attachment upload approvals,
and one-shot settings are never copied. The fork applies the current canonical global,
Workspace, provider, and per-principal rules when a future action is attempted.

## Durable and temporary ownership

### Durable source

A durable fork is one SQLite transaction containing the new conversation, every copied
message in lineage order, all included message-owned sidecars and valid citation-owner
links, the fork's active leaf, required per-conversation policy rows, and the sanitized
project-context JSON (`project_instructions_enabled`, declarative binding ID, locator
fingerprint, and a null `project_instruction_notice_key`). It uses:

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

The detached temporary builder clears every field excluded by the copy allowlist before
the new session is registered. In particular it clears usage/cost records, context
summary and boundary, pinned prefill/one-shots, staged evidence and attachments,
continuation/recovery owners, project-instruction notice key, governed citation-owner
state, tool/activity state, and video paths/store keys. Promotion later consumes this
already-sanitized session; it must not rediscover or recopy state from the source or
reconstruct omitted citation provenance.

If the temporary fork is later promoted/saved, it becomes an independent durable root:

- `root_id` is its new conversation ID;
- `parent_conversation_id` is null; and
- `forked_from_message_id` is null.

The temporary source remains temporary and unchanged. Process-local provenance is not
converted into a foreign-key claim.

An ordinary non-ephemeral, persistable session that has not yet acquired durable IDs
creates a **saved independent root** immediately. The modal therefore uses
`Creates: Saved chat · <place>`, and the normal durable transaction preallocates the
fork conversation ID with `root_id` equal to that ID and null
`parent_conversation_id`/`forked_from_message_id`. The source session is not persisted
or otherwise changed. This avoids a third rail state: only an explicitly temporary
source produces an unsaved fork.

## Commit, publication, and idempotency

### Durable operation

The flow is:

1. Validate the dialog fence and build the immutable snapshot.
2. Preallocate one fork conversation ID and bind it to the modal submission.
3. Write the entire durable projection in one ChaChaNotes transaction, including the
   sanitized project-context JSON and any fresh links to still-valid immutable citation
   traces.
4. After an ambiguous return, query that conversation ID. If the row exists with the
   expected source root, parent conversation, fork boundary, title, and active leaf,
   treat it as the same completed operation. If it does not exist, retry the atomic
   transaction with the same conversation ID. A conflicting row is an error.
5. Project Workspace membership idempotently after commit.
6. Hydrate/register the new live session from the committed projection.
7. Activate the new tab.

The preallocated conversation ID is the durable idempotency key for an ambiguous
postcommit result. SQLite transaction atomicity means a matching conversation row
proves the message/sidecar bundle committed; absence proves that bundle did not. No
per-row digest, operation table, or second idempotency framework is introduced.

No live fork is published before the database transaction succeeds. Transaction
failure rolls back the new conversation, messages, sidecars, citation-owner links,
active leaf, policy rows, and project-context JSON together. The fork creation path
must not defer project-context persistence to the current best-effort postcommit
helper. A postcommit Workspace link, hydration, mount, or activation failure is a
degraded success, not a rollback.

### Temporary operation

The snapshot is converted to a complete detached session before registration. Store
registration publishes it under one new session ID. Activation follows. Replaying the
same modal submission uses the same session ID and resolves to the already registered
fork rather than appending another one.

If that temporary fork is later promoted, its sanitized project-context JSON is a
transaction contribution to the existing atomic promotion bundle, not a subsequent
best-effort write. Promotion reload therefore either sees all three retained
declarative fields with a null notice key or rolls back the complete save.

## Component ownership

Implementation should preserve the Console module ownership ratchet:

- `Chat/console_chat_fork.py` owns frozen snapshot/copy records, the canonical
  composite `ForkEligibility`/`ForkFence`, copy allowlists, identity remapping, and
  pure projection/fingerprint helpers.
- `ConsoleChatStore` supplies serialized canonical tree, content-token, and text-
  sibling reads, validates the store-owned portion of the source fence, and registers
  detached sessions. It does not claim ownership of generated-image browsing or
  persist by deep-copying its internal dictionaries.
- `ChatPersistenceService` owns the atomic durable write and extends the
  conversation-creation seam to accept existing lineage columns, sanitized project-
  context JSON, and governed citation-owner link contributions. The existing
  `fork_conversation_into_workspace` method remains a differently named membership
  link and must not be reused as the clone implementation.
- `UI/Console_Modules/image.py` extends the existing screen-owned generation-browse
  state with a monotonic revision and exposes narrow app-loop capture/revalidation;
  no image-fence business logic is added to the screen monolith.
- `UI/Console_Modules/session.py` owns dialog orchestration, source-session targeting,
  the monotonic validation/cancellation generation, coordination of generated-image
  fence reads, publication, activation, and partial-success recovery.
- `UI/Console_Modules/message.py` dispatches the selected-message request through a
  named session-controller callable wired in `UI/Console_Modules/wiring.py`; it does
  not absorb persistence or tab logic.
- `Chat/console_message_actions.py` owns action ordering and presentation. It consumes
  the store-derived `ForkEligibility`; it does not infer active-path, deletion, or
  staleness from one message.
- `Widgets/Console/console_transcript.py` renders the action, tooltip, key binding, and
  action guide. Its USER/ASSISTANT `More…` menu captures a stable message/action target,
  closes before dispatch, and owns safe teardown/focus restoration.
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
| Validation is cancelled before commit | Invalidate the submission generation; any late validator completion is a no-op and creates nothing. |
| Title invalid | Inline validation; no snapshot persistence. |
| Attachment/image projection unsupported or corrupt | Fail closed before commit and name the affected content class; never omit silently. |
| Active citation trace is revoked, body-mismatched, or cannot be linked safely | Preserve the canonical unavailable state when already unavailable; otherwise fail before commit rather than claim provenance for different text. Temporary forks truthfully omit inspectable provenance. |
| Project binding no longer validates | The fork may still be created with declarative association, but future use is unavailable until normal preflight succeeds; no stale body or permission is copied. |
| SQLite transaction fails | Roll back the complete fork; preserve dialog/title; Retry uses the same safe submission identity only after confirming no commit. |
| Transaction result is ambiguous | Resolve the preallocated conversation ID. Reuse the atomically committed fork when its root/parent/boundary/title match, retry with that ID when absent, and fail on a conflicting row. |
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
- Provider credentials, continuation tokens, endpoint secrets, and permission
  decisions are excluded by construction.
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
- New message/turn/variant IDs, rebuilt sidecar owners/positions, and parent remaps
  contain no source ownership.
- Stable external source references remain valid; internal references remap; dangling
  message references fail.
- Complete USER/ASSISTANT, stopped-nonempty, failed-nonempty, failed-empty,
  discarded, streaming, pending, deleted, and stale targets have the specified
  eligibility/reason behavior.
- A durable conversation with any unsaved node in the copied lineage is ineligible and
  remains unchanged; once normal persistence supplies all IDs, eligibility can be
  recomputed.
- Unknown sidecar types fail closed.
- Source objects and persistence rows are unchanged after success and every failure.

### Persistence tests with real in-memory SQLite

- Durable fork conversation ancestry uses existing root/parent/boundary columns.
- Messages, attachments, eligible citation-owner links, selected generation data,
  active leaf, policy rows, and sanitized project-context JSON commit atomically.
- Injected failure at every write stage leaves no partial fork.
- Failure at the project-context write rolls back the complete fork. Reload preserves
  exactly the enabled flag, binding ID, and locator fingerprint, with a null notice key.
- Ambiguous retry with the preallocated conversation ID returns the atomically
  committed matching fork, retries when absent, and rejects a lineage/title collision.
- The fork reloads with the same visible path and can accept a divergent next turn.
- An active citation trace reloads through a fresh owner link to the same immutable
  trace/payload identities. Revoked and body-mismatched traces never acquire a fork
  owner link.
- Workspace projection failure preserves the committed fork and reconciles without a
  duplicate.
- Derived memory, usage, continuation, checkpoints, feedback, and unknown future
  message envelopes do not appear in copied rows.

### Variant and media tests

- Text sibling selection, generated-image selection, attachments, and citations copy
  only the visible supported envelope.
- Changing an earlier copied text sibling or generated-image browse position while the
  dialog is open invalidates the fence. The generated-image case verifies the captured
  `(message_id, position, browse_revision)` and attachment/metadata fingerprint.
- Generated video always reopens as a named tombstone with regeneration metadata and
  no copied source path/store key/bytes.
- In-snapshot media references remap; missing required image/attachment payloads fail
  before commit.
- An out-of-snapshot video `source_image_message_id` is cleared; an in-snapshot source
  is remapped.

### Authority and temporary tests

- Source and fork receive different scratch roots and lease generations.
- Approvals/tool grants do not appear in the fork.
- Same Workspace association is retained while folder/project instruction authority is
  freshly validated.
- `project_instruction_notice_key` is null in every fork, including after temporary
  promotion.
- Library policy values are seeded into a fresh owner/revision; derived retrieval and
  compaction state is absent.
- Temporary-to-temporary fork writes no DB rows, diverges independently, and later
  promotion creates a durable root with null parent/fork columns.
- A non-ephemeral session without durable IDs creates a saved independent-root fork
  without persisting the source.
- Saving a temporary fork does not persist its temporary source.
- Temporary forks retain citation markers as text but have no governed trace/payload/
  owner copy; later promotion neither reconstructs nor exposes that omitted inspector
  provenance.
- Temporary promotion cannot persist source usage, context summary/boundary, pinned
  prefill, staged evidence/attachments, continuation/recovery, tool/activity state, or
  video store keys.
- Temporary promotion writes sanitized project-context JSON inside its atomic bundle;
  reload retains the three declarative fields and a null notice key.

### UI tests

- `Fork` renders immediately before `Regenerate` for eligible selected messages.
- `f`, button click, tooltip, guide, screen help, action IDs, and controller dispatch
  remain synchronized.
- USER/ASSISTANT primary rows contain direct actions plus `More…`; Save as, Helpful,
  Not helpful, and Delete live in that menu; media controls remain with media cards.
  Menu traversal and stable-target dispatch work by keyboard. The menu closes before a
  selected action launches its modal, closes on click-away/selection change/removal/
  recomposition, and restores focus only to a still-valid opener. Removing the opener
  with no remaining selection focuses the composer.
- Modal title selection, 60-character normalization, typing, Enter, Escape/backdrop,
  all six modal states, focus order, double-submit fencing, and narrow-terminal layout
  work in Textual Pilot.
- A barrier-controlled validator cancelled by Escape/backdrop cannot enter commit or
  create a fork when its late result returns.
- Modal summary distinguishes USER, ASSISTANT, partial, variants, attachments, and
  ephemeral-video behavior; it shows an excerpt and saved/temporary destination.
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
| Copy only plain text | Loses attachments, citations, images, configuration, and the user's selected variant. |
| Copy the complete hidden branch tree | Surprises users, retains unselected alternatives, and makes the chosen message an unclear boundary. |
| Copy scratch, approvals, continuation, or checkpoints | Transfers live authority/recovery ownership and can leak files or resume an operation under a different chat identity. |
| Copy ephemeral video bytes | Contradicts ADR-044's message-keyed ephemeral storage and explicit save-only escape hatch. |
| Auto-persist a temporary source before forking | Mutates the original and changes its privacy/lifecycle without consent. |
| Best-effort partial fork | Creates an apparently complete chat with silently missing user data. The safe full projection must fail before commit instead. |

## Related decisions

- [ADR-024: RAG citation provenance and source resolution](../../../backlog/decisions/024-rag-citation-provenance-and-source-resolution.md)
- [ADR-033: Application session state ownership](../../../backlog/decisions/033-application-session-state-ownership.md)
- [ADR-044: Ephemeral generated-video storage](../../../backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md)
- [ADR-052: Conversation memory and compaction](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md)
- [ADR-069: Project-instruction local state and preflight](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md)
- [ADR-079: Per-conversation Library authority](../../../backlog/decisions/079-console-library-conversation-authority.md)
- [ADR-082: Per-chat private scratch](../../../backlog/decisions/082-console-per-chat-private-scratch-space.md)
