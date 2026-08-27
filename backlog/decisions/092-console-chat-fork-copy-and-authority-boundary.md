# ADR-092: Define the Console chat-fork copy and authority boundary

Status: Accepted (revision 1: clarifies action capacity, modal lifecycle, lineage-wide
variant fencing, current source contracts, and minimal retry ownership)

Date: 2026-08-26

Related Task: Not yet assigned

Related Spec: [Console chat fork from a message](../../Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md)

Extends: ADR-033, ADR-044, ADR-052, ADR-069, ADR-079, and ADR-082

## Context

The Console exposes per-message actions such as Copy, Edit, and Regenerate, but cannot
branch an earlier visible point into a new chat. Users need to explore a different
direction without changing the source conversation's active leaf, selected variant,
title, history, or live tab.

The apparent operation—copy messages through one boundary—crosses several ownership
models. Console has a native active tree plus render-only TOOL/activity markers;
visible variants may be session-only; attachments, citations, images, and
videos have different storage contracts; durable messages use a separate identity
namespace; and each live session owns security-sensitive scratch, instruction
preflight, permissions, and recovery state.

A generic object copy or message-sibling operation would either mutate the source,
copy authority, retain stale owner IDs, or produce a fork that silently loses content.
The feature therefore needs a canonical projection and an explicit durable/temporary
lineage policy.

## Decision

### 1. A fork is a new chat

Forking creates a new Console session and, for a durable source, a new durable
conversation. It is not a message sibling, a Regenerate variant, a Workspace
membership link, or another tab pointing at the same conversation ID. The source
remains open and receives no mutation.

The selected boundary is included. A user-message boundary does not automatically
generate an assistant response.

### 2. The copy source is one immutable active-lineage snapshot

The canonical source is the native USER/ASSISTANT parent chain from the root through
the selected stable message. Rendered transcript output is not a source of truth
because it contains spliced TOOL/activity rows. Messages after the boundary, off-path
branches, display-only rows, and unselected variants are excluded.

Opening the naming dialog captures a source/session version fence plus, for every
message in the lineage, its content token and selected text-sibling identity/token.
Generated-image browsing has a separate screen-owned fence:
`(native_message_id, selected_position, browse_revision)` plus a deterministic
fingerprint of the selected attachment and canonical generation metadata. Confirming
revalidates the store-owned tree/text fence and the screen-owned image fence on the
Textual app loop before one transition to commit. A changed path, boundary, content,
text sibling, image position/revision, or image payload anywhere in the lineage fails
before mutation; the operation never silently switches to current state. No positional
generation-envelope ID/version is invented.

The visible variant of every copied lineage message is projected directly. A
session-only visible sibling does not need to be kept or persisted in the source
first. Forking never calls a source-mutating sibling/Keep path.

For a durable source, however, every node in the copied lineage must already have a
persisted message ID, including the boundary used by `forked_from_message_id`. A
settled message left unsaved by a persistence failure is ineligible until normal source
persistence succeeds. Forking never repairs or persists the source implicitly.

### 3. All fork-owned identities are fresh and relationships are remapped

The fork allocates fresh conversation/session, native message, persisted message,
turn, real text generation/variant, and other independently identified mutable copied-
record identities. Attachment and generation-metadata rows are rebuilt under fresh
message owners while retaining checked positional ordering; they do not have invented
row IDs. Immutable governed citation trace/payload identities are preserved; only an
eligible durable message-owner link is fresh. Parent links are remapped within their
correct native or durable namespace. An internal cross-reference is remapped when its
target is in the snapshot; an explicitly stable external record may remain referenced;
every other relationship is cleared honestly or rejected if required. No fork-owned
record retains a dangling source-message owner.

For a durable source, the new conversation uses the existing schema:

- `root_id` = the source conversation's canonical root;
- `parent_conversation_id` = the source conversation ID; and
- `forked_from_message_id` = the original persisted boundary message ID.

No migration is introduced.

### 4. Copy user-visible durable content through an allowlist

The snapshot includes stable USER/ASSISTANT content, the selected text or generated-
image variant, sent attachments, bounded selected-generation provenance, and non-empty
stopped or failed partial responses. Inline citation markers/footer text already in the
message body copy as text. For a durable source with active governed provenance, a
durable fork creates a fresh message-owner link to the same immutable trace/payload
only after current namespace, revocation, body, revision, and fingerprint checks. A
body mismatch or failed active-trace check aborts; canonically unavailable provenance
stays unavailable without an active link. Temporary forks copy no governed citation
trace, payload, owner, or transient presentation state, and later promotion does not
reconstruct it.

It also copies declarative future-turn configuration: role/persona identity, system
prompt, provider/model and compatible parameters, speech preferences, runtime mode,
source/RAG/item-scope settings, effective Library policy values, and declarative
project-binding selection. The fork stays in Default Chats or the same named Workspace.

Snapshot construction fails before publication when required in-scope content cannot
be cloned honestly. New sidecar or generation-envelope types are excluded until
explicitly added to the allowlist and tested.

### 5. Exclude runtime, recovery, derived, and accounting state

The fork excludes drafts, prefill, staged files/evidence, later turns, prompt queues,
runs, wakeups, fleet/subagent state, agent todos, TOOL/activity records, tool results,
review state, feedback, annotations, original-attempt previews, provider continuation,
turn preparation, dispatch checkpoints, recovery owners, retry state, usage/cost
ledger entries, serialized provider history, retrieval results, compaction/memory
records, caches, scroll/focus/selection state, undo state, and speech playback.

Copied messages did not create new provider usage. Derived context is invalid under
the fork's fresh message identities and must be recomputed.

ADR-044 remains authoritative for video: copy the human-readable marker and bounded
regeneration metadata, remap an in-snapshot `source_image_message_id`, clear an
out-of-snapshot one, and render the forked card as the normal missing-video tombstone.
Never copy or alias ephemeral bytes, paths, URLs, or store keys.

### 6. Copy configuration, not authority

The new live session receives fresh private scratch under ADR-082. Scratch paths,
files, lease generations, selected local roots, approvals, tool grants, permission
decisions, and resolved instruction bodies are never copied.

The same Workspace association does not bypass current binding or read/write
validation. Project instructions retain only declarative selection inputs and undergo
fresh fingerprint validation, notice, and preflight under ADR-069 before use.
`project_instruction_notice_key` is always cleared so source consent cannot suppress
the fork's fresh destination/locator notice.

Effective Library policy values seed a new conversation policy owner and revision
under ADR-079. RAG configuration may copy, but retrieved evidence and memory do not.
Permission policy is re-resolved for future operations.

### 7. Durable and temporary forks have different ancestry

A durable source produces a durable fork in one ChaChaNotes transaction containing the
conversation, copied messages and included sidecars, valid citation-owner links, active
leaf, required policy rows, and sanitized project-context JSON. That JSON contains the
retained enabled flag, declarative binding ID, and locator fingerprint with a null
`project_instruction_notice_key`; the fork path must not defer it to the current best-
effort postcommit helper. The source is version-checked but never updated. Workspace
registry membership remains an idempotent postcommit projection; its failure does not
roll back the committed fork.

An explicitly temporary source produces a temporary fork and no durable rows. If that
fork is saved later, it becomes an independent durable root with its own `root_id` and
null `parent_conversation_id`/`forked_from_message_id`. Saving the fork neither saves
nor mutates the temporary source. Its sanitized project-context JSON joins the atomic
promotion bundle rather than a subsequent best-effort write; omitted governed citation
provenance is not reconstructed.

A non-ephemeral, persistable source without durable IDs creates a saved independent-
root fork immediately using the durable transaction and `Creates: Saved chat` UI. The
fork's `root_id` is its preallocated conversation ID and its parent/fork columns are
null. The source is not persisted or mutated. Only an explicitly temporary source
creates an unsaved fork, so no third UI destination or publication mode is introduced.

Before registration, the detached fork clears every excluded field that ordinary
promotion could otherwise persist: usage/cost, context summary/boundary, pinned
prefill, staged evidence/attachments, continuation/recovery ownership, governed
citation-owner state, tool/activity state, project-instruction notice key, and video
path/store keys. Later promotion consumes only that sanitized fork state and never
rereads the source.

### 8. Commit precedes publication and retries are idempotent

For durable forks, the application preallocates one conversation ID and binds it to the
validated modal submission. All fork rows commit in one SQLite transaction before a
live session is published. Retry after an ambiguous return queries that ID: a row with
the expected root, parent, boundary, and title is the same atomic success; absence may
retry the transaction with the same ID; a conflicting row is an error. Transaction
atomicity supplies the bundle guarantee. No per-row digest, operation table, or new
idempotency column is introduced.

The atomic bundle includes the sanitized project-context JSON and eligible fresh
message-owner links to immutable citation traces. A failure at either write rolls back
the entire fork. The same project-context requirement applies as a transaction
contribution when a temporary fork is later promoted.

A precommit failure leaves the naming dialog and title available for Retry and creates
no fork. A postcommit Workspace, hydration, registration, or tab-activation failure is
a degraded success: the UI states that the named fork exists and provides a way to
open it from the Console rail. It never reports rollback or duplicates the
conversation.

Temporary forks are fully constructed before store registration. Replaying a
submission uses the same session ID; activation failure leaves the registered fork
discoverable.

Each async validation attempt also owns a monotonically increasing submission
generation. Escape/backdrop during validation invalidates it before dismissal. The
validator compares its captured generation on the app loop immediately before the
single transition to commit; a late result with a stale generation is a no-op. Once
the committing transition is published, cancellation is unavailable and the modal
explains that creation is finishing.

### 9. The UI exposes the boundary before commit

Eligible real USER/ASSISTANT messages show a text-labelled `Fork` action immediately
before Regenerate, with tooltip/dialog label `Fork chat`. The conflict-free selected-
message shortcut is `f` and must appear in the action guide and screen help.

The USER/ASSISTANT primary row contains Copy, applicable Speak/Stop, Edit, Fork,
applicable Regenerate/Retry, applicable Continue, and `More…`. Save as…, Helpful, Not
helpful, and Delete move into `More…`; media controls live with their media card. The
menu binds its opener's message/action IDs, closes before dispatch, and never rereads
the current selection. Escape/click-away, selection loss, removal, or recomposition
closes it safely; focus returns to the opener only if that exact opener still exists.
Otherwise focus returns to the selected transcript row, or to the composer when
selection was cleared. Specialized compact display rows keep their one or two direct
actions and never show Fork.

The small modal focuses an editable title defaulted to `Forked from <source title>`
(`Untitled chat — fork` when untitled), accepts Enter, and shows a bounded boundary
excerpt, response count/selection when relevant, and `Creates: Saved chat · <place>`
or `Creates: Temporary chat · Save later to keep it`. Fork titles reuse
`derive_console_session_title` with a fixed 60-character bound.

A non-ephemeral source that lacks durable IDs still says `Creates: Saved chat`: its
fork is committed immediately as an independent durable root without saving the
source. Temporary modal copy also states that citation markers remain as text while
inspectable governed provenance is not copied.

Editing, validating, committing, precommit-error, stale-source, and committed-but-not-
opened states each define controls, status copy, Enter/Escape/backdrop behavior, and
focus. A commit cannot be cancelled, but an Escape request explains that it is
finishing. Normal success says that the saved/temporary fork opened and the original
remains open. The video warning states that copied video will be unavailable and names
Save a copy as the recovery. Other uncommon exclusions use one disclosure.

A selected row shows an ineligible reason in visible action-help text and `f` repeats
it; the design does not rely on disabled-button focus, color, or hover. Complete
USER/ASSISTANT messages, non-empty stopped responses, and non-empty failed responses
are eligible. Pending, streaming, empty failed, and discarded messages are not.
Double submission is fenced and the button shows `Forking…` while in flight.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Create a message sibling | Mutates one conversation tree and does not create an independent chat. |
| Open the same conversation in another tab | Both tabs share message and active-leaf ownership, violating source isolation. |
| Deep-copy the live session | Copies transient runtime state and authority, retains stale IDs, and has no durable sidecar contract. |
| Copy rendered transcript rows | Includes synthetic TOOL/activity presentation and loses canonical tree relationships. |
| Keep the visible variant before copying | Mutates the source and makes the fork boundary depend on a separate write. |
| Text-only or best-effort copy | Silently loses supported user-visible attachments, citations, media, or configuration. |
| Show every message action directly | Produces ten to fifteen equal-weight controls on richer rows; a labelled `More…` keeps secondary actions discoverable without crowding the fork boundary. |
| Compare a digest for every copied row on retry | SQLite already commits the fork as one transaction. One preallocated conversation ID and lineage check prevent duplication with much less machinery. |
| Copy the whole hidden branch tree | Retains unselected alternatives and weakens the meaning of the chosen boundary. |
| Share scratch, approvals, continuation, or recovery state | Transfers authority and may leak files or resume work under the wrong identity. |
| Copy video bytes | Contradicts ADR-044's ephemeral, explicit-save-only contract. |
| Persist a temporary source automatically | Changes the original's privacy and lifecycle without consent. |

## Consequences

- Forking needs a dedicated pure projection contract and persistence operation rather
  than reuse of the existing Workspace-link method.
- Durable conversation creation must expose the schema's existing lineage fields and
  atomically write all included sidecars.
- Variant, attachment, citation, image, and future sidecar owners need explicit fork
  projectors; unknown owners fail closed.
- Copied historical messages are independent records and may increase durable storage,
  while immutable blob sharing is permitted only where the backing store already
  guarantees ownership-safe sharing.
- A fork can show an expired video even while the source can still play it. The modal
  and tombstone make that intentional boundary visible.
- The source and fork may share declarative Workspace context but never live scratch or
  cached instruction/permission authority.
- Temporary forks cannot acquire durable ancestry retroactively; later save creates a
  new root.
- Database commit and UI activation remain honestly separate. Postcommit presentation
  failures require recovery UX instead of rollback claims.
- The selected-message action row gains one `More…` menu and moves existing secondary
  actions into it; direct keyboard shortcuts remain available for common actions.
- New per-message data does not become forkable accidentally; it must join the
  allowlist with identity, privacy, and persistence tests.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md)
- [ADR-024: RAG citation provenance and source resolution](024-rag-citation-provenance-and-source-resolution.md)
- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-044: Ephemeral generated-video storage](044-ephemeral-generated-video-storage-playback-and-streaming.md)
- [ADR-052: Conversation memory and compaction](052-console-conversation-memory-and-compaction-policy.md)
- [ADR-069: Project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-079: Per-conversation Library authority](079-console-library-conversation-authority.md)
- [ADR-082: Per-chat private scratch](082-console-per-chat-private-scratch-space.md)
