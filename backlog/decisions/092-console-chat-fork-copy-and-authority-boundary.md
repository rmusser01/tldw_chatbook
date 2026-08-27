# ADR-092: Define the Console chat-fork copy and authority boundary

Status: Accepted

Date: 2026-08-26

Related Task: Not yet assigned

Related Spec: [Console chat fork from a message](../../Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md)

Extends: ADR-033, ADR-044, ADR-052, ADR-069, ADR-079, ADR-082, ADR-089, and
ADR-090

## Context

The Console exposes per-message actions such as Copy, Edit, and Regenerate, but cannot
branch an earlier visible point into a new chat. Users need to explore a different
direction without changing the source conversation's active leaf, selected variant,
title, history, or live tab.

The apparent operation—copy messages through one boundary—crosses several ownership
models. Console has a native active tree plus render-only TOOL/activity markers;
visible variants may be session-only; attachments, citations, thinking, images, and
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

Opening the naming dialog captures a source/session/message/variant version fence.
Confirming revalidates that exact fence and builds an immutable allowlisted snapshot.
A changed path, boundary, content version, or visible variant fails before mutation;
the operation never silently switches to current state.

The visible variant of every copied lineage message is projected directly. A
session-only visible sibling does not need to be kept or persisted in the source
first. Forking never calls a source-mutating sibling/Keep path.

### 3. All fork-owned identities are fresh and relationships are remapped

The fork allocates fresh conversation/session, native message, persisted message,
turn, generation/variant, attachment, citation, and other copied sidecar identities.
Parent links are remapped within their correct native or durable namespace. An internal
cross-reference is remapped when its target is in the snapshot; an explicitly stable
external record may remain referenced; every other relationship is cleared honestly
or rejected if required. No fork-owned record retains a dangling source-message owner.

For a durable source, the new conversation uses the existing schema:

- `root_id` = the source conversation's canonical root;
- `parent_conversation_id` = the source conversation ID; and
- `forked_from_message_id` = the original persisted boundary message ID.

No migration is introduced.

### 4. Copy user-visible durable content through an allowlist

The snapshot includes stable USER/ASSISTANT content, the selected text or generated-
image variant, sent attachments, stable user-visible citations/source notices,
displayable
thinking or text-free proprietary evidence owned by that generation, bounded selected-
generation provenance, and non-empty stopped partial responses. Message-owned rows
receive new IDs. Stable external Library/source references may remain references.

It also copies declarative future-turn configuration: role/persona identity, system
prompt, provider/model and compatible parameters, speech preferences, runtime mode,
source/RAG/item-scope settings, effective Library policy values, and declarative
project-binding selection. The fork stays in Default Chats or the same named Workspace.

Snapshot construction fails before publication when required in-scope content cannot
be cloned honestly. New sidecar types are excluded until explicitly added to the
allowlist and tested.

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
regeneration metadata, remap or clear message references, and render the forked card as
the normal missing-video tombstone. Never copy or alias ephemeral bytes, paths, URLs,
or store keys.

### 6. Copy configuration, not authority

The new live session receives fresh private scratch under ADR-082. Scratch paths,
files, lease generations, selected local roots, approvals, tool grants, permission
decisions, and resolved instruction bodies are never copied.

The same Workspace association does not bypass current binding or read/write
validation. Project instructions retain only declarative selection inputs and undergo
fresh fingerprint validation and preflight under ADR-069 before use.

Effective Library policy values seed a new conversation policy owner and revision
under ADR-079. RAG configuration may copy, but retrieved evidence and memory do not.
Permission and privacy policy are re-resolved for future operations. Next-send and
per-conversation exchange-capture overrides are not copied; the fork uses the current
canonical global/default capture policy under ADR-089.

### 7. Durable and temporary forks have different ancestry

A durable source produces a durable fork in one ChaChaNotes transaction containing the
conversation, copied messages and included sidecars, active leaf, and required policy
rows. The source is version-checked but never updated. Workspace registry membership
remains an idempotent postcommit projection; its failure does not roll back the
committed fork.

An explicitly temporary source produces a temporary fork and no durable rows. If that
fork is saved later, it becomes an independent durable root with its own `root_id` and
null `parent_conversation_id`/`forked_from_message_id`. Saving the fork neither saves
nor mutates the temporary source. A persistable-but-not-yet-durable source follows the
same independent-root ancestry rule because no durable source identity existed when
the fork was made.

### 8. Commit precedes publication and retries are idempotent

For durable forks, the application preallocates every fork-owned ID and binds that
immutable expected write set to the validated snapshot/submission token. All fork rows
commit before a live session is published. Retry after an ambiguous result queries
those exact IDs: a complete matching write set is the same success, no rows may be
retried, and a partial or mismatched collision is an error. No new idempotency column
is required.

A precommit failure leaves the naming dialog and title available for Retry and creates
no fork. A postcommit Workspace, hydration, registration, or tab-activation failure is
a degraded success: the UI states that the named fork exists and provides a way to
open it from the Console rail. It never reports rollback or duplicates the
conversation.

Temporary forks are fully constructed before store registration. Replaying a
submission uses the same session ID; activation failure leaves the registered fork
discoverable.

### 9. The UI exposes the boundary before commit

Eligible real USER/ASSISTANT messages show a text-labelled `Fork` action immediately
before Regenerate, with tooltip/dialog label `Fork chat`. The conflict-free selected-
message shortcut is `f` and must appear in the action guide and screen help.

The small modal focuses an editable title defaulted to `Forked from <source title>`,
accepts Enter, cancels with Escape, and displays the exact cutoff/variant plus the
fresh-authority and video exclusions. A selected row shows a disabled reason in
visible action-help text and `f` repeats it; the design does not rely on disabled-button
focus, color, or hover. Double submission is fenced and the button shows `Forking…`
while in flight.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Create a message sibling | Mutates one conversation tree and does not create an independent chat. |
| Open the same conversation in another tab | Both tabs share message and active-leaf ownership, violating source isolation. |
| Deep-copy the live session | Copies transient runtime state and authority, retains stale IDs, and has no durable sidecar contract. |
| Copy rendered transcript rows | Includes synthetic TOOL/activity presentation and loses canonical tree relationships. |
| Keep the visible variant before copying | Mutates the source and makes the fork boundary depend on a separate write. |
| Text-only or best-effort copy | Silently loses supported user-visible attachments, citations, media, thinking, or configuration. |
| Copy the whole hidden branch tree | Retains unselected alternatives and weakens the meaning of the chosen boundary. |
| Share scratch, approvals, continuation, or recovery state | Transfers authority and may leak files or resume work under the wrong identity. |
| Copy video bytes | Contradicts ADR-044's ephemeral, explicit-save-only contract. |
| Persist a temporary source automatically | Changes the original's privacy and lifecycle without consent. |

## Consequences

- Forking needs a dedicated pure projection contract and persistence operation rather
  than reuse of the existing Workspace-link method.
- Durable conversation creation must expose the schema's existing lineage fields and
  atomically write all included sidecars.
- Variant, attachment, citation, thinking, image, and future sidecar owners need
  explicit fork projectors; unknown owners fail closed.
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
- New per-message data does not become forkable accidentally; it must join the
  allowlist with identity, privacy, and persistence tests.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md)
- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-044: Ephemeral generated-video storage](044-ephemeral-generated-video-storage-playback-and-streaming.md)
- [ADR-052: Conversation memory and compaction](052-console-conversation-memory-and-compaction-policy.md)
- [ADR-069: Project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-079: Per-conversation Library authority](079-console-library-conversation-authority.md)
- [ADR-082: Per-chat private scratch](082-console-per-chat-private-scratch-space.md)
- [ADR-089: Full semantic capture policy](089-console-full-semantic-capture-policy.md)
- [ADR-090: Displayable thinking ownership](090-console-thinking-block-ownership-and-replay.md)
