# ADR-079: Workspace file inspector direct-user authority and save publication

- **Status:** Proposed
- **Date:** 2026-08-31
- **Related task:** N/A — design phase
- **Supersedes:** N/A
- **Extends:** [ADR-028: Settings owns workspace management; folders are file-tool access roots](028-settings-workspaces-category-and-folder-roots.md)
- **Relates to:** [ADR-069: Console project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md), [Agent Change Review design](../../Docs/superpowers/specs/2026-08-02-agent-change-review-design.md), [Workspace Files design](../../Docs/superpowers/specs/2026-08-31-workspace-files-inspector-design.md)

## Context

The Console needs a Workspace Files surface where a user can inspect and deliberately edit files belonging to any named workspace without activating it or changing the current Console task/session. Existing folder bindings already express the maximum local-filesystem authority of a workspace: read-only or read/write. Agent tool execution adds a separate approval layer.

Several boundaries make this more than a modal UI decision:

- a direct user Save must not be confused with an agent tool call or inherit agent permission prompts;
- a rendered file tree cannot remain authoritative after a binding is revoked, archived, or retargeted;
- Agent Change Review currently compares a baseline and terminal root snapshot, so a manual inspector save during that window could be attributed to the agent;
- a filesystem replacement can succeed even when a later durability step fails, so a binary success/failure API can mislead users or cause destructive retries;
- safe publication semantics and metadata preservation vary by platform; and
- filenames, file text, and Git output can contain terminal/markup control data;
- asynchronous filesystem and Git work needs bounded lifecycle and shutdown semantics; and
- Git decoration is useful but must not become a filesystem authority source, execute repository-controlled behavior, or mutate repository state.

Reusing File Notes would also import synchronization and database ownership that do not belong to direct workspace inspection.

## Decision

### 1. Workspace Files is a non-activating Console modal

Workspace Files is a near-full-screen modal owned by the current Console screen. It receives a stable workspace ID and displays a persistent notice naming both the inspected and active workspaces. Opening, navigating, saving, and closing it do not activate the inspected workspace or mutate Console context. Console admits only one visit: repeated activation focuses that visit, and another workspace cannot retarget or stack a second modal.

The modal owns one selected binding, one selected file, and at most one edit buffer. Recursive filtering is limited to the selected binding; changing bindings is an explicit, revalidated navigation transition rather than a workspace-wide search side effect. It uses the Console safe-dismiss contract for Back to Console, Escape, and backdrop dismissal. While it covers the Console, a privacy-minimized attention summary can report that approvals or new run activity need attention and return the user to the Console, but cannot expose or resolve those items. It is a dedicated UI/service surface and does not reuse File Notes synchronization, persistence, or editor ownership.

### 2. Direct user actions use binding authority, not agent approval

A workspace scope snapshot is an address, not a capability. Every list, read, filter, compare, reload, Git, and save operation revalidates current registry ownership, binding identity and fingerprint, archive/status state, canonical containment, link policy, target type, limits, and applicable access mode.

A read-only local-folder binding permits direct inspection. A read/write binding may permit explicit manual Save when the file and platform publisher satisfy the safe-publication contract. Manual Save does not use MCP/agent tool approval because it is an immediate, foreground user action. Agent writes continue to require the existing agent permission/approval path.

The binding mode remains the single shared maximum authority. We do not add an independent “manual editing permission” toggle. Settings copy must make the distinction between binding authority and agent approval explicit.

Revocation, archival, removal, or retargeting invalidates the open scope. Cached content and a draft may remain visible for recovery, but the modal does not silently follow either the old or new root.

### 3. Canonical-root leases separate manual and agent provenance

One app-scoped root mutation coordinator is keyed by canonical physical roots and detects ancestor/descendant overlap using platform-aware path components rather than string prefixes.

Before an agent run establishes a change-review baseline or dispatches mutating work, it atomically acquires leases for its participating writable roots. A conflict produces a visible, recoverable admission failure; no agent change-capture window begins.

Entering inspector Edit acquires a manual lease for the selected root. A running agent lease keeps viewing available but disables Edit. A manual lease blocks new overlapping agent-write admission until the edit session releases it. The pinned UI continuously names this reservation. Save and Revert keep the clean edit session and its lease active; explicit **Done editing** returns to Viewing and releases it without closing the inspector. File/binding navigation and dismissal also end the edit session after dirty-state resolution. Invalidated binding authority releases its old canonical-root lease while preserving the draft for Copy. Multi-root acquisition is canonicalized, sorted, and all-or-none. Manual and agent leases release in `finally` on every defined terminal lifecycle path; an agent lease remains held through terminal change-snapshot/review completion, including failure recording.

This guarantees that Chatbook-controlled inspector publications do not occur within an overlapping Agent Change Review baseline-to-terminal window. It does not attempt to coordinate external editors; exact-baseline conflict checks remain required.

### 4. Safe editability is capability-based and fail-closed

Normal writable UTF-8 files are editable when Chatbook can preserve their byte/line-ending policy, final-newline state, mode, required supported metadata, and safe regular-file identity through publication. Eligibility is based on content and filesystem facts rather than an extension allowlist. The exact baseline includes strong content identity, stable platform file identity, kind/link facts, parent identity, and every metadata field the publisher promises to preserve; mtime or content hash alone is insufficient, and a platform that cannot supply stable file/parent identity is read-only.

Binary or invalid UTF-8 content, mixed newlines, unsafe links/aliases, special or multiply-linked files, version-control internals, oversized content, hostile control text, unsupported metadata, or a platform publisher that cannot meet the contract are read-only or metadata-only with a plain-language reason. Files over the editable threshold through 8 MiB use revision-pinned pages of at most 100,000 decoded characters. Raw path identity is never reconstructed from display text; control, bidi, markup, and undecodable path data is rendered through a one-way safe formatter. There is no best-effort or “try anyway” write path.

### 5. Save reports publication and durability separately

Save performs final authority, containment, file-identity, and exact-baseline validation. For an eligible file it writes an exclusive same-directory temporary file, flushes content, preserves and validates supported metadata, atomically replaces the target, performs supported parent durability steps, and reopens/verifies final bytes, identity, and promised metadata.

The contract prevents silent overwrite of an external change detectable before the final pre-publication identity check. It does not claim to exclude an external race after that check.

The service returns typed outcomes:

- `not_published`: the target was not replaced and the draft remains unsaved;
- `published_durable`: publication and supported durability checks succeeded and the verified final bytes become the new baseline; or
- `published_durability_unknown`: replacement occurred but a later durability step failed or could not be confirmed.

The user can cancel Save only before publication linearizes. A cancellation that wins returns `not_published`; once publication wins, cancellation and dismissal are suppressed and the inspector remains mounted until a terminal result. The final outcome is never retried automatically. The inspector offers recovery based on any final identity it could verify. If a final read verifies the draft bytes, they become the clean baseline while the durability warning remains visible. If final bytes cannot be verified, the draft and prior baseline remain pinned and another Save is disabled until Refresh or Compare establishes current disk identity. Because replacement is known, the path is recorded as edited during the modal visit.

Graceful application quit reuses the same dirty guard. If a Save is already active, quit briefly freezes typing and waits for that single operation instead of cancelling or duplicating it. Quit proceeds only after a durable verified result leaves no dirty draft; every conflict, failure, cancellation, access change, or unresolved publication outcome keeps the inspector open for recovery. Forced process termination is outside this guarantee.

### 6. Git decoration is isolated and non-authoritative

Git status is provided by a separate read-only adapter with bounded concurrency/time/output, absolute executable resolution, `--no-replace-objects`, literal pathspecs, NUL-delimited porcelain-v2 parsing, fsmonitor/rename/maintenance/gc suppression, and an isolated non-interactive no-lazy-fetch environment with system/global config disabled. It never invokes hook-capable behavior, prompts, pagers, editors, caller-supplied Git redirects, or output-derived command arguments. Parsed raw path identity remains separate from escaped display text and every result is revalidated inside the selected binding. The adapter does not refresh or write the index, stage changes, modify configuration, or provide filesystem authority.

Git absence, timeout, malformed output, and nested-repository complexity fail locally as unavailable/truncated decoration. They do not prevent browsing or change Save eligibility.

### 7. Inspector state is ephemeral and privacy-minimized

Drafts, exact revisions, conflict snapshots, selected paths, filters, Git output, pending navigation, and the `Edited this visit` ledger remain in memory for the modal visit. They do not enter the database, sync engine, agent context, conversation, or Agent Change Review.

Persistent logs exclude file content, drafts, paths, filter strings, raw Git output, and raw filesystem exception text. Logs use opaque IDs, bounded counts, and sanitized error codes. Copy draft is an explicit user-requested clipboard transfer to shared external state.

### 8. Async work and teardown are bounded

List, read, and filter lanes each permit one active operation and at most one coalesced latest request. Each binding’s Git sub-lane follows the same rule under a small modal-wide concurrency cap. Save is single-flight. Git subprocesses receive bounded process-group termination. Stale generation tokens prevent late publication, while graceful unmount/quit joins or terminates owned work and releases leases in `finally`. Post-publication Save prevents graceful unmount until its one terminal outcome. The design does not create unbounded queues, workers, subprocesses, or duplicate modal visits.

## Consequences

### Positive

- Users can inspect a non-active workspace without losing Console context.
- One existing binding authority model covers direct inspection/editing while agent approvals retain their separate purpose.
- Manual inspector changes cannot be attributed to an overlapping Chatbook agent run.
- Unsafe files and unsupported platforms fail closed with a useful read-only experience.
- Typed publication outcomes prevent unsafe automatic retry after a successful replacement.
- Git improves review context without becoming an authority or availability dependency.
- Revision-pinned paging makes large-file inspection useful without whole-file editor risk.
- Bounded single-flight/coalesced lifecycle rules prevent duplicate work and orphan leases.
- No database schema or persisted draft state is required.

### Costs and constraints

- Agent run admission must integrate with a new app-scoped canonical-root coordinator before inspector editing ships.
- The inspector must expose and test explicit Done-editing lease release; merely saving or leaving a clean editor mounted does not release the root.
- Publication needs platform-specific implementation and real-filesystem tests; some files or platforms will be read-only.
- Exact revision and promised-metadata comparison can conservatively classify more files as conflicting or read-only.
- Operations must revalidate on every call, increasing service complexity compared with trusting an open tree.
- The UI must preserve recoverable drafts across binding conflicts and uncertain publication while avoiding persistent storage.
- The modal must implement the app quit hooks, generic Console-attention bridge, bounded worker/subprocess teardown, and hostile-text display path.
- Memory-only drafts cannot survive force kill, host crash, or operating-system termination.
- Settings documentation must explain binding maximum authority versus agent approval.

## Alternatives considered

### Reuse File Notes

Rejected. File Notes owns database synchronization and note semantics. Workspace inspection needs direct filesystem identity, bounded arbitrary-file viewing, explicit publication, and no sync/persistence ownership.

### Extend Agent Change Review into an editor

Rejected. Change Review is agent-turn evidence after or during a controlled run. Making it a general editor would mix manual and agent provenance and would not solve inspection of untouched files or non-active workspaces.

### Route manual Save through agent/MCP approvals

Rejected. The user is directly invoking a foreground operation and already selected a workspace binding authority in Settings. Agent approval semantics would misrepresent the actor and create redundant prompts.

### Add a separate manual-edit permission toggle

Rejected. A second permission axis would make read/write bindings ambiguous. The binding remains maximum authority; agent execution adds its own approval gate.

### Permit concurrent inspector and agent writes with a warning

Rejected. The existing whole-root baseline/terminal comparison can include user mid-run edits. A warning does not preserve provenance. Canonical-root mutual exclusion is required for Chatbook-controlled edits.

### Perform best-effort in-place writes

Rejected. In-place or metadata-losing writes can corrupt content, expose partial bytes, or turn an uncertain result into a destructive retry. Unsupported cases remain read-only.

### Autosave or persist drafts

Rejected for v1. Either adds recovery, privacy, schema, lifecycle, and conflict semantics beyond the inspection goal. Save remains explicit and drafts are memory-only.

### Quit immediately or cancel an active Save

Rejected. Immediate quit can lose a draft or hide an already-linearized replacement, while cancelling after publication has no honest meaning. Graceful quit reuses the dirty guard and waits for the single active Save terminal outcome.

### Treat content hash or mtime as the complete baseline

Rejected. An external replacement can preserve bytes or timestamps while changing identity, links, or metadata the publisher promises to preserve. Conflict detection uses the complete `FileRevision`.

### Queue every filesystem and Git request

Rejected. Rapid navigation/filtering can otherwise create unbounded stale work and complicated teardown. Read-like lanes keep only one active and one latest coalesced request; Save remains single-flight.

### Navigate to a separate Console screen

Rejected. A modal preserves visible Console context, supports backdrop dismissal, and matches the requirement that inspection not become a context switch.

## Follow-up

After the written design is approved, create three dependency-ordered Backlog tasks: read-only inspector, secure editing/publication with root coordination, and isolated Git decoration. Each task must link this ADR and the design specification, include its own targeted and live evidence, and follow repository Definition of Done before being marked complete.
