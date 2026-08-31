# Workspace Files Inspector — Product and Technical Design

- **Date:** 2026-08-31
- **Status:** Proposed; the conversational design is approved and this written specification awaits review.
- **Decision record:** [ADR-079](../../../backlog/decisions/079-workspace-file-inspector-direct-user-authority-and-save-publication.md)

## Summary

The Console will let a user open a near-full-screen **Workspace Files** modal for any visible named workspace without activating that workspace or changing the current Console task, session, conversation, or context. The modal provides a bounded file tree, file viewing, explicit editing and saving for safely writable text files, undo/redo, Revert, conflict handling, and isolated read-only Git decoration.

The feature is an inspector and manual editor. It is not an agent tool, a second workspace activator, a File Notes synchronization surface, or an extension of Agent Change Review.

## Problem

The Console exposes workspaces and their agent activity but does not provide a direct way to inspect the files behind a workspace. A user planning or reviewing work must leave the Console or activate a different context merely to examine files. That makes it harder to inspect files an agent touched, files it may touch next, and neighboring implementation details needed for better instructions.

## Goals

- Open files for any visible named workspace while preserving the active Console context.
- Make workspace identity and the fact that the inspector is non-activating continuously obvious.
- Support fast, bounded inspection across all current local-folder bindings of that workspace.
- Allow deliberate manual edits to ordinary, safely publishable UTF-8 text files.
- Prevent stale reads, stale writes, path escapes, and silent overwrites.
- Keep manual edits outside Agent Change Review attribution for an overlapping agent turn.
- Show useful Git state without letting Git availability or failure affect file authority.
- Remain usable in narrow terminals and dismiss through Back/Close, Escape, or backdrop click.

## Non-goals

- Creating, renaming, moving, or deleting files or directories.
- Save As, export, multiple tabs, or multiple simultaneous edit buffers.
- Autosave, filesystem watchers, persistent drafts, or a persistent visit ledger.
- Activating the inspected workspace or modifying Console context.
- Binding creation, removal, retargeting, or permission management; those remain in Settings.
- Sending file content, paths, drafts, or Git output into agent context or conversation history.
- Attributing inspector saves to an agent turn or adding them to Agent Change Review.
- Git mutations, staging, committing, index refresh, or repository maintenance.
- Editing binary, non-UTF-8, unsafe-link, special, or otherwise unsafe-to-publish files.
- Editing version-control internals such as `.git`.

## Product principles

1. **Inspection is not activation.** Opening or using the modal never changes the active workspace or Console state.
2. **Direct user action has direct user authority.** A manual Save is not an agent tool call and does not use the MCP/agent approval flow.
3. **One binding mode is the maximum authority.** Read-only bindings can be inspected; read/write bindings may permit manual Save and agent writes remain separately approval-gated.
4. **Safe files are editable; the rest explain why they are read-only.** There is no risky best-effort write path.
5. **Filesystem facts are revalidated at every operation.** A previously rendered tree row is never treated as a capability token.
6. **Publication outcomes are honest.** The UI distinguishes “not published” from “published but durability could not be confirmed.”
7. **Manual and agent provenance do not overlap.** A canonical-root edit lease prevents inspector publication during an overlapping agent change-capture window.

## User experience

### Entry points

Two entry points open the same modal:

- The active Workspace context exposes **Show Files**.
- Each named workspace in the all-workspaces list exposes a compact **Files** action with the tooltip `Show files for <workspace>`.

The crowded workspace switcher does not gain a fourth persistent button. Both entry points emit a typed `WorkspaceFilesRequested(workspace_id)` intent. Names, list positions, labels, and widget IDs are display data and are never parsed to resolve the workspace.

The default workspace and workspaces with no local-folder bindings show a disabled action with a short reason. If a stale event reaches the modal after bindings disappear, the modal opens an empty recovery state rather than switching context or selecting another workspace.

### Modal shell

The modal covers most of the Console while leaving a visible backdrop. Its header contains:

- **Back** / **Close inspector** in the upper-left area;
- `Workspace files — <inspected workspace>`; and
- a pinned identity notice: `Inspector only · Console remains <active workspace>`.

The active-workspace notice remains visible while navigating, editing, resolving conflicts, and saving. The modal records its opener and restores focus to it when possible, falling back to a stable Console control if the opener was recomposed or removed.

The modal uses the Console safe-dismiss contract by overriding `SafeModalDismissMixin._perform_safe_cancel`. Back/Close, Escape, and backdrop click therefore enter the same dirty/save-aware dismissal path.

### Wide layout

- Left pane: binding selector, path/name filter, bounded directory tree, Git and visit decorations.
- Right pane: file identity, read-only reason or editor, status, and context-sensitive actions.
- The selected binding is always explicit. Binding rows use a disambiguated label/path and show `read-only` or `read/write`, plus the agent-project marker when applicable.
- Unavailable bindings remain visible with their reason; the UI does not silently substitute another root.

Dotfiles are visible. Version-control internals are locked/excluded. Common generated caches are initially hidden or collapsed with an explicit reveal action.

### Narrow layout

The same modal becomes a staged single-pane flow:

1. roots and file tree;
2. viewer/editor for the selected file;
3. **Back to files** returns to the tree;
4. **Close inspector** remains a separate action.

Resizing preserves the inspected workspace, selected binding, selected file, draft, baseline, dirty state, undo history where the editor permits, and logical focus. It must not dismiss or recreate the modal.

### Viewer and editor

Opening a file begins in viewing mode. A normal writable file exposes **Edit**. Entering Edit acquires the canonical-root manual edit lease. If an overlapping agent run owns that root, viewing remains available but Edit is disabled with `Agent is working in this root. Try editing when the run finishes.`

Editing provides:

- explicit **Save**;
- editor-native undo and redo;
- **Revert**, which restores the exact loaded baseline and clears undo/redo;
- dirty-state indication; and
- **Copy draft**, which copies only the current draft.

Returning exactly to the baseline through Undo makes the buffer clean. Redo may make it dirty again.

### Navigation and dismissal guard

When a draft is dirty, any action that would replace or discard it—selecting another file or root, Back to files, Close, Escape, or backdrop click—shows the same guard:

- **Save and continue**;
- **Discard**; or
- **Keep editing**.

The requested navigation is stored as a typed pending intent and runs only after the guard resolves. A second navigation cannot replace the pending intent while Save is publishing.

### Saving

Save briefly freezes editor input, file/root navigation, dismissal, and competing actions. The modal remains mounted and displays `Finishing save…` once publication becomes non-cancellable. It waits for the final result rather than closing optimistically.

Save can produce these user-visible outcomes:

- **Saved:** publication and the supported durability steps succeeded; the exact final bytes are re-read and become the new clean baseline.
- **Conflict:** the disk identity differs from the loaded baseline; no publication occurred.
- **Binding changed:** workspace or binding identity/authority changed; no publication occurred and the draft remains available.
- **Save failed:** publication did not occur; the draft remains dirty.
- **Published; durability unknown:** replacement occurred but final durability could not be confirmed. The app does not retry automatically. The modal stays open with Compare/Refresh/Copy recovery actions and reports any final on-disk identity it could verify. If a final read verifies the draft bytes, they become the clean baseline while the durability warning remains visible. If final bytes cannot be verified, the app pins the draft and prior baseline, disables another Save, and requires Refresh or Compare before further publication.

The UI never implies that a reported failure guarantees unchanged disk state if replacement already occurred.

### Conflict resolution

A conflict view pins three exact identities:

- **Base** — bytes originally loaded;
- **Draft** — the current editor buffer; and
- **Disk** — bytes read for the conflicting disk identity.

Compare is read-only. **Reload from disk** performs a fresh identity validation rather than trusting the displayed snapshot, replaces the editor baseline and draft only after success, and clears undo/redo. **Keep draft** returns to the editor without changing the baseline. A later Save must pass the full conflict check again.

### Copy draft

Copy draft uses the application clipboard service and copies no surrounding metadata. Clipboard success is announced only when the platform confirms it; otherwise the UI says `Copy requested`. A clipboard failure leaves the buffer, baseline, dirty state, and disk unchanged.

### Decorations

Decorations are independent signals and must not be communicated by color alone:

- Git state, such as modified, untracked, conflicted, ignored, or repository unavailable;
- **Unsaved** editor state;
- **Conflict** state; and
- **Edited this visit**, a memory-only set of root-qualified paths added after a known successful publication.

`Edited this visit` is not an agent-attribution signal and is cleared when the modal visit ends.

## State model

The modal owns a single selected file and at most one edit buffer. Its primary states are:

- `Viewing`
- `EditingClean`
- `Unsaved`
- `SavingPrePublication`
- `SavingPublishedPendingDurability`
- `Saved` (brief feedback, then `EditingClean`)
- `Conflict`
- `BindingChanged`
- `SaveFailed`
- `PublishedDurabilityUnknown`
- `MissingOrUnsupported`

The modal also owns the selected `FileRef`, exact baseline `FileRevision`, current draft and draft revision, pending navigation, root edit lease, operation generations, and visit ledger.

Transitions that replace the buffer must resolve dirty state first. Filesystem and Git results carry an operation token containing at least the modal visit, workspace/binding fingerprint, root-qualified path, relevant baseline, and buffer revision. Results whose token no longer matches current modal state are discarded without changing the UI.

## Architecture

### Source map

New Console UI modules:

- `tldw_chatbook/UI/Console_Modules/workspace_files_modal.py`
- `tldw_chatbook/UI/Console_Modules/workspace_files_tree.py`
- `tldw_chatbook/UI/Console_Modules/workspace_file_editor.py`

Console integration remains thin:

- `tldw_chatbook/UI/Console_Modules/wiring.py` routes typed entry intents.
- Existing workspace tray/context widgets emit `WorkspaceFilesRequested`.
- `tldw_chatbook/UI/Screens/chat_screen.py` installs/dismisses the modal and supplies the active workspace identity; it does not perform filesystem work.

New workspace services:

- `tldw_chatbook/Workspaces/file_inspector_service.py`
- `tldw_chatbook/Workspaces/file_git_status_reader.py`
- `tldw_chatbook/Workspaces/root_mutation_coordinator.py`

`file_inspector_service.py` may define frozen transport types such as `WorkspaceInspection`, `BindingScope`, `FileRef`, `FileRevision`, `DirectorySnapshot`, `FileSnapshot`, `SaveCommand`, and `SaveOutcome`. It has no Textual, clipboard, notification, worker, or Console-controller dependency. Its registry dependency is an injected protocol.

Leaf widgets emit typed intents upward. The modal is the sole UI owner that calls the service. The service reports filesystem facts and typed outcomes; it does not decide whether an asynchronous result is still current.

### Worker lanes and cancellation

Directory listing, file reading, filter walking, Git querying, and saving run in separate worker lanes. Operations expected to exceed 100 ms never block the Textual event loop.

List/read/filter/Git work is logically cancellable. A thread-level filesystem operation may finish after cancellation, but its stale token prevents UI publication. Save is cancellable only before its publication linearization point. After replacement begins, the modal remains mounted and awaits a terminal publication outcome.

## Authority and scope

Opening the modal captures the inspected workspace ID/name plus the binding IDs and fingerprints visible at that moment. This is an address snapshot, not a durable capability.

Every list, read, filter, compare, reload, Git, and save operation revalidates:

- the workspace still exists, is unarchived, and owns the binding;
- the binding is still a supported local-folder binding with the same identity and target fingerprint;
- the operation is contained by the canonical binding root;
- path traversal and link policy are satisfied;
- the binding access mode permits the operation;
- the target is the expected regular file or directory; and
- size, encoding, newline, metadata, and platform publisher requirements are satisfied.

A new, removed, revoked, archived, or retargeted binding never silently retargets an existing tree row or file buffer. The modal preserves cached content and any draft, disables new filesystem/Git operations and Save as appropriate, and asks the user to reopen the workspace scope. Copy draft remains available.

Read-only bindings permit direct inspection. Read/write bindings permit explicit manual Save only when the selected file passes the safety policy and the root edit lease is held. Agent writes continue to require their existing tool permission/approval path.

Settings copy must state that a binding’s mode is the shared maximum authority for direct user actions and agents; agent approval is an additional gate, not a second binding permission model.

## Canonical-root edit lease

The app owns one `RootMutationCoordinator`, keyed by canonical physical roots and aware of overlapping ancestor/descendant roots. It coordinates only Chatbook-controlled inspector editing and agent run admission; external editors remain outside this mechanism and are covered by baseline conflict detection.

### Agent admission

Before an agent run establishes its baseline or dispatches a mutating tool, it requests leases for all writable bound roots that participate in change capture. Multiple roots are canonicalized, sorted, and acquired atomically as a set to avoid partial admission and deadlock. A conflicting manual lease causes a visible, recoverable admission failure; the run does not start a change-capture window and does not silently wait forever. Leases are released after the terminal change snapshot and review material are complete.

### Inspector editing

Entering Edit requests a manual lease for the selected canonical root. A conflicting agent lease leaves the file viewable but not editable. While held, the manual lease blocks admission of a new overlapping agent-write run. The lease is released when the edit session ends or the modal closes. Switching roots first resolves any dirty draft, releases the old lease, and acquires the new root before enabling Edit.

If a binding is retargeted while leased, Save fails revalidation and the old canonical-root lease is released when the edit session exits. A rendered binding label never changes the lease key.

This coordination guarantees that a publication made through Workspace Files does not occur inside Agent Change Review’s baseline-to-terminal snapshot window. It does not claim to attribute or block changes from external programs.

## File viewing and editability policy

The user-facing rule is:

> Normal writable UTF-8 files can be edited. If Chatbook cannot save a file safely, it remains read-only and explains why.

Extension allowlists are not used. Eligibility is determined from bytes, file identity, links, metadata, binding mode, and platform publication capability.

### Editable

A file is editable when all of the following are true:

- it is a regular file reached without following a symlink, reparse point, or unsafe alias;
- it has a single safe link identity under the canonical root;
- it is UTF-8, optionally with a UTF-8 BOM;
- it uses uniform LF or CRLF newlines;
- its decoded content is at most 200,000 characters;
- its required mode and platform metadata can be preserved;
- the binding is read/write; and
- the active platform publisher can meet the save contract.

Save preserves BOM presence, newline convention, final-newline state, file mode, and required supported platform metadata.

### View-only or metadata-only

- More than 200,000 characters through 8 MiB: bounded read-only excerpt of at most 100,000 characters, with truncation disclosed.
- More than 8 MiB: metadata only.
- Invalid UTF-8, binary content, mixed newline convention, symlink/reparse targets, special files, multiply-linked files, unsafe metadata, version-control internals, or unsupported publication semantics: read-only or metadata-only with a specific reason.

There is no “try anyway” write option.

### Directory listing and filter bounds

- A directory renders 200 entries per page and scans at most 10,000 immediate entries.
- Opening the modal does not recursively scan all roots.
- Filter matches root-relative path/name only and starts a cancellable bounded walk.
- A filter request visits at most 50,000 entries across the inspected workspace and returns at most 500 results.
- Version-control internals, hidden generated caches, and symlinked directories are not traversed.
- Truncation and exclusions are visible, with an option to narrow the query or explicitly reveal supported hidden cache entries.

## Save publication contract

Save accepts a `SaveCommand` containing the current workspace/binding fingerprint, root-qualified path, exact base revision, draft bytes/encoding policy, and operation token.

The service performs final authority and containment validation, then uses descriptor-based/no-follow access where the platform supports it to confirm the target remains the expected regular file. It compares the exact current disk identity with the base revision immediately before publication.

For an eligible file, the publisher:

1. creates an exclusive temporary file in the target directory;
2. writes the complete encoded draft;
3. flushes file data as supported;
4. applies and validates preserved metadata;
5. performs an atomic same-directory replacement;
6. flushes parent-directory metadata where the platform supports and requires it; and
7. reopens and hashes the final target to report its exact final revision.

The guarantee is deliberately narrow: no external change detectable before the final pre-publication identity check is silently overwritten. An external process can still race after that check; the app does not claim a cross-process lock it cannot enforce.

Typed terminal outcomes are:

- `not_published`: the baseline does not advance; the draft remains unsaved.
- `published_durable`: the final verified revision becomes the clean baseline and the root-qualified path enters the visit ledger.
- `published_durability_unknown`: replacement occurred but a later durability step failed or could not be confirmed. The operation is never automatically retried. A verified final revision equal to the draft becomes the clean baseline while the warning remains; without verification, the draft and prior baseline remain pinned and another Save is disabled until recovery. Because replacement is known, the root-qualified path enters the visit ledger.

Temporary artifacts are cleaned up only when their identity is known and cleanup cannot affect the target. Error details are sanitized before reaching logs or notifications.

## Isolated Git decoration

`file_git_status_reader.py` is a read-only, non-authoritative adapter. It may invoke an installed Git executable with a porcelain-v2, NUL-delimited status mode. It:

- never stages, refreshes, writes the index, or mutates repository configuration;
- uses `GIT_TERMINAL_PROMPT=0` and `GIT_OPTIONAL_LOCKS=0` with a scrubbed environment;
- has a 10-second timeout and 2 MiB combined-output cap;
- queries per binding and filters results to that binding;
- discovers/query nested repositories lazily when their directory is expanded; and
- refreshes on modal open, explicit Refresh, and known successful Save.

Git failure, absence, malformed output, timeout, or truncation produces a local unavailable/truncated decoration. It never grants or revokes file access, prevents tree browsing, or changes Save decisions.

## Privacy and persistence

Workspace/root labels and root-relative paths may be rendered locally. Canonical roots may be displayed where needed to disambiguate authority. The following are never written to persistent application logs, agent context, conversations, review metadata, databases, or sync state:

- file content or drafts;
- filter strings;
- root-relative or canonical paths;
- raw Git output; and
- raw filesystem exception text.

Operational logs use sanitized error codes, opaque workspace/binding identifiers, and bounded counts. The selected file, draft, undo state, pending navigation, conflict snapshots, Git cache, and `Edited this visit` set are memory-only for the modal visit.

Clipboard use is an explicit user-requested transfer to shared external state and is described as such by the Copy action.

## Accessibility and input

- Every decoration has text/icon semantics in addition to color.
- The focus order follows header, roots/filter/tree, file identity, editor, then actions.
- The backdrop is not focusable while the modal is open.
- Escape uses safe dismissal and is suppressed while publication is non-cancellable.
- The screen does not bind terminal-convention control keys or shadow Console globals. Editor-native undo/redo is exposed through the editor’s supported interaction and visible actions rather than new screen bindings.
- Status changes are announced concisely without moving focus unexpectedly.

## Verification strategy

Verification uses real temporary filesystems and repositories for authority/publication behavior, deterministic barriers for races, and production-shaped Textual tests for modal behavior. Unit tests alone are insufficient evidence for publication, compositor, or live Console-context preservation.

### Behavior matrix

| Precondition | Action or interleaving | Service outcome | Required visible result | Prohibited side effect / evidence |
|---|---|---|---|---|
| Workspace B is not active | Open B from either entry | Inspection for B | Header names B; notice says Console remains A | Active workspace, task, conversation, composer, and approval state fingerprints remain unchanged |
| Read A is slow, then user selects B | B read finishes before A | B snapshot accepted; A token stale | Viewer shows only B | A bytes never flash or replace B |
| Binding is read-only | Select file and attempt Edit/Save | Editability denied | File is viewable with read-only reason | No temp file, write, approval prompt, or permission mutation |
| Binding is revoked or retargeted while open | Read or Save | `binding_changed` | Cached content/draft retained; reopen guidance | No access to old or new target through stale row |
| Agent owns overlapping root | Enter Edit | Lease denied | Viewing works; Edit explains agent conflict | No manual publication |
| Inspector owns overlapping root | Start agent-write run | Admission conflict | Recoverable run-start message | No agent baseline or mutating dispatch begins |
| External editor changes disk after load | Save draft | `conflict` / `not_published` | Base/Draft/Disk conflict view | External bytes are not silently overwritten |
| Replace succeeds, directory flush fails | Finish Save | `published_durability_unknown` | Modal stays open with warning and recovery | No automatic retry; UI does not claim disk unchanged |
| Clipboard adapter fails | Copy draft | Clipboard failure | Error; draft remains available | No disk, persistence, or dirty-state change |
| Git missing, malformed, or times out | Open/Refresh | Decoration unavailable | Tree and editor remain functional | Authority and Save eligibility unchanged |
| Huge directory/filter corpus | Page or filter | Bounded/truncated result | Counts and truncation disclosed | Event-loop heartbeat remains responsive; caps are not exceeded |
| Unique secret in path/content/draft/filter/error | Exercise read/edit/error paths | Sanitized operational result | Secret may appear only in intended local view | Secret absent from captured logs, notifications, conversation, DB, and review metadata |
| Dirty narrow modal is resized | Wide → narrow → wide | UI-only transition | Draft, baseline, pending guard, and logical focus preserved | Modal is not dismissed or recreated |
| Workspace is archived while open | Read/Git/Save | Scope invalid | Cached view/draft and Copy remain; operations disabled | No new filesystem or Git access |
| File is binary, unsafe-linked, mixed-newline, or publisher-unsupported | Open file | View/metadata-only | Specific plain-language reason | No “try anyway” or best-effort write |

### Test layers

1. **Pure service tests**
   - containment, canonical binding identity, link and special-file rejection;
   - encoding/newline/size classification;
   - directory/filter bounds and truncation;
   - operation token and typed outcome construction.

2. **Publication and race tests on real temporary filesystems**
   - exact baseline conflict, retarget/revoke races, and external modification barriers;
   - atomic replacement, metadata preservation, final re-read/hash;
   - injected pre-publication and post-replacement failures;
   - Linux, macOS, and Windows-specific publisher behavior where CI supports it;
   - a platform incapable of satisfying the contract produces read-only classification.

3. **Root coordinator tests**
   - ancestor/descendant overlap;
   - atomic multi-root agent admission and deterministic ordering;
   - manual-versus-agent exclusion, release on all terminal paths, and no baseline on denied admission.

4. **Git adapter tests with real repositories**
   - tracked/untracked/conflicted/ignored parsing, rename paths, nested repositories, timeout/output caps;
   - index and repository fingerprints prove no mutation.

5. **Production-shaped Textual tests**
   - mount `TldwCli` using its exact `CSS_PATH` stack;
   - exercise both entry points, safe dismiss paths, dirty guard, save freeze, stale-result suppression, focus restoration, and narrow transitions;
   - re-query widgets after recomposition;
   - inspect compositor frames and geometric containment, not only widget existence.

6. **Live scratch verification**
   - launch with an isolated `TLDW_CONFIG_PATH` and temporary roots;
   - open a non-active workspace through both entry points;
   - view, edit, Save, conflict, Copy, dismiss, and resize using actual UI input;
   - compare before/after fingerprints for unrelated Console state and for intended file publication only.

Privacy tests seed unique secrets and assert their absence from every captured persistence/logging channel. Test assertions must inspect both positive outcomes and prohibited side effects.

Repository policy requires targeted verification first. A full suite is run only after asking the user, while a Backlog task cannot be marked Done until its Definition of Done is satisfied.

## Delivery slices

The overall v1 consists of three dependency-ordered, independently reviewable slices. Backlog task files are created only after this written design is approved.

### Slice 1 — Read-only inspector

- Both typed entry points and non-activating modal shell.
- Binding/root presentation, bounded tree, safe file viewer, filter, responsive layout, focus/dismiss behavior.
- Revalidation, stale-result suppression, privacy controls, and read-only/live evidence.
- Delivers standalone inspection value without editor or Git assumptions.

### Slice 2 — Secure editing and publication

- Single buffer, Edit/Save/undo/redo/Revert, dirty guard, Copy draft, and conflict view.
- Canonical-root mutation coordinator integrated with agent admission.
- Safe file classifier and platform publisher with typed publication outcomes.
- Race, platform, provenance, privacy, and live Save verification.

### Slice 3 — Isolated Git decoration

- Read-only Git adapter, bounded refresh, nested-repository behavior, accessible tree/status decoration.
- Failure isolation, no-mutation evidence, and integrated live verification.

## Acceptance criteria

### Overall v1

- [ ] Either entry point opens the selected named workspace without changing any Console context.
- [ ] Every visible local-folder binding is explicitly represented with identity, access mode, and availability.
- [ ] Tree, viewer, filter, and Git work are bounded, cancellable where safe, and stale-result resistant.
- [ ] Ordinary writable UTF-8 files can be deliberately edited and atomically published; unsupported files are read-only with a reason.
- [ ] Dirty navigation/dismissal, conflicts, binding changes, and uncertain publication preserve recoverable user work.
- [ ] Inspector publication and overlapping agent change-capture windows are mutually excluded by canonical root.
- [ ] Git decoration is isolated, read-only, accessible, and non-authoritative.
- [ ] No file data or sensitive path/filter/error material enters persistence, logs, agent context, conversation, or Agent Change Review.
- [ ] Wide and narrow layouts meet the same safety model and preserve modal state across resize.
- [ ] Targeted automated and live evidence covers successful behavior and prohibited side effects.

### Slice completion gates

- [ ] Slice 1 can ship as a useful read-only inspector with no hidden editing hooks.
- [ ] Slice 2 does not begin publication until the root coordinator is integrated with agent admission.
- [ ] Slice 3 cannot affect authority, editability, or Save behavior when Git is absent or failing.

## ADR check

- **ADR required:** yes
- **ADR path:** `backlog/decisions/079-workspace-file-inspector-direct-user-authority-and-save-publication.md`
**Reason:** The feature establishes long-lived direct-user filesystem authority, a cross-module root-mutation/provenance boundary, platform publication guarantees, and a new Console application structure. ADR-079 extends the existing workspace binding authority model and relates it to Console project-instruction and Agent Change Review boundaries.

No schema migration or persistent state is introduced by this design.
