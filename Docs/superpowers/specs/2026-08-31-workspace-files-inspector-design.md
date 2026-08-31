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
- Remain usable in narrow terminals and dismiss through Back to Console, Escape, or backdrop click.

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

- The active `ConsoleWorkspaceContext` card exposes **Show Files** on its own compact action row immediately after the existing **RAG Scope** row. It is not added to the width-sensitive **Switch** / **New** row.
- Each named workspace header in the grouped all-workspaces conversation browser exposes a permanently rendered, seven-cell **Files** control, including its padding, immediately before the existing three-cell collapse toggle. The flexible workspace label truncates before either fixed control does.

The separate workspace switcher remains unchanged and does not gain a fourth persistent button. The two entry actions are text-labeled, keyboard reachable, present at rest, and never hover-only. In the active card, **Show Files** follows **RAG Scope** in focus order. In a grouped-browser header, the order is workspace header, **Files**, then collapse toggle. Both entry points emit a typed `WorkspaceFilesRequested(workspace_id)` intent. Names, list positions, labels, and widget IDs are display data and are never parsed to resolve the workspace.

The default workspace and workspaces with no local-folder bindings keep the action visible as a focusable, pressable-but-blocked control rather than an unfocusable disabled button. Its tooltip and activation response both say `No local folders are attached. Add one in Settings.` A stale event that reaches the modal after bindings disappear opens the same empty recovery state rather than switching context or selecting another workspace.

### Modal shell

The modal covers most of the Console while leaving a visible backdrop whenever the terminal has enough room. Its header contains:

- **← Back to Console** in the upper-left area; this is the always-visible dismissal action;
- `Workspace files — <inspected workspace>`; and
- a pinned identity notice: `Inspector only · Console remains <active workspace>`.

Directly below the identity notice, a pinned contract row names the current mode, selected folder access, draft/save state, and root-reservation state. Examples are `Viewing · read/write`, `Unsaved · local draft only`, and `Editing <folder> · overlapping agent writes paused`. It shows **Done editing** at the far edge whenever a manual edit lease is held. The active-workspace notice and contract row remain visible while navigating, editing, resolving conflicts, and saving.

The modal records its opener and restores focus to it when possible, falling back to the Console composer if the opener was recomposed or removed.

The modal uses the Console safe-dismiss contract by overriding `SafeModalDismissMixin._perform_safe_cancel`. Back to Console, Escape, and backdrop click therefore enter the same dirty/save-aware dismissal path.

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
4. **← Back to Console** remains a separate dismissal action.

Resizing preserves the inspected workspace, selected binding, selected file, draft, baseline, dirty state, undo history, and logical focus. It must not dismiss or recreate the modal.

### Responsive geometry

The implementation uses independent compact-width and short-height states rather than one vague “narrow” flag:

- **Wide:** at least 112 columns and 30 rows. The modal uses 96% of the terminal up to 152 columns by 52 rows, keeps a backdrop, and displays two panes. The tree pane is at least 34 cells wide; the editor pane is at least 64 cells wide; remaining width goes to the editor.
- **Compact:** fewer than 112 columns. The modal uses the staged single-pane flow. At fewer than 100 columns it becomes truly full-screen so a decorative backdrop cannot consume editor cells; backdrop dismissal is consequently unavailable, while Back to Console and Escape remain.
- **Short:** fewer than 30 rows. Header copy compresses to title, identity, and contract rows; secondary help moves behind **Help**; the body scrolls beneath a pinned one-row action region, and a one-row fold indicator announces hidden body content.
- **Supported minimum:** 80 columns by 24 rows. Below that, the modal refuses to open and reports `Workspace Files needs at least 80 × 24 terminal cells.` without changing Console context.

The initial verification matrix is:

| Terminal | Presentation | Required behavior |
|---|---|---|
| 80 × 24 | full-screen, compact + short | one pane, one-line paths, pinned contract/action rows, fold indicator |
| 100 × 30 | near-full-screen, compact | staged tree/viewer, normal header, no clipped controls |
| 120 × 40 | wide | two panes at their minimums, visible backdrop, pinned status/actions |
| 160 × 50 | wide | tree remains bounded; additional width belongs to the editor |

Paths use middle ellipsis while reserving cells for status marks. Full root-relative and canonical paths are available in the selected-file identity region. At each layout transition, logical focus is remapped to the equivalent control: selected tree row, editor, current recovery action, or contract-row action. Resize during dirty guard, Conflict, or either Saving state preserves that state and never changes the focused logical action.

### Viewer and editor

Opening a file begins in viewing mode. A normal writable file exposes **Edit**. Entering Edit acquires the canonical-root manual edit lease and changes the contract row to `Editing <folder> · overlapping agent writes paused`. If an overlapping agent run owns that root, viewing remains available and the blocked Edit action has adjacent copy: `An agent is working in this folder. Editing will be available when that run finishes.` The reason is never tooltip-only.

Editing provides:

- a primary **Save** action;
- visible **Undo** and **Redo** actions plus editor-native undo/redo behavior;
- **More…**, containing **Revert** and **Copy draft**; and
- **Done editing** in the pinned contract row.

**Revert** asks once `Discard this draft and return to the loaded file?`; confirming restores the exact loaded baseline and clears undo/redo. **Copy draft** copies only the current draft. The ordinary editor action row is `[Save] [Undo] [Redo] [More…]`; Conflict and uncertain-publication states replace it rather than append more peer actions.

In `EditingClean`, Save is unavailable with adjacent text `No changes to save`; the contract row remains the primary carrier of the clean state. Disabled styling alone never communicates this restriction.

Returning exactly to the baseline through Undo makes the buffer clean. Redo may make it dirty again.

Save and Revert leave the editor in `EditingClean` and retain the lease because the user is still explicitly editing. **Done editing** is the normal release path. When clean, it returns to Viewing, releases the lease, and restores focus to the selected tree row in wide mode or the viewer’s **Edit** action in compact mode. When dirty, it enters the same guard as navigation. Selecting another file, selecting another binding, **Back to files**, or dismissing the modal ends the edit session after any dirty state is resolved and releases the lease before the destination is entered.

### Navigation and dismissal guard

When a draft is dirty, any action that would replace or discard it—**Done editing**, selecting another file or folder binding, Back to files, Back to Console, Escape, or backdrop click—replaces the ordinary action bar with one inline guard owned by the existing modal:

- **Save and continue**;
- **Discard**; or
- **Keep editing**.

The guard names the pending destination, for example `Save before opening src/app.py?` or `Save before closing Workspace Files?`. **Keep editing** receives initial focus. Escape or backdrop click while the guard is visible means Keep editing: it clears the pending intent and returns focus to the editor. **Discard** is the guard’s single explicit confirmation and does not open a second modal.

The requested navigation is stored as a typed pending intent and runs only after the guard resolves. A second navigation cannot replace the pending intent while Save is publishing. The inline guard never mounts a nested modal, so `SafeModalDismissMixin` continues to have one dismissal owner.

### Saving

Save briefly freezes editor input, file/root navigation, dismissal, and competing actions. During `SavingPrePublication`, the action bar contains a visible **Cancel save** action and status `Preparing save…`. Once pressed, the action disables and status becomes `Cancelling save…` until the single race outcome is known. A cancellation acknowledged before the publication linearization point returns to `Unsaved`, preserves the draft and baseline, clears any pending navigation, and restores editor focus. If publication wins the race, **Cancel save** disappears and the non-interactive status becomes `Finishing save…`; cancellation, dismissal, and navigation are then suppressed. The modal remains mounted and waits for the terminal result rather than closing optimistically. A Conflict, folder-access change, Save failure, or uncertain-publication outcome also clears pending navigation and requires the user to resolve the visible state before navigating again.

Save can produce these user-visible outcomes:

- **Saved:** publication and the supported durability steps succeeded; the exact final bytes are re-read and become the new clean baseline. `Saved · disk verified` remains visible for at least 1.5 seconds or until the user’s next action and is announced without moving focus.
- **Conflict:** the disk identity differs from the loaded baseline; no publication occurred.
- **Folder access changed:** workspace or folder identity/authority changed; no publication occurred. Copy says `Folder access changed — this draft is safe here, but Chatbook will not write until you reopen Workspace Files.` The draft remains available, and the now-invalid manual lease is released.
- **Save failed:** publication did not occur; the draft remains dirty.
- **Saved, confirmation incomplete:** replacement occurred but final durability could not be confirmed. Copy leads with `The file was replaced, but Chatbook could not confirm that it was safely committed to disk. Your draft is still available. Refresh is the safest next step.` The app does not retry automatically. The modal stays open with primary **Refresh**, then **Compare** and **More… → Copy draft**, and reports any final on-disk identity it could verify. If a final read verifies the draft bytes, they become the clean baseline while the warning remains visible. If final bytes cannot be verified, the app pins the draft and prior baseline, disables another Save, and requires Refresh or Compare before further publication.

The UI never implies that a reported failure guarantees unchanged disk state if replacement already occurred.

### Plain-language status copy

`Binding`, `fingerprint`, `canonical root`, `publication`, and `durability` are internal design/service terms. The user-facing modal uses folder, file, draft, access, and Save language:

| Internal state | Required leading copy | Primary recovery |
|---|---|---|
| No binding | `No local folders are attached. Add one in Settings.` | **Open Settings** |
| Agent lease conflict | `An agent is working in this folder. Editing will be available when that run finishes.` | continue viewing |
| Manual lease blocks agent | `Workspace Files is editing <folder>. Choose Done editing before starting this agent.` | **Return to inspector** |
| Binding changed | `Folder access changed — this draft is safe here, but Chatbook will not write until you reopen Workspace Files.` | **Copy draft**, then **Reopen** |
| Unsupported edit | `Chatbook can show this file but cannot save it safely: <specific reason>.` | continue viewing |
| Published durability unknown | `The file was replaced, but Chatbook could not confirm that it was safely committed to disk. Your draft is still available.` | **Refresh** |
| Git unavailable | `Git status is unavailable. File access and Save are unaffected.` | optional **Refresh Git** |
| Filter truncated | `Showing 500 results; narrow the filter.` | focus Filter |

Specific reasons and paths are rendered locally and safely truncated; raw exception text and service enum names never appear in UI copy.

### Conflict resolution

A conflict view pins three exact identities:

- **Base** — bytes originally loaded;
- **Draft** — the current editor buffer; and
- **Disk** — bytes read for the conflicting disk identity.

Compare is read-only. In wide mode it shows Base/Draft/Disk columns only when each meets its minimum width; otherwise it uses the compact comparison flow. Compact comparison presents a Base/Draft/Disk selector above one read-only viewport, so the three identities have a deterministic linear reading order. **Reload from disk** performs a fresh identity validation rather than trusting the displayed snapshot, replaces the editor baseline and draft only after success, and clears undo/redo. **Keep draft** returns to the editor without changing the baseline. A later Save must pass the full conflict check again. **Compare** receives initial focus because it is non-destructive.

### Copy draft

Copy draft uses the application clipboard service and copies no surrounding metadata. Clipboard success is announced only when the platform confirms it; otherwise the UI says `Copy requested`. A clipboard failure leaves the buffer, baseline, dirty state, and disk unchanged.

### Decorations

Decorations are independent signals and must not be communicated by color alone:

- Git state, such as modified, untracked, conflicted, ignored, or repository unavailable;
- **Unsaved** editor state;
- **Conflict** state; and
- **Edited this visit**, a memory-only set of root-qualified paths added after a known successful publication.

`Edited this visit` is not an agent-attribution signal and is cleared when the modal visit ends.

Each tree row reserves cells in this order: indentation/caret, type glyph, middle-elided name, one primary state mark, and at most one secondary mark. State precedence is **Conflict > Unsaved > Git > Edited this visit**. Conflict uses `[!]`, Unsaved uses `[*]`, common Git states use `[M]`, `[U]`, `[A]`, `[D]`, or `[R]`, and Edited this visit uses `[V]`. When more states exist than fit, the selected-file identity region spells all of them out. Root-level Git unavailable/truncated state appears on the binding header rather than impersonating a file state.

Selection and keyboard focus use structural row/background treatment plus a leading focus indicator; neither relies on filename color. A visible **Help** action opens a compact legend containing the full text for every mark. The filename truncates before reserved state cells do.

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

### Loading, empty, and changing-data states

The modal shell and binding identities mount immediately. Each asynchronous region owns an explicit local state instead of blank space:

- tree: `Loading folder…`, `This folder is empty.`, `Folder unavailable — <safe reason>.`, or page position;
- viewer: `Reading file…`, the bounded content, or `File no longer exists. Refresh the folder.`;
- filter: the defined progress/partial/zero/excluded/truncated states; and
- Git: `Checking Git…` or a fail-soft unavailable/truncated label that never blocks file work.

Directory pages carry a directory identity. If the directory changes between pages, the modal discards the mixed snapshot and offers **Refresh folder** rather than merging incompatible pages. File disappearance, permission loss, and type change move only that region to `MissingOrUnsupported`, preserve any existing draft, and expose the relevant safe recovery. Operation tokens prevent a late tree, file, filter, or Git result from reviving an earlier selection.

## Architecture

### Source map

New Console UI modules:

- `tldw_chatbook/UI/Console_Modules/workspace_files_modal.py`
- `tldw_chatbook/UI/Console_Modules/workspace_files_tree.py`
- `tldw_chatbook/UI/Console_Modules/workspace_file_editor.py`

Console integration remains thin:

- `tldw_chatbook/UI/Console_Modules/wiring.py` routes typed entry intents.
- `tldw_chatbook/Widgets/Console/console_workspace_context.py` owns the exact active-card and grouped-browser entry placement and emits `WorkspaceFilesRequested`.
- `tldw_chatbook/UI/Screens/chat_screen.py` installs/dismisses the modal and supplies the active workspace identity; it does not perform filesystem work.

New workspace services:

- `tldw_chatbook/Workspaces/file_inspector_service.py`
- `tldw_chatbook/Workspaces/file_git_status_reader.py`
- `tldw_chatbook/Workspaces/root_mutation_coordinator.py`

`file_inspector_service.py` may define frozen transport types such as `WorkspaceInspection`, `BindingScope`, `FileRef`, `FileRevision`, `DirectorySnapshot`, `FileSnapshot`, `SaveCommand`, and `SaveOutcome`. It has no Textual, clipboard, notification, worker, or Console-controller dependency. Its registry dependency is an injected protocol.

Leaf widgets emit typed intents upward. The modal is the sole UI owner that calls the service. The service reports filesystem facts and typed outcomes; it does not decide whether an asynchronous result is still current.

### Worker lanes and cancellation

Directory listing, file reading, filter walking, Git querying, and saving run in separate worker lanes. Operations expected to exceed 100 ms never block the Textual event loop.

List/read/filter/Git work is logically cancellable. A thread-level filesystem operation may finish after cancellation, but its stale token prevents UI publication. Save is cancellable only before its publication linearization point through the visible **Cancel save** action. Cancellation and publication race through one typed terminal outcome: an acknowledged cancellation returns `not_published`; once publication wins, the operation becomes non-cancellable. After replacement begins, the modal remains mounted and awaits a terminal publication outcome.

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

Entering Edit requests a manual lease for the selected canonical root. A conflicting agent lease leaves the file viewable but not editable. While held, the manual lease blocks admission of a new overlapping agent-write run, and the pinned contract row says so. Save and Revert retain the lease in `EditingClean`; the explicit **Done editing** action releases it and returns to Viewing. File navigation, binding navigation, Back to files, and modal dismissal also end the edit session after dirty state is resolved, release the lease, and open the destination in Viewing. The inspector never carries an edit lease into another file implicitly.

Conflict and `SaveFailed` retain the lease while the recoverable draft remains in the edit session. `BindingChanged` releases the invalid lease immediately while preserving the draft for Copy. `PublishedDurabilityUnknown` retains the lease until Refresh/Compare resolves current disk identity or the user explicitly chooses Done editing and confirms that later agent activity may change the folder. If a binding is retargeted while leased, Save fails revalidation and the coordinator releases the old canonical-root key; a rendered binding label never changes the lease key.

When agent admission is denied by a manual lease, the recovery message says `Workspace Files is editing <folder>. Choose Done editing before starting this agent.` and focuses or reopens the existing inspector when the user invokes its recovery action.

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
- The binding selector is an explicit authority boundary. With one binding, it is selected automatically. With multiple bindings, the first readable binding in stable registry order is selected and focus starts on the selector so the choice is immediately visible. If none is readable, the first listed binding opens as an unavailable recovery state. No selection silently falls forward if that binding later becomes unavailable.
- Filter operates only inside the selected binding, matches a case-insensitive literal substring of the root-relative path/name, and starts a cancellable bounded walk after a 150 ms typing debounce; Enter starts immediately, and an empty field clears filtering. Special characters are literal rather than glob syntax.
- A filter request visits at most 50,000 entries inside the selected binding and returns at most 500 results. Results are therefore unambiguous without cross-root path qualification.
- Version-control internals, hidden generated caches, and symlinked directories are not traversed.
- Directories sort before files; each group sorts by Unicode casefold, then the exact name for deterministic ties.
- Filter state is explicit: `idle`, `searching`, `partial`, `complete`, `truncated`, `cancelled`, or `failed`. Searching shows visited/result counts plus **Cancel** and **Clear**. Partial results may be opened because each open revalidates authority and identity. Cancel retains the labeled partial results; Clear restores the pre-filter tree expansion and selection when those identities remain valid. Truncation says `Showing 500 results; narrow the filter.` Zero results distinguish no matches from only-excluded matches.
- **Reveal generated caches** is a modal-visit setting. Changing it invalidates the active filter generation and reruns only after explicit confirmation; it never silently changes the visible result set.
- Selecting another binding is a typed navigation intent. It resolves dirty state, ends any edit session, releases the old lease, clears the filter, and then lists only the newly selected binding. Unavailable bindings remain selectable as recovery states but never cause fallback to another root.

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

- Every decoration has a visible text/glyph meaning in addition to color, and the selected-file region expands compact marks into full words.
- The backdrop is not focusable while the modal is open. Pressable-but-blocked and unavailable actions expose their reason through focused adjacent text and activation feedback, never only through hover or dimming.
- The screen does not bind terminal-convention control keys or shadow Console globals. Editor-native undo/redo remains owned by the editor, while visible Undo/Redo buttons make the actions discoverable.
- Loading, tree expansion/collapse, page position, filter progress/truncation, Save transitions, conflicts, and errors are announced concisely without moving focus unexpectedly.

The input/focus contract is:

| State/location | Initial or retained focus | Keys/actions | Resulting focus |
|---|---|---|---|
| Modal open, one binding | selected tree root | Up/Down select; Left collapse/parent; Right expand; Enter open; `f` focuses Filter when no text editor has focus | wide keeps tree focus after open; compact moves to viewer |
| Modal open, multiple bindings | binding selector | Up/Down choose; Enter list selected binding | selected root in tree |
| Filter | filter field | typing updates bounded filter; Escape clears an active query first; Tab reaches Cancel/Clear/results | first result only on explicit navigation |
| Wide tree file open | selected tree row | continued arrows preview other files; Tab enters viewer/actions | exact selected row is remembered |
| Compact viewer | viewer or first safe action | **Back to files** returns to tree; **Back to Console** dismisses | exact selected tree row |
| Viewing editable file | **Edit** when entered from compact viewer; tree remains focused in wide preview | Edit requests lease | editor after lease success; blocked reason after denial |
| Editing | editor | text input stays with editor; Tab reaches Save/Undo/Redo/More and contract-row Done editing | editor after Save/Undo/Redo |
| Dirty guard | **Keep editing** | Escape/backdrop = Keep editing; Save/Discard run pending intent | editor when kept; destination after resolution |
| Saving pre-publication | **Cancel save** | Cancel requests typed cancellation | editor if `not_published`; status if publication won |
| Saving after publication | non-interactive `Finishing save…` status | input, Escape, and dismissal suppressed | prior logical focus restored after outcome |
| Conflict | **Compare** | Base/Draft/Disk selector in compact; Reload or Keep draft | editor after Keep draft; selected file identity after Reload |
| Saved | editor | no forced focus movement | current logical focus |
| Done editing | contract-row action | dirty state invokes guard; clean state releases lease | selected tree row in wide; Edit in compact viewer |

Escape has ordered meaning: during non-cancellable publication it does nothing; inside the dirty guard it means Keep editing; while focus is in a non-empty filter it clears that filter; otherwise it requests normal safe dismissal. **Back to files** exists only in the compact viewer/editor; **← Back to Console** is always the dismissal action. The exact modal opener is restored through `SafeModalDismissMixin`, including its click-chain shield and widget-ID fallback after recomposition.

## Verification strategy

Verification uses real temporary filesystems and repositories for authority/publication behavior, deterministic barriers for races, and production-shaped Textual tests for modal behavior. Unit tests alone are insufficient evidence for publication, compositor, or live Console-context preservation.

### Behavior matrix

| Precondition | Action or interleaving | Service outcome | Required visible result | Prohibited side effect / evidence |
|---|---|---|---|---|
| Workspace B is not active | Open B from either entry | Inspection for B | Header names B; notice says Console remains A | Active workspace, task, conversation, composer, and approval state fingerprints remain unchanged |
| Active rail is 24–30 cells or grouped header label is long | Focus both entry actions | UI-only | Complete text action remains visible and focusable; workspace label truncates first | No clipped or invisible clickable region; switcher remains unchanged |
| Workspace has no local folders | Focus/press Files | Blocked intent | Inline/activation guidance points to Settings | No modal, activation, or unfocusable disabled mystery control |
| Read A is slow, then user selects B | B read finishes before A | B snapshot accepted; A token stale | Viewer shows only B | A bytes never flash or replace B |
| Workspace has bindings A and B with the same relative path | Filter while A is selected | Results scoped to A | Contract/filter status names A; only A results appear | No B traversal, ambiguity, or silent binding change |
| Binding is read-only | Select file and attempt Edit/Save | Editability denied | File is viewable with read-only reason | No temp file, write, approval prompt, or permission mutation |
| Binding is revoked or retargeted while open | Read or Save | `binding_changed` | Cached content/draft retained; reopen guidance | No access to old or new target through stale row |
| Agent owns overlapping root | Enter Edit | Lease denied | Viewing works; Edit explains agent conflict | No manual publication |
| Inspector owns overlapping root | Start agent-write run | Admission conflict | Recoverable run-start message | No agent baseline or mutating dispatch begins |
| Clean editor owns root | Choose Done editing, then start agent | Lease released | Viewing state; agent admission can proceed | No stale inspector lease |
| Dirty editor owns root | Choose Done editing | Pending dirty guard | Keep editing focused; destination named | Lease not released until Save/Discard resolves |
| Save is preparing | Choose Cancel before/at publication race | Typed terminal outcome | Unsaved editor if cancellation wins; Finishing save if publication wins | No ambiguous cancelled message and no cancellation after publication |
| External editor changes disk after load | Save draft | `conflict` / `not_published` | Base/Draft/Disk conflict view | External bytes are not silently overwritten |
| Replace succeeds, directory flush fails | Finish Save | `published_durability_unknown` | Modal stays open with warning and recovery | No automatic retry; UI does not claim disk unchanged |
| Clipboard adapter fails | Copy draft | Clipboard failure | Error; draft remains available | No disk, persistence, or dirty-state change |
| Git missing, malformed, or times out | Open/Refresh | Decoration unavailable | Tree and editor remain functional | Authority and Save eligibility unchanged |
| Huge directory/filter corpus | Page or filter | Bounded/truncated result | Counts and truncation disclosed | Event-loop heartbeat remains responsive; caps are not exceeded |
| Unique secret in path/content/draft/filter/error | Exercise read/edit/error paths | Sanitized operational result | Secret may appear only in intended local view | Secret absent from captured logs, notifications, conversation, DB, and review metadata |
| Dirty narrow modal is resized | Wide → narrow → wide | UI-only transition | Draft, baseline, pending guard, and logical focus preserved | Modal is not dismissed or recreated |
| Modal is exercised at 80×24, 100×30, 120×40, and 160×50 | Resize/open in every primary state | UI-only | Required compact/wide/short contract and focus mapping hold | No clipped controls, offscreen actions, lost undo, or hidden status |
| Workspace is archived while open | Read/Git/Save | Scope invalid | Cached view/draft and Copy remain; operations disabled | No new filesystem or Git access |
| File is binary, unsafe-linked, mixed-newline, or publisher-unsupported | Open file | View/metadata-only | Specific plain-language reason | No “try anyway” or best-effort write |

### Test layers

1. **Pure service tests**
   - containment, canonical binding identity, link and special-file rejection;
   - encoding/newline/size classification;
   - selected-binding-only filter scope, stable ordering, filter states, bounds, and truncation;
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
   - manual-versus-agent exclusion, explicit Done-editing release, release on all terminal paths, uncertain-publication retention, and no baseline on denied admission.

4. **Git adapter tests with real repositories**
   - tracked/untracked/conflicted/ignored parsing, rename paths, nested repositories, timeout/output caps;
   - index and repository fingerprints prove no mutation.

5. **Production-shaped Textual tests**
   - mount `TldwCli` using its exact `CSS_PATH` stack;
   - exercise both exact entry surfaces at the incumbent rail widths, including blocked entry guidance and unchanged switcher geometry;
   - exercise safe dismiss paths, inline dirty guard defaults, pre-publication Cancel race, explicit Done editing, stale-result suppression, and focus restoration;
   - exercise 80×24, 100×30, 120×40, and 160×50 in Viewing, Unsaved, Saving, Conflict, and uncertain-publication states;
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
- [ ] Entry actions remain visible, focusable, and unclipped in incumbent rail/group geometry; the separate workspace switcher remains unchanged.
- [ ] Every visible local-folder binding is explicitly represented with identity, access mode, and availability.
- [ ] Filtering is bounded to the explicitly selected binding, exposes complete progress/truncation states, and never traverses another binding.
- [ ] Tree, viewer, filter, and Git work are cancellable where safe and stale-result resistant.
- [ ] Ordinary writable UTF-8 files can be deliberately edited and atomically published; unsupported files are read-only with a reason.
- [ ] Dirty navigation/dismissal, conflicts, binding changes, and uncertain publication preserve recoverable user work.
- [ ] Inspector publication and overlapping agent change-capture windows are mutually excluded by canonical root.
- [ ] The edit lease is continuously visible and **Done editing** releases it without closing the inspector; every other terminal path has a tested lease outcome.
- [ ] Pre-publication Save can be cancelled visibly, while post-publication Save remains mounted and non-cancellable until its terminal outcome.
- [ ] Git decoration is isolated, read-only, accessible, and non-authoritative.
- [ ] No file data or sensitive path/filter/error material enters persistence, logs, agent context, conversation, or Agent Change Review.
- [ ] The specified wide/compact/short layouts meet the same safety model at 80×24, 100×30, 120×40, and 160×50 and preserve modal state across resize.
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
