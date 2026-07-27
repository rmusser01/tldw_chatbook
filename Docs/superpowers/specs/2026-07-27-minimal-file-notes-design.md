# Minimal File Notes Design

Date: 2026-07-27  
Task: [TASK-900](../../../backlog/tasks/task-900%20-%20Add-minimal-disk-backed-File-Notes-editor.md)  
Decision: [ADR-029](../../../backlog/decisions/029-file-notes-disk-authority.md)

## Goal

Make one existing Git-managed Markdown/text folder usable as Chatbook's File
Notes interface without changing the files' authority, the user's Git workflow,
or existing Database Notes.

## User experience

Library Notes gets a `Database | Files` source switch. Database mode remains
unchanged. Files mode replaces the normal Library rail and canvas with its own
two-pane workspace; it is not nested as a third pane.

Files mode has:

- a collapsible folder tree in the left navigator;
- search results replacing the tree while a search is active;
- an editor with a relative-path breadcrumb, save state, and file actions;
- `dirty`, `saving`, `saved`, `conflict`, and `error` save states;
- a current-session list of paths Chatbook created, modified, moved, deleted,
  or restored;
- a small `Recently deleted` group, shown only when restorable tombstones
  exist;
- Navigator and Editor views on narrow terminals instead of squeezed panes.

The initial file actions are create, rename/move into an existing directory,
delete, restore, and protect/unprotect history. Create and move require an
absent destination enforced by the filesystem operation, not only by a prior
existence check; they never replace another file. Folder creation and Git
commands are not included.

With no configured root, Files mode shows one `Choose folder…` action. The
chosen canonical path persists across restarts. Changing roots first passes the
same editor-leave guard as changing files. A missing or unreadable root is shown
as offline and is never created automatically or interpreted as mass deletion.

## Storage and file rules

The canonical selected root and relative path together identify a note and
namespace its replica/recovery rows. Supported files are `.md`, `.markdown`,
`.txt`, and `.text`. Scans ignore `.git`, do not traverse symlink directories
or files, and reject any read or mutation whose resolved path is outside the
root.

The editor always loads the current file from disk. UTF-8 and UTF-8-with-BOM
files with uniform LF or CRLF body newlines are editable; their newline style
and final-newline state are preserved. Files that cannot be decoded, contain
mixed newline styles, or exceed the existing Library note size guard remain
path-visible but read-only.

Chatbook always removes an optional UTF-8 BOM from the editable body and keeps
its exact bytes in the preserved prefix. After that BOM, a frontmatter block is
recognized only when the first line is exactly `---` and a later line is
exactly `---` or `...`. The preserved prefix extends through the closing
delimiter, including that line's terminating newline when present, and only the
remaining bytes are exposed as the body. Saving concatenates the untouched
prefix with the edited body using the original newline convention. Without both
delimiter lines, only the optional BOM is preserved separately and the rest is
edited as one body.

## Save and external-change flow

1. Opening records the SHA-256 of the exact disk bytes.
2. Body changes schedule the existing debounced autosave behavior.
3. Before the first save of a protected note in an editing session, SQLite
   commits the exact current bytes as that session's pre-edit checkpoint. If
   this fails, the file is not written. An editing session begins on open or
   reload and ends when that file is left.
4. Before writing, Chatbook reads and hashes the file again.
5. If the hash differs, autosave stops and the editor shows a conflict with
   Reload and Save Copy actions. It never silently overwrites.
6. Otherwise Chatbook writes a same-directory temporary file, preserves the
   original permission bits, rechecks the source hash immediately before
   publication, and publishes it with `os.replace` only if it still matches.
7. The new hash becomes the editor baseline and the SQLite replica is updated.

This prevents every external change detected before publication from being
overwritten. No portable filesystem API can eliminate a non-cooperating
writer's final race after the last check.

A lightweight timer polls path, size, and modification time every one to two
seconds. Only changed candidates are rehashed. A manual Refresh action runs the
same reconciliation. External creates enter the tree, replica, and FTS;
external modifications refresh those surfaces unless the file has a dirty open
editor, which becomes a conflict; external deletions leave their last replica
bytes recoverable but disappear from tree/search; and external renames are
treated as one deletion plus one creation. Initial scans and reconciliation run
off the UI thread. They update navigator/search/status widgets directly and
never trigger a full Library recompose while an editor is open. No file-watcher
dependency is added.

Changing the selected file, source, root, Library rail destination, or app
screen first awaits pending autosave. A remaining dirty, conflict, or error
state vetoes that navigation and keeps the editor and its retained draft
mounted. Narrow mode switches programmatically from the actual available
canvas width: selecting a file opens Editor and Back returns to Navigator
without remounting the draft.

## SQLite replica and recovery

One separate `file_notes.sqlite` in Chatbook's user-data directory contains:

- `files`: canonical root, relative path, latest exact bytes, hash, observed
  metadata, and optional deletion timestamp;
- `revisions`: canonical root, relative path, exact prior bytes, hash, reason,
  and timestamp;
- `protected_paths`: canonical root plus a protected file path or folder
  prefix;
- an FTS5 index over decoded current content.

All linked files receive a current replica for search and recovery. Historical
revisions are recorded only for paths selected with Protect and are coalesced
to one pre-edit checkpoint per open editing session. Unprotect stops future
history without deleting existing recovery data. Delete always requires a
committed snapshot and tombstone, then rechecks that the disk hash still matches
that snapshot immediately before unlinking. A mismatch clears the tombstone and
becomes a conflict. If the transaction or unlink fails, the file is not deleted
and Chatbook clears the tombstone; the next reconciliation also clears any
stale tombstone for a file still present. Restore writes the stored exact bytes
only when the target path is absent. Tombstones remain available from
`Recently deleted` after restart; the current-session list additionally exposes
Restore for deletions made during that Chatbook session.

If the replica is unavailable, disk browsing and unprotected editing continue
with an error state. Protected edits and deletion stop because their promised
recovery copy cannot be recorded.

## Existing code boundaries

- Reuse the Library screen and `LibraryNotesCanvas` styling/state patterns.
- Reuse Textual's existing tree, input, text-area, timer, and worker primitives.
- Reuse the existing `SelectDirectory` picker for selecting the root.
- Add a small File Notes service and replica module; do not extend ChaChaNotes
  or the existing bidirectional sync engine.
- Keep the Files workspace in one composite Library widget so polling and
  background Database Notes refreshes cannot remount its editor.
- Initialize File Notes only when Files mode is opened.

## Focused verification

Tests cover only the data-loss and primary-use boundaries:

- body edits preserve frontmatter/BOM/newline bytes;
- a changed disk hash blocks autosave overwrite;
- atomic save updates the real file and replica;
- protected save cannot write before its pre-edit checkpoint commits;
- dirty/conflicted drafts veto file, root, source, rail, and screen changes;
- polling updates mounted widgets without remounting the editor;
- delete cannot unlink before its snapshot/tombstone commits;
- create and move refuse an existing destination;
- restore remains discoverable after restart and reproduces exact bytes;
- reads and mutations cannot follow symlinks or escape the selected root;
- unavailable roots do not create directories or mark every note deleted;
- root changes cannot collide with another root's replica/recovery rows;
- tree, search replacement, editor state, and narrow switching work in the
  mounted Library surface.

No full-suite or platform qualification gate is part of this implementation.

## Explicit non-goals

Multiple active roots, folder mutation, Git staging/commit/push controls, RAG,
MCP, keywords, templates, non-UTF-8 editing, native filesystem adapters,
cross-process leases, paired databases, recovery quotas, bulk recovery tools,
and crash/power-cut certification.
