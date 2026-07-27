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
unchanged.

Files mode has:

- a collapsible folder tree in the left navigator;
- search results replacing the tree while a search is active;
- an editor with a relative-path breadcrumb, save state, and file actions;
- `dirty`, `saving`, `saved`, `conflict`, and `error` save states;
- a current-session list of paths Chatbook created, modified, moved, deleted,
  or restored;
- Navigator and Editor views on narrow terminals instead of squeezed panes.

The initial file actions are create, rename/move into an existing directory,
delete, restore, and protect/unprotect history. Create and move require an
absent destination; they never replace another file. Folder creation and Git
commands are not included.

## Storage and file rules

The selected root and relative path identify a note. Supported files are
`.md`, `.markdown`, `.txt`, and `.text`. Scans ignore `.git`, do not traverse
symlink directories or files, and reject any read or mutation whose resolved
path is outside the root.

The editor always loads the current file from disk. UTF-8 and UTF-8-with-BOM
files are editable. Files that cannot be decoded are visible but read-only.

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
3. Before writing, Chatbook reads and hashes the file again.
4. If the hash differs, autosave stops and the editor shows a conflict with
   Reload and Save Copy actions. It never silently overwrites.
5. Otherwise Chatbook writes a same-directory temporary file, preserves the
   original permission bits, rechecks the source hash immediately before
   publication, and publishes it with `os.replace` only if it still matches.
6. The new hash becomes the editor baseline and the SQLite replica is updated.

This prevents every external change detected before publication from being
overwritten. No portable filesystem API can eliminate a non-cooperating
writer's final race after the last check.

A lightweight timer polls path, size, and modification time every one to two
seconds. Only changed candidates are rehashed. A manual Refresh action runs the
same reconciliation. External creates enter the tree, replica, and FTS;
external modifications refresh those surfaces unless the file has a dirty open
editor, which becomes a conflict; external deletions leave their last replica
bytes recoverable but disappear from tree/search; and external renames are
treated as one deletion plus one creation. No file-watcher dependency is added.

## SQLite replica and recovery

One separate `file_notes.sqlite` in Chatbook's user-data directory contains:

- `files`: relative path, latest exact bytes, hash, observed metadata, and
  optional deletion timestamp;
- `revisions`: relative path, exact prior bytes, hash, reason, and timestamp;
- `protected_paths`: protected file paths or folder prefixes;
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
only when the target path is absent.

If the replica is unavailable, disk browsing and unprotected editing continue
with an error state. Protected edits and deletion stop because their promised
recovery copy cannot be recorded.

## Existing code boundaries

- Reuse the Library screen and `LibraryNotesCanvas` styling/state patterns.
- Reuse Textual's existing tree, input, text-area, and timer primitives.
- Reuse the enhanced directory picker for selecting the root.
- Add a small File Notes service and replica module; do not extend ChaChaNotes
  or the existing bidirectional sync engine.
- Initialize File Notes only when Files mode is opened.

## Focused verification

Tests cover only the data-loss and primary-use boundaries:

- body edits preserve frontmatter/BOM/newline bytes;
- a changed disk hash blocks autosave overwrite;
- atomic save updates the real file and replica;
- delete cannot unlink before its snapshot/tombstone commits;
- create and move refuse an existing destination;
- restore reproduces exact bytes;
- reads and mutations cannot follow symlinks or escape the selected root;
- tree, search replacement, editor state, and narrow switching work in the
  mounted Library surface.

No full-suite or platform qualification gate is part of this implementation.

## Explicit non-goals

Multiple active roots, folder mutation, Git staging/commit/push controls, RAG,
MCP, keywords, templates, non-UTF-8 editing, native filesystem adapters,
cross-process leases, paired databases, recovery quotas, bulk recovery tools,
and crash/power-cut certification.
