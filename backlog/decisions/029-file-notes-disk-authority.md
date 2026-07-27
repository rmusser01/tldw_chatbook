# 029 - File Notes use disk authority with one SQLite replica

Date: 2026-07-27
Status: accepted
Related task: [TASK-900](../tasks/task-900%20-%20Add-minimal-disk-backed-File-Notes-editor.md)

## Context

Users already keep Markdown and text notes in Git-managed folders. Chatbook must
edit those files directly so ordinary `git status`, `git add`, and `git commit`
continue to work. Database Notes and the existing bidirectional sync engine use
different ownership rules and must not become an equal authority for these
files.

## Decision

- The selected canonical folder plus each relative path are authoritative and
  namespace replica/recovery rows.
- Chatbook reads editor content from disk and writes ordinary filesystem
  changes beneath that folder.
- One dedicated SQLite database outside the selected folder stores the latest
  observed bytes for search/recovery plus coalesced revisions for explicitly
  protected paths.
- SQLite is never an editor authority. It can be rebuilt from disk; a stale row
  cannot overwrite a file.
- Every save compares the current disk hash with the hash observed when the
  editor loaded or last saved. A mismatch becomes a visible conflict.
- A protected note's exact pre-edit bytes commit before its first write in an
  editing session; failure stops that write.
- A valid leading frontmatter block is kept as untouched bytes while only the
  body is edited.
- Delete commits a recovery snapshot and tombstone before unlinking the file.
- A missing root is offline, never auto-created or treated as mass deletion.
- Git remains external. Chatbook only reports files changed during the current
  Chatbook session.

## Consequences

- Database Notes, ChaChaNotes, RAG, MCP, Sync, and Git integrations need no
  schema or behavior changes.
- One selected root, UTF-8 text files, moves into existing directories, and
  polling-based external-change detection are sufficient for the first release.
- Multiple roots, Git controls, folder mutation, watcher dependencies,
  platform-specific filesystem adapters, storage pairing, quotas, and recovery
  tooling are deferred until demonstrated need.
