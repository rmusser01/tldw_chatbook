# Minimal File Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one usable disk-backed Markdown/text notes workspace to Library without changing Database Notes or Git.

**Architecture:** A small filesystem service owns path-safe operations and exact-byte parsing. One SQLite module owns the current replica, FTS, protected checkpoints, and tombstones. One composite Textual widget owns the File Notes UI and retained draft; `LibraryScreen` only switches sources and delegates its existing leave guard.

**Tech Stack:** Python standard library, SQLite/FTS5, Textual, pytest.

---

ADR required: yes
ADR path: `backlog/decisions/029-file-notes-disk-authority.md`
Reason: Disk authority, the independent replica, conflict policy, and Library ownership are long-lived storage/UI boundaries.

## Files

- Create `tldw_chatbook/Notes/file_notes_replica.py` — SQLite schema and replica/recovery operations.
- Create `tldw_chatbook/Notes/file_notes_service.py` — safe scanning, exact-byte documents, writes, moves, deletion, restore, and reconciliation.
- Create `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` — tree/search/editor, autosave, polling, root selection, narrow mode, and session changes.
- Modify `tldw_chatbook/UI/Screens/library_screen.py` — Database/Files switch and leave-guard delegation only.
- Create `Tests/Notes/test_file_notes_replica.py`.
- Create `Tests/Notes/test_file_notes_service.py`.
- Create `Tests/UI/test_library_file_notes_workspace.py`.
- Modify `backlog/tasks/task-900 - Add-minimal-disk-backed-File-Notes-editor.md` only for final AC/notes/status.

## Task 1: SQLite replica

- [ ] In each new test module, stub the optional `parakeet_mlx` module before
  importing Chatbook code so the repository's pre-existing macOS MLX collection
  abort does not block these focused tests. Do not change production config.
- [ ] Write failing replica tests for:
  - root-namespaced current-byte upsert and FTS search;
  - protected file/folder-prefix matching;
  - one pre-edit checkpoint per supplied session key;
  - snapshot+tombstone preparation, rollback, persistent deleted listing, and exact-byte restore lookup.
- [ ] Run:
  `../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_replica.py --tb=short`
  and confirm failures are for the missing module.
- [ ] Implement `FileNotesReplica` with three ordinary tables plus manually maintained FTS5:
  - `files(root, relative_path, raw_bytes, content_hash, decoded_text, size, mtime_ns, deleted_at)`;
  - `revisions(root, relative_path, raw_bytes, content_hash, kind, session_key, created_at)`;
  - `protected_paths(root, relative_path, is_prefix)`.
- [ ] Use one connection/transaction context, parameterized SQL, and `UNIQUE(root, relative_path)`. Do not add migrations, repositories, interfaces, triggers, quotas, or pairing metadata.
- [ ] Run the replica test file and commit:
  `feat(notes): add file notes replica`.

## Task 2: Filesystem service

- [ ] Write failing service tests for:
  - `.git`/symlink exclusion and resolved-root containment;
  - UTF-8/BOM/frontmatter split plus LF/CRLF/final-newline preservation;
  - mixed-newline, undecodable, and over-limit read-only results;
  - hash-conflict save, protected-checkpoint-before-write, exact atomic save,
    and preservation of `stat.S_IMODE` permission bits;
  - exclusive create and no-clobber move;
  - delete snapshot, immediate pre-unlink hash recheck, tombstone clearing on
    mismatch/unlink failure, and restore only to an absent path;
  - offline-root reconciliation without mass deletion;
  - external create/modify/delete projection updates and Chatbook-only session changes.
- [ ] Run:
  `../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_service.py --tb=short`
  and confirm failures are for the missing module.
- [ ] Implement `FileNotesService` using:
  - `os.walk(..., followlinks=False)`, `hashlib.sha256`, `tempfile`,
    `os.replace`, `stat.S_IMODE`, exclusive create, and filesystem-enforced
    no-clobber move (`os.link` then source unlink; failure leaves the source);
  - the existing Library 2,000,000-character/8,000,000-byte guard values;
  - `get_safe_relative_path()`/`validate_filename()` where their behavior matches the spec;
  - plain dataclasses for entries, open documents, and operation results.
- [ ] Keep the final hash recheck immediately before `os.replace`. Do not reuse the bidirectional sync engine or add a watcher/dependency.
- [ ] Run service and replica tests and commit:
  `feat(notes): add disk-authoritative file notes service`.

## Task 3: Library workspace

- [ ] Write failing mounted tests for:
  - `Choose folder…`, offline root, and persisted root;
  - tree mode replaced by search results;
  - file open/body edit/save-state flow;
  - create, rename/move, delete, protect/unprotect, and persistent restore
    controls invoking the real service;
  - conflict Reload and Save Copy plus conflict/error leave veto and successful flush;
  - Recently deleted restore after a new workspace instance;
  - polling and a background Database Notes snapshot update keeping the same
    mounted workspace and `TextArea`;
  - narrow Navigator → Editor → Back behavior;
  - Database Notes composition remaining unchanged in Database mode.
- [ ] Run:
  `../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_workspace.py --tb=short`
  and confirm failures are for the missing widget/integration.
- [ ] Implement `LibraryFileNotesWorkspace` with Textual `Tree`, `Input`, `TextArea`, `Button`, timers, and workers. Keep draft/hash/session state on the retained widget; never recompose the editor for polling or status changes.
- [ ] Add the smallest `LibraryScreen` seam:
  - a `Database | Files` source strip while Notes is selected;
  - early Files-workspace composition instead of the normal rail/canvas;
  - File Notes delegation from `flush_pending_work()` and source/rail changes;
  - suppress only the existing full-screen recompose in
    `_apply_local_source_snapshot()` while Files mode is active, retaining the
    snapshot for later Database mode.
- [ ] Persist only the canonical selected root through existing config helpers. Use existing `SelectDirectory`; add no picker or global CSS.
- [ ] Run the three new focused test files and commit:
  `feat(library): add file notes workspace`.

## Task 4: Focused closeout

- [ ] Run only:
  `../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_replica.py Tests/Notes/test_file_notes_service.py Tests/UI/test_library_file_notes_workspace.py --tb=short`
- [ ] Run `git diff --check`.
- [ ] Check all TASK-900 acceptance criteria only when proven by those tests, add concise Implementation Notes, and set TASK-900 to Done through the Backlog CLI.
- [ ] Commit task closeout:
  `docs(notes): complete minimal file notes task`.

## Explicit omissions

No full test suite, CI run, performance certification, native filesystem adapter, cross-process lease, paired database, Git commands, RAG/MCP integration, multiple active roots, folder mutation, quota UI, or recovery tooling.
