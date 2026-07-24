# Legacy Notes Sync Containment Plan

**Goal:** Complete TASK-493 by pinning each legacy Notes sync pass to one
canonical directory identity, rejecting unsafe descendants, and preserving
existing file modes during atomic replacement.

**Architecture:** Add a Notes-specific `PinnedSyncRoot` context that opens the
resolved selected root once and performs POSIX scan/read/write operations
relative to verified directory descriptors. Keep the user's lexical root as a
database lookup alias, but pass only canonical paths to filesystem and
successful metadata-update operations. The generic atomic writer remains
unchanged because other exports do not share the Notes mode-preservation
contract.

**Tech Stack:** Python 3.11+, POSIX `dir_fd`/`O_NOFOLLOW`, SQLite, pytest.

---

### Task 1: Lock containment and compatibility contracts with failing tests

**Files:**
- Create: `Tests/Notes/test_sync_containment.py`
- Modify: `Tests/Notes/test_sync_engine.py`

- [x] Cover selected-root symlink compatibility and canonical metadata
  normalization after a successful note update.
- [x] Cover outside-root and in-root file symlinks, directory symlinks,
  hardlinks, non-regular entries, and simulated nested-device entries.
- [x] Cover final-target and intermediate-parent replacement races.
- [x] Prove rejected entries do not abort safe siblings and expose bounded
  per-entry diagnostics.
- [x] Cover existing `0600`, `0640`, and `0644` preservation plus new-file
  `0600` creation.
- [x] Cover unsupported-platform fail-closed behavior without a false safety
  claim.
- [x] Run the new tests and record the expected failures.

### Task 2: Add the pinned Notes filesystem boundary

**Files:**
- Create: `tldw_chatbook/Notes/sync_paths.py`
- Test: `Tests/Notes/test_sync_containment.py`

- [x] Resolve the selected root once, verify/open its canonical directory
  without following it again, and record its device and inode identity.
- [x] Validate relative paths and walk every existing descendant from the root
  descriptor with no-follow directory opens.
- [x] Scan/read only verified regular, single-link, same-device files and
  reject links, reparse-like entries, mounts, and identity changes.
- [x] Create missing directories privately beneath verified descriptors.
- [x] Atomically replace via a verified parent descriptor, rechecking parent
  and final-target identity immediately before replacement.
- [x] Preserve the prior mode for replacement and use `0600` for a new file.
- [x] Return bounded reason codes suitable for per-entry diagnostics.

### Task 3: Route the legacy engine through the pinned boundary

**Files:**
- Modify: `tldw_chatbook/Notes/sync_engine.py`
- Test: `Tests/Notes/test_sync_engine.py`
- Test: `Tests/Notes/test_library_notes_sync_integration.py`

- [x] Open one `PinnedSyncRoot` for the full sync pass and close it in all
  terminal paths.
- [x] Replace `rglob`, direct reads, pathname `mkdir`, and generic atomic
  writes with descriptor-root operations.
- [x] Query existing rows by both lexical and canonical root spellings.
- [x] Store canonical root/file paths only after successful file or database
  synchronization metadata updates.
- [x] Convert entry rejection into a skipped-item diagnostic and continue
  unrelated entries.
- [x] Preserve cancellation, conflict-resolution, progress-callback, and
  session-summary behavior.

### Task 4: Verify and close TASK-493

**Files:**
- Modify:
  `backlog/tasks/task-493 - Contain-legacy-Notes-sync-paths-and-preserve-file-modes.md`

- [x] Run focused containment, existing sync-engine, and Notes integration
  suites.
- [x] Run broader Notes tests and private-path regressions.
- [x] Run changed-file Ruff, Python compilation, and `git diff --check`.
- [x] Run a canonical `/private/tmp` sentinel probe covering a root alias,
  outside sentinel link/hardlink rejection, safe sibling import, `0600` new
  creation, and `0640` replacement preservation.
- [x] Self-review descriptor lifetimes, identity checks, platform fallbacks,
  alias queries, and metadata-update ordering.
- [x] Check all TASK-493 acceptance criteria, add evidence and implementation
  notes, set the task Done through Backlog, and commit only TASK-493 files.

## ADR Check

ADR required: no

ADR path: `backlog/decisions/022-local-private-data-boundary.md` (existing)

Reason: ADR-022 already fixes the legacy Notes containment and
mode-preservation policy. TASK-493 implements that accepted decision without
changing the future authority/recovery design reserved by ADR-021.
