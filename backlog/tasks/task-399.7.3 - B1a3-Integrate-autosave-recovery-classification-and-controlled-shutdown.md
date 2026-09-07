---
id: TASK-399.7.3
title: B1a3 Integrate autosave recovery classification and controlled shutdown
status: To Do
assignee: []
created_date: '2026-07-23 15:36'
labels:
  - notes
  - filesystem
  - recovery
  - ui
dependencies:
  - TASK-399.7.2
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399.7
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate journaled save with editor persistence, deterministic interrupted-operation recovery, independent recovery access, and an app-owned shutdown barrier without yet exposing writable mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Autosave preserves the existing two-second debounce, enforces a 30-second maximum interval from first dirty input, allows Save now to bypass both timers, ignores stale editor generations, and converts failed saves into durable actionable draft or Attention state without discarding the buffer.
- [ ] #2 Only the elected process holding exclusive mutation ownership classifies interrupted create/save operations from observed disk, expected hashes, metadata fingerprints, journal state, and exact-owned artifacts; it never blindly replays a mutation and cleans artifacts only after durable byte and metadata capture.
- [ ] #3 Recovery-only access enumerates, verifies, and exact-exports retained safety copies, drafts, and conflict sides without opening ChaChaNotes or file_notes.db; Recovery items appears only for genuine retained evidence.
- [ ] #4 Changed this session is an in-memory process-lifetime ordered map keyed by note UUID, records only completed Chatbook working-tree changes, coalesces repeated saves and latest paths or terminal state, excludes no-ops and external changes, and retains the 5,000 most recently changed distinct notes.
- [ ] #5 Controlled shutdown closes new command and editor admission, stops new autosave scheduling, awaits an active mutation through completion or durable Attention, flushes or safely retains every dirty buffer, releases leases, and only then permits generic worker cancellation.
- [ ] #6 If recovery storage cannot retain the only live draft, Library or app shutdown remains vetoed while exact export and explicit discard are offered; the UI never calls that buffer recoverable before durable verification.
- [ ] #7 Fault tests cover editor navigation, Library reconstruction, process termination boundaries, recovery corruption or low space, and startup classification while Database Notes remains usable.
- [ ] #8 This child exposes no writable control or read/write mode transition.
<!-- AC:END -->
