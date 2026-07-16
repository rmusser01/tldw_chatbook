---
id: TASK-218
title: Preserve pending Console attachment across screen navigation
status: Done
assignee:
  - '@claude'
created_date: '2026-07-13 09:30'
updated_date: '2026-07-16 15:34'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 (PR #621) deliberately drops a staged-but-unsent attachment when the user navigates away from Console (screen-state serialization is metadata-only by spec; raw bytes never serialize). Decide and implement a preservation strategy that keeps the no-bytes-in-screen-state constraint (e.g. re-process from file_path on restore, or a bounded in-memory app-level stash).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Staging an attachment, visiting another destination, and returning to Console preserves the pending attachment (or shows an explicit, honest notice if the source file vanished)
- [x] #2 Raw attachment bytes still never enter screen-state serialization
- [x] #3 Behavior covered by a mounted navigation round-trip test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
User-approved design: bounded app-level in-memory stash (chosen over drop-with-toast for clipboard grabs). ChatScreen._stash_console_pending_attachments snapshots every session's staged PendingAttachment objects (bytes included) onto app_instance._console_pending_attachment_stash at _serialize_native_console_state time, overwriting whole each save (no stale entries); _adopt_console_pending_attachments re-stages them after store.restore_state, dropping dead-session entries and emptying the stash. Bytes never enter screen-state serialization — the round-trip test walks the whole saved payload asserting no bytes instances (AC#2); the composer '2 files' indicator rebuilds via the normal sync path (label derives from store pendings, no extra wiring). AC#1's vanished-file notice is moot under this design: bytes are preserved in memory, nothing re-processes from disk; app restart drops pendings (accepted trade). Helpers degrade gracefully on bare/unit screens (no app_instance → no-op) so the existing _bare_console_screen round-trip tests run unedited. Mounted AC#3 tests: ConsoleHarness→save_state→RestoredConsoleHarness preservation (path-backed + path-less clipboard pending, byte-identical), dead-session pruning + stash emptied. Sweep 270/0 across stash+native-flow+screen-state+store suites. Files: UI/Screens/chat_screen.py, Tests/UI/test_console_pending_attachment_stash.py.
<!-- SECTION:NOTES:END -->
