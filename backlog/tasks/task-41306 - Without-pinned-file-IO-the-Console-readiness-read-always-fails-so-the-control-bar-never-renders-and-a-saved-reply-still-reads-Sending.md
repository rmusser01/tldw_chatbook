---
id: TASK-41306
title: "Without pinned file I/O (Windows), the Console readiness read always fails, so the control bar never renders and a saved reply still reads Sending…"
status: To Do
created_date: 2026-11-16 19:55
labels:
- follow-up
- windows
- console
- bug
priority: high
references:
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- tldw_chatbook/Backup_Recovery/raw_participants.py
- https://github.com/rmusser01/tldw_chatbook/pull/3541
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40714 (PR #3541) found it and left it unfixed. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/`.

**Why.** TASK-40714 lets a Send run when the MCP store is unbound, which is every Windows install. The next step on Windows then fails. In TASK-40714's whole-process probe (`raw_participants._pinned_io_available` patched to False at import), the reply was saved, but the readiness display never warmed, the control bar never rendered, and the transcript still read "Sending…". This is likely the next Windows blocker after TASK-40714. It has not been reproduced on real Windows.

**Evidence.**
- `UI/Console_Modules/console_spend_projection.py:346-348` (`ConsoleReadinessConfigProjection.for_screen` at :260 → its nested `read_current` at :267): `participant = raw._states[active].participant`, then `raw.participant_identity(participant)`.
- Without pinned I/O, the config operation has no installed participant. `participant_identity` (`Backup_Recovery/raw_participants.py:152-162`) then raises `RecoveryRequired("raw_participant_not_installed")` on every read.
- `_pinned_io_available` (`raw_participants.py:281-289`) needs the POSIX guards, `O_DIRECTORY`, `O_NOFOLLOW` and `dir_fd` support for open/stat/mkdir/rename/unlink, so it is False on Windows.
- The checked display proof in the same file reads `raw.participant_identity(proof.participant)` at `:939`. It is not yet known whether it fails the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With pinned I/O unavailable for the whole process, a Console Send saves its reply, the control bar renders, and the transcript shows the reply instead of "Sending…"; a test shows this and is red on dev
- [ ] #2 With pinned I/O available, the readiness display still refuses a config read whose participant identity changed (the existing custody tests still pass)
- [ ] #3 The task notes record a run of the same Send on real Windows (CI or by hand), or say why one could not be run
<!-- AC:END -->
