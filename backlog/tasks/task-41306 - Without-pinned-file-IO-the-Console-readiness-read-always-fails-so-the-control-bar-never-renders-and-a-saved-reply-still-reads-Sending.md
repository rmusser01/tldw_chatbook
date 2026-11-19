---
id: TASK-41306
title: "Without pinned file I/O (Windows), the Console readiness read always fails, so the control bar never renders and a saved reply still reads Sending…"
status: Done
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

## Implementation Plan

ADR required: no. The fix keeps the existing custody rule (a config read with an installed participant must name it) and only stops treating "no participant, no pinned I/O" (the Windows posture every other raw reader already accepts) as a refusal.

1. Reproduce: load a pytest plugin that sets `raw_participants._pinned_io_available` False at import (before the conftest's first config read) and run the Windows Send node; confirm the control bar never renders because `read_current` calls `participant_identity(None)`.
2. Fix in `read_current` (`UI/Console_Modules/console_spend_projection.py`): when the config operation has no participant and was opened without pinned I/O, issue no checked display proof instead of refusing the read. Presentation then takes the existing live config route (`_checked_display_status` returns None), so the display-time check (:939) never sees a proof without a participant. Pinned operations still call `participant_identity`.
3. Test: a parent test runs the Windows Send node in a child pytest with the plugin loaded (whole process unpinned); the Send node also waits for "Sending…" to leave the screen after the reply. Red at base, green at head; mutation check by reverting the fix.
4. AC #2: run the existing checked-display custody tests (`Tests/UI/test_console_checked_display_scope.py` and the other readiness display files) on base and head.
5. AC #3: no Windows host here and CI may not be triggered; leave unticked and say so.

## Implementation Notes

**Closed without a product change: the premise does not hold on real Windows (2026-11-17, base fbf1c9ea74 = dev 5e3c8b2bd4 + filing). The ACs are left unticked because the failure they describe does not occur on any supported platform.**

- On `nt`, `raw_participants` and `private_paths` import `os` from `Utils/platform_files.py`, which is a `WindowsOS()` facade (`Utils/windows_files.py`), not the stdlib module. It defines `O_DIRECTORY`/`O_NOFOLLOW`/`O_NONBLOCK`/`O_NOCTTY`, `supports_dir_fd` (open, stat, mkdir, readlink, rename, replace, unlink, rmdir, link), `supports_follow_symlinks` ({stat}), `geteuid`, `fstat`, `fchmod` and `fsync`. A probe that installed a `WindowsOS()` as both modules' `os` returned `(_posix_guards_available(), _pinned_io_available()) == (True, True)`. So on Windows the config operation gets an installed participant and `read_current` never calls `participant_identity(None)`.
- The failure is real only when `_pinned_io_available()` is False for the whole process. Reproduced with a `-p` plugin that patches it at import: the Windows-profile Send node failed with "The Console control bar did not render within 15.0 s (... the readiness warm-up was not current)". No supported platform (Linux, macOS, Windows) takes that posture.
- PR #3534 (`codex/windows-post-save`, 5bff138fda) does not touch `_pinned_io_available` or `read_current`'s participant lines, and reports real-Windows Sends passing, which agrees with the above.
- A fix and test were written (81da87d9bc, not landed). When a config operation has no participant and no pinned I/O, the read issued no display proof and presentation took the existing live config route; a child pytest ran the Send with pinned I/O off for the whole process and was red without the fix, and 16 readiness/display custody files gave the same 5 failures on base and head. Dropped: it guards a posture no shipped platform has.
- The same finding makes TASK-40714's "the MCP store is unbound on every Windows install" false: `mcp_source_participants.bind` returns early only when `_pinned_io_available()` fails, so on Windows the store binds at the default data path. TASK-41308 corrects that wording; the non-default data directory case TASK-40714 fixed still stands.
- AC #3: not run; no Windows host here, and CI may not be triggered from this stream.
