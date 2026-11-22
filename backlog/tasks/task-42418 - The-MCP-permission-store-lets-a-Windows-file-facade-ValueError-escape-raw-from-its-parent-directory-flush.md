---
id: TASK-42418
title: "The MCP permission store lets a Windows file-facade ValueError escape raw from its parent-directory flush"
status: To Do
assignee: []
created_date: 2026-11-22 21:17
updated_date: 2026-11-22 21:17
labels:
- follow-up
- mcp
- windows
- reliability
priority: low
references:
- tldw_chatbook/MCP/permission_store.py
- tldw_chatbook/Utils/windows_files.py
- tldw_chatbook/Tool_Packs/receipt_store.py
- Tests/MCP/test_permission_store.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-21 review of the PR #3684 port (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** On Windows, `_fsync_parent_directory` (`MCP/permission_store.py:252-265`) opens the parent through the native file facade (`Utils/platform_files.os`) and flushes it. The facade refuses some paths with `ValueError` where the standard library raises `OSError` or succeeds: `local_absolute_drive_path_required` for a path without a drive letter, such as a UNC share or a `\\?\` path (`Utils/windows_files.py:666`, `:822`), `windows_path_too_long` (`:563`), and the component checks (`:140`, `:145`). `_save_locked` creates, writes and replaces the file with the standard library first (`permission_store.py:1112-1141`), so on such a profile path the new file is already in place when the flush raises. That `ValueError` escapes raw, where every other failure of this save is an `OSError`, and the in-memory `updated_at` stamp is skipped (`:1143-1144`). The UI callers catch `Exception`, so nothing crashes, but the save reads as an unexplained failure after it landed.

The PR #3684 port fixed the same shape in the Tool Pack receipt store by catching `(OSError, ValueError)` (efe8c8824b, `Tool_Packs/receipt_store.py:496`, `:621`). The POSIX branch here already tolerates only the unsupported-directory errnos and re-raises the rest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A refusal the native facade raises as ValueError during the Windows parent-directory flush surfaces from a permission-store save exactly as an OSError from that flush does
- [ ] #2 A test that simulates the Windows branch with the facade raising ValueError fails before the change and passes after it, and runs on the platforms CI uses
- [ ] #3 Every other caller in `MCP/permission_store.py` that opens or flushes through the native facade is checked for the same shape, and the notes list what was found
<!-- AC:END -->
