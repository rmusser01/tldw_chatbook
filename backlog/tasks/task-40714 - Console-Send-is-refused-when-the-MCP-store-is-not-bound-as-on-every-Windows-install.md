---
id: TASK-40714
title: "Console Send is refused with 'Console preparation source unavailable' when the MCP store is not bound, as on every Windows install"
status: To Do
created_date: 2026-11-13 06:45
labels:
- follow-up
- console
- bug
- windows
priority: high
references:
- tldw_chatbook/UI/Console_Modules/turn_admission.py
- tldw_chatbook/Chat/console_configuration_preparation.py
- tldw_chatbook/Backup_Recovery/mcp_source_participants.py
- Tests/UI/test_console_send_draft_snapshot.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** On Windows, every ordinary Console Send is refused with "Console preparation source unavailable; Send again." The provider is never called, and Send again gives the same refusal. `[console] agent_runtime` defaults to on (`UI/Console_Modules/turn_admission.py:304`), so this is the default path.

**Path.**
- Send reaches `UI/Console_Modules/wiring.py:3737` (`_capture_console_configuration_async`), then `Chat/console_configuration_preparation.py:2806`, which asks `standard_console_sources(service)`.
- That is False when the MCP store was never bound: `MCP/console_snapshot.py:135-137` raises `mcp_source_not_installed`, which is caught at :220-223.
- So line 2905 sets `captured = None`, and the stock screen falls back to `capture_authority` (:2929).
- `UI/Console_Modules/turn_admission.py:184-189` then raises whenever the screen has a Console runtime, which a real ChatScreen always has.
- On a stock screen, every non-standard path ends in that refusal. No fallback exists.

**When the MCP store is unbound.**
- `Backup_Recovery/mcp_source_participants.py:105` returns from `bind()` without binding when `_pinned_io_available()` is False. `Backup_Recovery/raw_participants.py:281-289` needs POSIX `O_DIRECTORY`/`O_NOFOLLOW` and `dir_fd` support, so the store is never bound on Windows.
- `bind()` also skips (:110-120) when the config cache is missing or the store path is not `user_data_dir/<leaf>`. Whether a non-default data directory reaches that branch has not been re-proven.

**Tests.** On 2026-11-12 (dev e0186fa10a), simulating Windows made 18 of 32 tests in `Tests/UI/test_console_send_draft_snapshot.py` fail, 8 of them with this refusal. Its named tests pass on dev only because commit 022fc46a86 moved them onto the default data path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With pinned I/O unavailable, as on Windows, an ordinary Console Send reaches the provider and its reply is saved.
- [ ] #2 With the MCP store unbound, the turn runs without MCP tools instead of being refused.
- [ ] #3 The Send draft snapshot tests pass under simulated Windows and under a non-default data directory, not only on the default data path.
<!-- AC:END -->
