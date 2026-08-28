---
id: TASK-23112
title: >-
  Boot import closure breached its 660 ratchet -- defer the 17 new eager
  modules
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - performance
  - startup
priority: high
dependencies:
  - task-23029
---

## Description

`test_app_import_own_module_count_stays_at_the_post_diet_size` is RED on
pristine dev (`b5eaa9cf64`, 2026-08-28): **666** own modules after `import
tldw_chatbook.app` against the **660** ratchet. Under ADR-097 (boot budgets
are ratchets) the constant does not rise -- the cost defers or is shed.

Vs the last in-budget state (`c6218918d1`, 657 modules): 17 modules added,
8 removed (TASK-23023's Research_Workspace diet). The added edges, traced
with an import-parent recorder (TASK-23029):

1. **`Chat/chat_persistence_service.py`** (+912 lines since the pin) gained
   module-scope imports worth ~12 boot modules: `Chat.attachment_core`
   (drags `Utils.file_handlers`), `Chat.console_chat_fork` (drags the
   `Event_Handlers.Chat_Events` package + `chat_image_events`),
   `Chat.library_activity` (drags `Chat.trajectory` + `Utils.log_sanitizer`),
   `Video_Generation.video_metadata` (drags `video_formats` + the package).
   This is the highest-yield single fix.
2. **`app.py`** gained a module-scope `Chat.console_raw_cli` edge, dragging
   `Tools.raw_cli_executor` and `Agents.run_log` (3 modules).
3. **`Chat.console_runtime`** now eagerly imports `Chat.thinking_blocks`
   (1 module).
4. `Widgets.splash_screen` -> `Widgets.pausable_progress` (TASK-23022) and
   `tldw_chatbook/__init__` -> `Utils.tiktoken_runtime` (ADR-093) look like
   genuine boot-path needs; verify rather than assume.

Beware the known traps: a deferral changes WHICH objects the build binds
(lessons, TASK-21108), a lazy facade protects nothing consumers import
directly (TASK-21200), and tests that patch moved names disconnect silently
(TASK-19830).

## Acceptance Criteria

- [ ] `test_app_import_own_module_count_stays_at_the_post_diet_size` passes
  on dev with `MAX_TLDW_MODULE_COUNT` still 660 (no exception-ledger entry)
- [ ] The deferred imports still resolve on their real use paths (targeted
  tests per moved edge, per the closure-guard house pattern in
  `Tests/Packaging/`)
- [ ] The `boot_import_modules.txt` snapshot is re-pinned via
  `scripts/update_boot_budget_snapshots.py` once the count is back under
  budget (the script refuses while over budget)
- [ ] If reality lands well under 660, apply ADR-097's tightening convention
  (lower to measured + 30)

## Evidence

TASK-23029's implementation notes carry the full trace and the guard's new
breach message, which names all 17 modules.
