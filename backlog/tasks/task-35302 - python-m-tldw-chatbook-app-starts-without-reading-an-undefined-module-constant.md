---
id: TASK-35302
title: 'python -m tldw_chatbook.app starts without reading an undefined module constant'
status: To Do
assignee: []
created_date: '2026-10-17 09:11'
labels:
- follow-up
- startup
- bug
dependencies: []
references:
- tldw_chatbook/app.py
- tldw_chatbook/app_entry.py
- tldw_chatbook/app_collections_wiring.py
- tldw_chatbook/app_skill_wiring.py
- tldw_chatbook/__main__.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while verifying PR #3144 (Phase 8) and checked again on origin/dev 111f8e48fc by reading the modules (the app was not launched).

app.py's `if __name__ == "__main__":` block (the last one, about line 5684) registers the running module as `tldw_chatbook.app` and calls `app_entry._run_module_main()`. The module-level assignment `_CONSOLE_SKILL_APP_SOURCE = (TldwCli, globals(), ...)` comes after that block (about line 5704; added in 3e618d7119, "wip: checkpoint reviewed Console preparation integration"). Under `python -m tldw_chatbook.app` the app therefore runs while app.py's body has not reached the assignment. `app_collections_wiring.py` (about line 186) reads `sys.modules["tldw_chatbook.app"]._CONSOLE_SKILL_APP_SOURCE` as an attribute, so it raises AttributeError; `app_skill_wiring.py` and `UI/Console_Modules/skill.py` read it with `.get()` and see None. `python -m tldw_chatbook` (`__main__.py` -> `cli.main_cli_runner`) imports app.py completely first and works. CLAUDE.md's Quick Commands gives `python3 -m tldw_chatbook.app` as the way to run the app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `python -m tldw_chatbook.app` reaches the running app with `_CONSOLE_SKILL_APP_SOURCE` defined, pinned by a test that fails on today's order (a subprocess start under a scratch profile, or a static check that no module-level statement follows app.py's main block)
- [ ] #2 `python -m tldw_chatbook` behaves as before
- [ ] #3 app.py's module-size row does not rise
<!-- AC:END -->
