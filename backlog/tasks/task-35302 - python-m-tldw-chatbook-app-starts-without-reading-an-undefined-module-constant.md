---
id: TASK-35302
title: 'python -m tldw_chatbook.app starts without reading an undefined module constant'
status: Done
assignee:
- '@claude'
created_date: '2026-10-17 09:11'
updated_date: '2026-10-19 00:10'
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
- [x] #1 `python -m tldw_chatbook.app` reaches the running app with `_CONSOLE_SKILL_APP_SOURCE` defined, pinned by a test that fails on today's order (a subprocess start under a scratch profile, or a static check that no module-level statement follows app.py's main block)
- [x] #2 `python -m tldw_chatbook` behaves as before
- [x] #3 app.py's module-size row does not rise
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review PR #3133 (a reader-side guard from an outside contributor) against the cause.
2. Move the assignment above app.py's main block; keep a reader-side guard as a backstop.
3. Pin the order with a static check and the reader with a stand-in app module; check both fail on dev.
4. Prove it at the real entry: a python -m-order probe and a sandboxed live launch of both entries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review of #3133 (by @Csavoldi): its guard is correct and stops the crash, but it keeps the cause. The record stays undefined for the whole life of a `python -m tldw_chatbook.app` process, every served web child included, so the collections wiring would always take its synchronous fallback on the UI loop (ensure_collections_capture_services) instead of the off-loop deferred initializer, and the two skill readers keep returning None, their stock paths off. It added no test. The guard itself is kept here as the backstop, with the skill readers' shape check.

Fix: app.py defines _CONSOLE_SKILL_APP_SOURCE before the main block (moved, so app.py stays at its 5,710-line budget). app_collections_wiring reads the record with vars(...).get() and returns None, the existing fallback, unless it is the 5-tuple, like app_skill_wiring.py and UI/Console_Modules/skill.py.

Evidence:
- Tests/CI/test_app_module_entry_contract.py (the PR Fast Lane runs Tests/CI): the check that nothing at module level follows app.py's last main block fails on dev ('line 5704'); the reader test, with a stand-in app module, fails on dev with the AttributeError (record missing) and a ValueError (record malformed). All three pass here.
- A probe that runs app.py through runpy as `-m` does, with the hand-off to app_entry._run_module_main replaced: dev 'record_defined=False'; this branch 'record_defined=True stock_record_matches=True'.
- Live, under a sandboxed scratch profile: `python -m tldw_chatbook.app` on dev 5eb9a37b34 crashed at startup (AttributeError: module '__main__' has no attribute '_CONSOLE_SKILL_APP_SOURCE'); on this branch the Console reached 'Setup complete ... Ready', ran 35 s with no error and quit cleanly. `python -m tldw_chatbook` did the same.
- Related suites (startup unlock, recovery entry points, startup hygiene, collections and skill wiring, 15 files) and all of Tests/CI, on dev and here: the same 17 failures in both, all in three files that fail on dev on this machine (test_recovery_entrypoints.py, test_console_skill_commands.py and one test of test_console_skill_trust_stock_owned.py); dev also failed test_stock_console_skill_setup_has_original_owned_lifetime[send_enter] once, a load flake that passed twice alone on each side. Tests/CI: 995 passed on dev, 998 here (the three new tests).

Files: tldw_chatbook/app.py, tldw_chatbook/app_collections_wiring.py, Tests/CI/test_app_module_entry_contract.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
`python -m tldw_chatbook.app` starts again: app.py defines the Console stock record before its main block, so served web children no longer crash at boot, and the collections wiring no longer crashes on a missing record. Supersedes #3133, whose guard is kept as the backstop.
<!-- SECTION:FINAL_SUMMARY:END -->
