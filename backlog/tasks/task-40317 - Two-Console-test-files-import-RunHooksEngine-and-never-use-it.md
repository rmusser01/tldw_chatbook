---
id: TASK-40317
title: "Two Console test files import RunHooksEngine and never use it"
status: Done
assignee: ['@claude']
updated_date: 2026-11-13 02:01
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- cleanup
priority: low
references:
- Tests/Chat/test_console_chat_controller.py
- Tests/Chat/test_console_local_review_hook.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cred stream). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

`Tests/Chat/test_console_chat_controller.py:12142` imports `RunHooksEngine, load_hooks_config`, but the engine is built with `trusted_hook_engine` (`:12156`); `Tests/Chat/test_console_local_review_hook.py:1821` has the same unused import. Both were left by 087fc5693f (2026-09-27). An unused import of a class whose constructor changed reads as if the test still builds it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Neither file imports a name it does not use (ruff F401 clean for both)
<!-- AC:END -->

## Implementation Plan

1. Re-check the premise on base 26b2e677dd with `ruff check --select F401 --isolated` on both files.
2. Drop the unused `RunHooksEngine` name from both imports; run the tests that use those helpers.

## Implementation Notes

The premise held on base 26b2e677dd: `ruff check --select F401 --isolated` reported exactly the two unused `RunHooksEngine` imports (`test_console_chat_controller.py:12142`, `test_console_local_review_hook.py:1821`) and nothing else. Removed the name from both imports; the same ruff command now reports "All checks passed!".

Covering run: `TestUserPromptSubmitHooks` plus the hook tests of `test_console_local_review_hook.py` (`-n 4`, load 52): 66 passed.

Files: `Tests/Chat/test_console_chat_controller.py`, `Tests/Chat/test_console_local_review_hook.py`.
