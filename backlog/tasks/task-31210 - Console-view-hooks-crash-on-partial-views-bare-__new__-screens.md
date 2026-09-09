---
id: TASK-31210
title: 'Console view hooks crash on partial views (bare __new__ screens)'
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-06 11:30'
labels:
  - console
  - bugfix
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
12 tests in `Tests/UI/test_console_native_chat_flow.py` failed at the branch
base (verified red on a clean HEAD worktree — pre-existing, not caused by the
task-31207 branch or the in-flight working-tree changes): every
serialize/restore round-trip test using `_bare_console_screen` (a ChatScreen
built via `__new__`, `__init__` never run) died with
`AttributeError: 'ChatScreen' object has no attribute '_retrieval'` when the
`_console_chat_store` setter forced runtime construction and
`ConsoleRuntime._bind_view_hooks` eagerly called `console_view_hooks()`.

Root cause: `console_view_hooks` evaluated five controller-owned slots
(`_retrieval`, `_skill`, `_fleet` ×3) with direct attribute reads, although
the hook system's own design (`_bind_view_hooks`' docstring) supports views
that claim the runtime before the objects owning slots exist — the same dict
already guarded `_session`/`_prompts` with `getattr`.

ADR required: no — routine bug fix aligning code with the existing documented
hook-binding design; no schema, interface, or boundary change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `console_view_hooks()` returns a full slot map (keys still match `CONSOLE_VIEW_HOOK_SLOTS` exactly) on a bare `__new__` ChatScreen with no controllers wired — absent controllers contribute `None`, never `AttributeError`
- [x] #2 The 12 previously-red `test_console_native_chat_flow.py` tests pass, with the rest of that file (314 total) and the hook-contract suites (`test_console_runtime_ownership.py`, `test_console_viewless_hooks.py`) staying green
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Verify the failure is pre-existing on a clean HEAD worktree (done — red at
   HEAD, so neither the appearance branch nor the in-flight tree caused it).
2. Guard the five controller-owned hook reads with the same `getattr(...)`
   shape the dict already uses for `_session`/`_prompts`; document why in a
   comment pointing at `_bind_view_hooks`' partial-view contract.
3. Rerun the failing file plus the hook-contract suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- PR-port note (2026-09-06): when this fix was ported for the PR against
  dev, `origin/dev` already carried equivalent `getattr` guards in
  `console_view_hooks` (fixed independently upstream). This task closes as
  verified-already-fixed on dev; no code from it ships in the PR, only this
  record.
- Single-method fix in `ChatScreen.console_view_hooks`
  (`UI/Screens/chat_screen.py`): `retrieval`/`skill`/`fleet` resolved via
  `getattr(self, ..., None)` and each of `_rag_capture_provider`,
  `set_pending_skill_install`, `set_pending_skill_script`,
  `wake_user_priority_probe`, `wake_conversation_in_view`, `delivery_ui_hook`
  read off those locals with `getattr(x, "name", None)`. Keys unchanged, so
  the slot-set equality contract holds; absent controllers bind `None` hooks,
  the same value shape the `_session`/`_prompts` guards already produced.
- Note: the `_fleet` slots were already mid-rename in the user's in-flight
  working tree (fleet extraction); the fix wraps those in-flight reads
  minimally and does not otherwise alter them.
- Evidence: `test_console_native_chat_flow.py` 314 passed (was 12 failed);
  `test_console_runtime_ownership.py` + `test_console_viewless_hooks.py` +
  `test_console_button_routing.py` 35 passed.
<!-- SECTION:NOTES:END -->
