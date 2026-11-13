---
id: TASK-40719
title: "Skill trust and first-send persistence are decided from caller frames and function identity, against ADR-356 rule 4"
status: To Do
created_date: 2026-11-13 06:45
labels:
- follow-up
- architecture
- agents
- console
priority: medium
references:
- backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
- tldw_chatbook/app_skill_wiring.py
- tldw_chatbook/Chat/console_received_dispatch.py
- Tests/Architecture/test_chat_code_identity_ratchet.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** ADR-356 rule 4 says: "Orchestration must not inspect function objects, code/defaults/closures, descriptors, caller stacks/locals or coroutine implementation to establish authority." Two sites on the ordinary chat path do exactly that, with the workflow flag on or off:
- `app_skill_wiring.py:30-51` takes `sys._getframe(1)`. It trusts a skill candidate only if the caller's `f_code` is the code of the `local_skill_trust_service` getter or of `ensure_local_skill_trust_service`, and the caller's locals match (:36-46).
- `Chat/console_received_dispatch.py:892-907` saves the first-send setting only while `_function_unchanged(_NATIVE_CONFIG_CHECK)` holds. That helper is `Agents/activation.py:139-167`, which compares `__code__`, defaults, kwdefaults and closure cells.
- `Chat/console_received_dispatch.py:789-792` also gates `_hook_read` on `submit_draft.__func__ is ConsoleChatController.submit_draft`.

If those functions are wrapped, reloaded or patched, skill trust is silently not recorded, or the first-send setting is not saved.

**Count.** `git grep -E -n '__code__|_getframe|\bcr_code\b|\bf_code\b' <rev> -- 'tldw_chatbook/*.py'` gives:
- 369 before #3480 (e10ce63fe8^1);
- 411 at its merge, unchanged at 7579f75ea9.

`Tests/Architecture/test_chat_code_identity_ratchet.py` (TASK-36944.9) covers only Chat/, where the count fell from 57 to 56. The growth is in Agents/ (25 → 36) and `app*.py` (27 → 29). TASK-39147's AC #3 is ticked, but these sites contradict it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Skill trust is recorded from runtime-owned registration, the same way whoever calls it, with no caller-frame inspection.
- [ ] #2 The first-send setting is saved whether or not the config check function is wrapped, and no Send path is gated on function or code identity.
- [ ] #3 An architecture test covering all of `tldw_chatbook/`, not only Chat/, fails when a new production site decides authority from caller frames or from function, code, default or closure identity.
<!-- AC:END -->
