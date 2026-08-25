---
id: TASK-22213
title: >-
  Put the Chat first-paint import leg on a diet and give it a guard that can see it
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - startup
  - console
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22213).

Measured this review: warm boot-to-`_ui_ready` regressed ~140 ms (~11%) vs pin
`35d4bf3a1` (1323-1368 -> 1413-1509 ms, five interleaved runs) while the app IMPORT
closure got smaller and every import guard stayed green — the growth is on the legs the
guards cannot see. The Chat first-paint import leg grew +11,638 LOC / +10 modules since
the pin. Named edges (lane 5's AST closure, not diff-grep):
`UI/Screens/chat_screen.py:51` module-level-imports the entire TrajectoryScreen (~4,600
LOC of trajectory work landed since the pin rides the Chat leg);
`Chat/console_voice_input` (2,260 LOC) newly on the leg via `chat_screen.py:241`;
`Widgets/Console/__init__.py` eagerly re-exports the new tree/speech/authority widgets;
`Internal_Prompts` (10 modules) is still on the mount leg via
`Chat/console_chat_controller.py:266` although TASK-21731's title claims otherwise — its
guard (`Tests/Packaging/test_rag_boot_import_closure.py`) imports one module, never
`chat_screen`. PIL and keyring also load pre-first-paint via chat_screen chains
(pre-existing; `session.py:189 -> visual_identity.py:24`; `image.py:38 ->
Image_Generation/config.py:15`).

## Acceptance Criteria

- [ ] `TrajectoryScreen` is not imported at chat_screen module level (screen-registry route or local import at the navigation seam)
- [ ] `Internal_Prompts` is off the Chat mount leg, or kept with a measured, stated cost
- [ ] The closure guard is extended to assert DEFERRED_PREFIXES absent after importing `UI.Screens.chat_screen` (closing the one-module blind spot)
- [ ] chat_screen module import time and boot-to-`_ui_ready` measured before/after with the review's interleaved A/B method; the regression is at least halved or the residual is attributed
- [ ] A `sys.modules` census at `_ui_ready` is pinned as a guard so mount-leg growth is visible in review (the import-weight guard's documented blind spot)
