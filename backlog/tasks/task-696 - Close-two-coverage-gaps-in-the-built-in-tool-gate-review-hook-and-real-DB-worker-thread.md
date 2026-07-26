---
id: TASK-696
title: >-
  Close two coverage gaps in the built-in tool gate (review hook, real-DB
  worker thread)
status: To Do
assignee: []
created_date: '2026-07-26 06:45'
labels:
  - tests
  - agents
dependencies:
  - TASK-545
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-545 P2's whole-branch review found two acceptance criteria that are satisfied at a lower layer than they claim. Neither is a defect — both were verified manually during review — but neither is protected by the suite, so a regression would pass CI.

**1. The review-hook layer has no real-tool coverage.** P2's ACs "enabling a reads-tagged tool produces a prompt rather than a silent execution" and "a sub-agent's gated call reaches an approval route" are proven only at the `BuiltinToolProvider.invoke()` refusal level. Nothing drives a real `write_file`/`read_file` `ToolCall` through `build_tool_review_hook` to assert a pending approval row is actually emitted; the only hook-level risk test still uses the synthetic `_FakeMutatingRiskyTool` (`Tests/Chat/test_console_chat_controller.py:2169`). The wiring was confirmed by inspection (`console_chat_controller.py:391-422`).

**2. The worker-thread AC is nominal, not substantive.** `Tests/Agents/test_builtin_gate_live_tools.py` monkeypatches `NotesInteropService` away, so it proves `asyncio.run` works off the main thread but not the thing the design spec actually flagged: that `CharactersRAGDB` — built for cross-thread use via `threading.local` and `check_same_thread=False` — really works on the agent's worker thread. A reviewer closed this manually (real DB, real worker thread, row persisted); the suite does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A test drives a real gated built-in tool call through `build_tool_review_hook` and asserts an approval row is emitted for it, replacing reliance on the synthetic risky-tool double for this path
- [ ] A test executes `create_note` on a non-main thread against a real `CharactersRAGDB` (temp path) and asserts the row persists and is readable
- [ ] Both tests fail if the behavior they cover regresses (demonstrate, e.g. by sabotage)
<!-- AC:END -->
