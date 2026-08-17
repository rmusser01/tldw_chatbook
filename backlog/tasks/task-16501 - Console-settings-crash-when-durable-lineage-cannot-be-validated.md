---
id: TASK-16501
title: Console settings crash when durable lineage cannot be validated
status: Done
assignee:
  - '@claude'
created_date: '2026-08-15 16:10'
updated_date: '2026-08-15 16:45'
labels:
  - console
  - bug
  - context-memory
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-reported crash: opening Console settings raises `TypeError: 'NoneType' object is not iterable` in `select_valid_memory()` (console_context_compaction.py:236), with `active_messages=None` and `candidates=()`.

`ConsoleChatController._durable_context_snapshots()` is typed `tuple[...] | None` and returns `None` whenever the active lineage cannot be validated (a user message on the active path with no `persisted_message_id` yet, a store `KeyError`, a missing/failing `get_message_version` reader, broken variant state). The two send-path consumers (`_compaction_admission`, `_apply_conversation_memory_preflight`) guard with `if not snapshots:`, but the two settings-surface consumers — `context_control_inputs()` and `reset_active_context_memory()` — pass the result straight into `select_valid_memory()`. The chat_screen caller catches only `(KeyError, ValueError)`, so the `TypeError` escapes and takes down settings opening (both the full modal and the Alt+M popover call `_active_console_context_control_state()`). The crash fires even with zero stored memories because the positions dict is built before candidates are iterated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Opening Console settings does not crash when the active durable lineage cannot be validated; the memory input degrades to None
- [x] #2 Reset-current-memory returns None (no deactivation) instead of crashing when the lineage cannot be validated
- [x] #3 Regression tests reproduce the None-snapshots state through the controller seam and were verified RED before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests in Tests/Chat/test_console_context_compaction.py using the controller preflight fixture plus an unpersisted user message on the active path (which makes `_durable_context_snapshots` return None); verify both tests RED with the reported TypeError.
2. Guard both settings-surface call sites in `console_chat_controller.py` with the same `if not snapshots:` pattern the send-path consumers already use, keeping `select_valid_memory`'s contract unchanged.
3. Verify tests GREEN; run the compaction and context-controls suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Guarded the two settings-surface consumers of `_durable_context_snapshots()` in console_chat_controller.py — `context_control_inputs()` and `reset_active_context_memory()` — with the same `if not snapshots:` degradation the send-path consumers (`_compaction_admission`, `_apply_conversation_memory_preflight`) already used. `select_valid_memory()`'s contract is unchanged: None still means "lineage cannot be validated right now", which correctly selects no memory rather than being coerced to an empty prefix.

Regression tests in Tests/Chat/test_console_context_compaction.py reach the None state through the product path: the controller preflight fixture plus one unpersisted user message on the active path (the realistic mid-persistence state from the user report), with a seeded active memory proving nothing gets selected or deactivated. Both tests verified RED with the exact reported `TypeError: 'NoneType' object is not iterable` at console_context_compaction.py:236 before the fix, GREEN after.

Not changed: chat_screen's `(KeyError, ValueError)` catch around `context_control_inputs` — after this fix the seam no longer raises for unvalidatable lineage, and widening that catch would hide real programming errors.

Files: tldw_chatbook/Chat/console_chat_controller.py, Tests/Chat/test_console_context_compaction.py.
<!-- SECTION:NOTES:END -->
