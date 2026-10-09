---
id: TASK-34437
title: Remove remaining legacy streaming no-op on ChatMessage and dangling tool-widget
  doc sketch
status: Done
created_date: 2026-10-08 00:36
updated_date: 2026-10-09 05:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34433: sibling seam ChatMessage.update_message_chunk (chat_message.py:312-321) is the identical watcher-less zero-caller no-op (comments cite nonexistent handle_streaming_chunk); Docs/Development/Tool-Calling/TOOL-CALLING.md:185 still sketches importing the deleted tool_message_widgets module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sibling entry removed with caller-grep evidence,TOOL-CALLING.md sketch updated or annotated historical,Targeted tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 21 review minors + task-21-report.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Commit 83a3f83595 (single commit, both targets).

1. Sibling no-op removed. Fresh repo-wide grep for
   `update_message_chunk` (--include="*.py", prod + Tests): production hit
   was ONLY the definition (chat_message.py:312); test hits were 34433's
   absence assertions for the Enhanced class; the rest docs/backlog/qa
   notes. message_text write audit on ChatMessage: `__init__` (line 91)
   plus the dead method (line 319) only -- no other watcher-less mutation
   entry points. Removed the method and replaced the stale
   "Remove repaint=True..." comment with the same no-repaint rationale
   34433 left on the Enhanced class (cites both task IDs). Task-14 TTS
   hooks (on_mount/on_unmount/watch_message_id_internal) untouched.
   app_speech.py verified importing cleanly post-change -- note it
   deliberately imports NEITHER chat widget class (TASK-21103); it goes
   through tts_widget_index, so the guardrail concern was moot but was
   verified anyway.
2. Test: added `test_legacy_chunk_entry_point_removed` to
   Tests/Widgets/test_chat_message_artifact_actions.py, parametrized over
   BOTH ChatMessage and ChatMessageEnhanced (mirrors 34433's
   absence-with-rationale pattern; both classes in one place so the
   sibling-seam lesson can't recur). No test pinned the sibling's old
   behaviour, so nothing needed rewriting.
3. Doc: TOOL-CALLING.md Phase-1 sketch replaced -- it imported the deleted
   `tool_message_widgets` module AND the equally nonexistent
   `get_tool_executor` factory. Replacement states the as-built seams
   (Agents/tool_catalog.py `ToolCatalogRegistry`; Console transcript TOOL
   marker rows + ConsoleToolDiffRow in console_transcript.py), historical
   components block annotated, Last Updated notes the correction.
4. Evidence: targeted suite (6 files touching ChatMessage /
   update_message_chunk) after: 140 passed, 1 failed. The failure
   (TestChatMessageEnhancedInitialization::test_user_message_initialization,
   recovered_media/get_user_data_dir plumbing) reproduced byte-identically
   at base commit 7839c07c24 in a throwaway worktree -- pre-existing,
   untouched files. Ruff error-code profiles on both touched Python files
   identical before/after (31 + 1 findings, all pre-existing).

Residuals (out of scope, recorded): AGENTS.md:54 still lists
`tool_message_widgets.py` under Key Widgets; TOOL-CALLING-IMPLEMENTATION.md
(class mentions) and TOOL-CALLING.md's dated historical `ToolExecutor`
mentions (no plain `ToolExecutor` class exists anymore -- only
Workspace/Remote variants) left as history.

ADR required: no -- mechanical dead-code removal + doc correction;
follows TASK-34433 precedent.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
