---
id: TASK-34433
title: Dead-code removal save_history tool widget branch legacy streaming entry
status: Done
created_date: 2026-10-07 02:43
dependencies:
- TASK-34426
updated_date: 2026-10-08 00:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 7 / F18: save_history has zero callers and per-message transactions tool_message_widgets has a swallowed WrongType bug on a dead branch and the legacy streaming entry point on ChatMessageEnhanced is a silent no-op
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Caller greps pasted as evidence before each removal,WrongType branch fixed or module deleted per caller audit,Legacy streaming entry removed with test callers updated,Targeted tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 21 (T21)
Final commit stack after history rebuild: 29b591c73d (save_history), 8490efb76b (tool_message_widgets), a760d88d00 (legacy streaming entry) — earlier hashes in notes were pre-rebuild. Deferred minors: TOOL-CALLING.md:185 dangling import sketch; sibling seam ChatMessage.update_message_chunk recorded as follow-up task.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Three removals, one commit each; every removal preceded by a fresh caller
grep (evidence in
`.superpowers/sdd/2026-10-06-nonconsole-efficiency-remediation/task-21-report.md`).

1. `save_history` (`3f27010cdc`) — repo-wide grep: zero production callers
   (only the unrelated `dictation.privacy.save_history` setting shares the
   name). Deleted the method (10k-row fetch + per-message transactions +
   O(V×N) variant fixpoint) plus its sole-consumer helper
   `_extract_message_payload` and the now-unused `base64`/`typing.Tuple`
   imports. Consumers updated: 3 persistence tests, the recovered-media
   lifecycle `save_history` mode, the semantic-mutation census (3 boundary
   routes, `_PERSISTENCE_MUTATORS` entry, counts 66/14/26 → 64/13/26) and
   `console-semantic-mutation-inventory.md` (public-owner row, census rows,
   prose counts, stale "bulk-save wrapper" sentence).
2. `tool_message_widgets` (`7aa797cf61`) — audit table: zero production
   imports; the module docstring's "sole production caller"
   `UI/CCP_Modules/ccp_message_manager.py` does not exist; tests were the
   only consumers → DELETE the module (per brief's tests-only branch,
   superseding the WrongType fix). `diff_widgets` keeps its live Console
   consumer. Deleted `test_tool_message_widgets.py` wholesale and the
   `TestToolExecutionWidgetDiffs` class from `test_tool_diff_widgets.py`
   (17 diff_widgets tests retained); fixed the dangling `ConsoleToolDiffRow`
   docstring cross-reference. Residual (harmless, out of scope): CSS rules
   `ChatMessage.-tool-call/-tool-result` in `_messages.tcss` now match
   nothing — left alone to avoid a stylesheet rebuild churn.
3. `ChatMessageEnhanced.update_message_chunk` (`82f0a63a59`) — zero callers
   anywhere; appended to `message_text`, a watcher-less reactive, so the
   composed Markdown never saw it. Removed the method, documented the
   reactive's no-repaint rationale, rewrote the test that pinned the old
   behaviour to assert the entry point is gone, and dropped the incidental
   chunk-lines from `test_ai_generation_state_handling`. Task 14's TTS
   widget-index hooks untouched (verified by `Tests/App/test_tts_widget_index.py`).
   `app_speech.py` was confirmed NOT to import the module (TASK-21103 comment).

Evidence: targeted suite before → after: 268 passed / 4 failed →
239 passed / the SAME 4 failed (failure IDs and the census failure
messages byte-identical; the 29-test delta is exactly the removed tests
3+1+20+5). The 4 failures pre-date this branch state (unreviewed
`credentials.py::_rewrite_database` dynamic SQL, census drift from other
wave tasks, and an environment-fence `raw_source_selection_changed` in
widget-init tests — proven pre-existing via a throwaway worktree at
49f63a2d53). Scoped ruff: error-code profile unchanged or net-negative
(185 → 174 findings; the `Tuple` deletion removed the last UP035).

ADR required: no — mechanical dead-code removal; ADRs 212–215 belong to
other tasks in this wave.

Known sibling seam (deliberately out of scope, recorded in
`backlog/docs/lessons-testing-evidence.md`): `ChatMessage.update_message_chunk`
(`Widgets/Chat_Widgets/chat_message.py:312-321`) is the identical
watcher-less no-op with zero callers and survives.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
