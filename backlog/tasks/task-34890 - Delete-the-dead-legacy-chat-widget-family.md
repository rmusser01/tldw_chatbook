---
id: TASK-34890
title: Delete the dead legacy chat-widget family
status: Done
created_date: 2026-10-12 17:51
labels:
- simplification
- dead-code
priority: high
assignee:
- '@zcode'
updated_date: 2026-10-12 19:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console transcript unification is complete but the previous generation of chat-message widgets was never carried out. About 2.1k LOC across 7 files is production-dead and misleads contributors.

Evidence (verified on dev df4985e): Verified on dev df4985e: zero production constructors for ChatMessageEnhanced and ChatMessage; console_transcript.py + ConsoleChatMessage is the sole live renderer (chat_screen.py:329 564). About 8 test files import the dead modules; Tests/UI/test_widget_css_consolidation.py:1198 enumerates ChatHandoffCard in a CSS census. Continues the task-34433 removal series. No ADR protects these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Widgets/Chat_Widgets chat_message.py and chat_message_enhanced.py and its top-level alias shim are removed,Event_Handlers/Chat_Events chat_messages.py catalog and the stale MIGRATION_GUIDE.md are removed,chat_handoff_card.py and chat_shell_bar.py are removed,Dependent tests retargeted or deleted and suite green,No production references remain and Console rendering unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify import graphs at HEAD for the 7 target files 2. Delete widget/catalog/guide files 3. Update comment-only references package/__init__ re-exports and CSS census 4. Delete or retarget the ~8 dependent test files 5. Targeted test run diffed against the pre-change baseline 6. Task notes + AC tick
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deleted 13 files (~2.6k LOC): chat_message.py, Chat_Widgets/chat_message_enhanced.py, top-level chat_message_enhanced shim, chat_handoff_card.py, chat_shell_bar.py, Chat_Events/chat_messages.py + MIGRATION_GUIDE.md, plus 5 test files. SCOPE ADDITION at HEAD re-verification: tts_widget_index.py + its app_speech.py integration were part of the same family (only the two dead widgets ever registered; console_transcript does not) - the index, its import, and four dead iteration loops in app_speech were removed with widget_found auto-play now unconditional (the only reachable branch; semantics preserved). Also fixed at HEAD beyond review: Tests/TTS/test_console_speak_autoplay.py imported ChatMessage + tts_widget_index for a legacy-branch regression pin (removed test + imports), and test_message_feedback kept only its live DB tests. Census: 3 DEFAULT_CSS entries removed (keeps the no-stale-entries invariant). Verification: 176 passed / 1 failed where the 1 is the pre-existing environmental css-allowlist failure (8 unrelated widgets, red at baseline and in the main checkout). Remaining mentions are historical comments only (tts_events narrative, persona-buddy closure docstring, maturity comment).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
