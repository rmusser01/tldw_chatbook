---
id: TASK-34890
title: Delete the dead legacy chat-widget family
status: To Do
created_date: 2026-10-12 17:51
labels:
- simplification
- dead-code
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console transcript unification is complete but the previous generation of chat-message widgets was never carried out. About 2.1k LOC across 7 files is production-dead and misleads contributors.

Evidence (verified on dev df4985e): Verified on dev df4985e: zero production constructors for ChatMessageEnhanced and ChatMessage; console_transcript.py + ConsoleChatMessage is the sole live renderer (chat_screen.py:329 564). About 8 test files import the dead modules; Tests/UI/test_widget_css_consolidation.py:1198 enumerates ChatHandoffCard in a CSS census. Continues the task-34433 removal series. No ADR protects these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Widgets/Chat_Widgets chat_message.py and chat_message_enhanced.py and its top-level alias shim are removed,Event_Handlers/Chat_Events chat_messages.py catalog and the stale MIGRATION_GUIDE.md are removed,chat_handoff_card.py and chat_shell_bar.py are removed,Dependent tests retargeted or deleted and suite green,No production references remain and Console rendering unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
