---
id: TASK-34891
title: Delete dead event catalogs and handler modules
status: To Do
created_date: 2026-10-12 17:52
labels:
- simplification
- dead-code
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The central message-catalog convention in Event_Handlers was aspirational and live dispatch never adopted it. Several catalogs and handler modules are import-graph verified dead and deletable with no wiring changes (~1.2k LOC).

Evidence (verified on dev df4985e): Verified zero importers in production and tests for: Event_Handlers/Chat_Events/chat_messages.py (445 LOC plus MIGRATION_GUIDE.md referencing nonexistent chat_events_refactored.py), Media_Creation_Events/swarmui_events.py (236), Audio_Events/dictation_integration_events.py (106), Event_Handlers/eval_db_operations.py (336; only its own test instantiates it), ingest_status_helper.py (126). Live dictation uses ConsoleDictationEvent in UI/Console_Modules/dictation.py:176. chat_messages.py overlaps TASK-34890; coordinate ordering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 swarmui_events.py and dictation_integration_events.py removed,eval_db_operations.py and its path test removed,ingest_status_helper.py and the ingest_events.py re-export shim removed,zero-byte llm_management_events_llamafile and _llamacpp removed,unused Audio_Events message classes pruned keeping VoiceCommandEvent and AudioLevelUpdateEvent,App boot and live event flows unchanged
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
