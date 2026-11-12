---
id: TASK-34891
title: Delete dead event catalogs and handler modules
status: Done
created_date: 2026-10-12 17:52
labels:
- simplification
- dead-code
priority: high
assignee:
- '@zcode'
updated_date: 2026-10-12 19:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The central message-catalog convention in Event_Handlers was aspirational and live dispatch never adopted it. Several catalogs and handler modules are import-graph verified dead and deletable with no wiring changes (~1.2k LOC).

Evidence (verified on dev df4985e): Verified zero importers in production and tests for: Event_Handlers/Chat_Events/chat_messages.py (445 LOC plus MIGRATION_GUIDE.md referencing nonexistent chat_events_refactored.py), Media_Creation_Events/swarmui_events.py (236), Audio_Events/dictation_integration_events.py (106), Event_Handlers/eval_db_operations.py (336; only its own test instantiates it), ingest_status_helper.py (126). Live dictation uses ConsoleDictationEvent in UI/Console_Modules/dictation.py:176. chat_messages.py overlaps TASK-34890; coordinate ordering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 swarmui_events.py and dictation_integration_events.py removed,eval_db_operations.py and its path test removed,ingest_status_helper.py and the ingest_events.py re-export shim removed,zero-byte llm_management_events_llamafile and _llamacpp removed,unused Audio_Events message classes pruned keeping VoiceCommandEvent and AudioLevelUpdateEvent,App boot and live event flows unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify import graphs for each catalog/module at HEAD 2. Delete swarmui_events dictation_integration_events eval_db_operations ingest_status_helper ingest_events shim and the two 0-byte files 3. Prune unused classes from Audio_Events catalogs keeping VoiceCommandEvent and AudioLevelUpdateEvent 4. Fix __init__ re-exports and the one dependent test 5. Targeted test run diffed vs baseline 6. Notes + AC tick
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deleted: Media_Creation_Events package (swarmui_events catalog + its re-exporting __init__), Audio_Events/dictation_integration_events.py, eval_db_operations.py (+ its path test), ingest_status_helper.py (finished-migration compat layer; its one dependent test in test_task12 deleted with it, the two live tests kept), ingest_events.py re-export shim (zero importers; the state-ownership guard now asserts the file stays absent instead of scanning its text), and the two 0-byte llm_management_events_{llamafile,llamacpp}.py files. Pruned from Audio_Events: DictationPaused/Resumed/StateChange + Recording Started/Stopped/Error + AudioDeviceChanged (zero refs incl. tests). DEVIATION from AC: the review claimed only VoiceCommandEvent and AudioLevelUpdateEvent were used - wrong at HEAD: Widgets/voice_input_widget.py uses Started/Stopped/Error/VoiceCommand/Partial/Final/AudioLevel. That widget is itself production-dead (test-only importers) so kept those classes and pruned only the truly unreferenced ones; voice_input_widget retirement filed as a note for a future task. Latent bug spotted in passing (NOT fixed, out of scope): UI/Dictation_Window_Improved.py:718 posts VoiceCommandEvent(command=...) without the required raw_text arg. Verification: edited tests green; Tests/Event_Handlers green; Tests/Audio failures are missing-env (TLDW_NATIVE_DUPLEX_ROOT) and test_application_state_ownership failures pre-exist with changes stashed (7) and in the untouched main checkout (8).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
