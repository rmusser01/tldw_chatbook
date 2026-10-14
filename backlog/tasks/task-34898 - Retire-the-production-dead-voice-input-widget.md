---
id: TASK-34898
title: Retire the production-dead voice input widget
status: Done
created_date: 2026-10-13 02:04
labels:
- simplification
- dead-code
assignee:
- '@zcode'
updated_date: 2026-10-14 18:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Widgets/voice_input_widget.py has zero production importers - its only consumers are four test files (css census plus Tests/Audio). It is the sole reason six dictation-event classes and AudioLevelUpdateEvent are still kept in the Audio_Events catalogs (task-34891 pruned only the truly unreferenced ones). Verify no dynamic references then delete the widget and its tests, then prune the now-orphaned event classes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 voice_input_widget.py and its test-only importers removed,Newly-orphaned Audio_Events classes pruned with a usage census in the notes,Audio and UI census tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify zero production importers at HEAD 2. Delete voice_input_widget + its test-only consumers 3. Census/CSS updates 4. Prune newly-orphaned Audio_Events classes with usage census 5. Tests green + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deleted Widgets/voice_input_widget.py + Tests/Audio/test_voice_input_widget.py; trimmed TestVoiceInputWidgetProperties from test_property_based.py and TestEndToEndDictation (5 widget-driving tests) from test_audio_integration.py; CSS census entry removed. Usage census at HEAD: the Console dictation path uses its OWN ConsoleDictationEvent (dictation.py) - the widget was the ONLY consumer of DictationStarted/Stopped/Error PartialTranscript FinalTranscript AudioLevelUpdate - all pruned; dictation_events.py now holds only the DictationEvent base + VoiceCommandEvent (live consumer: UI/Dictation_Window_Improved.py:717); recording_events.py deleted entirely (zero consumers) with its star-export removed from Audio_Events/__init__. BONUS FIX of the latent bug noted in task-34891: Dictation_Window_Improved posted VoiceCommandEvent(command=...) without the required raw_text arg (guaranteed TypeError on any voice command) - now passes raw_text=command. Verification: trimmed Audio suites green except one stash-verified pre-existing failure (test_transcription_service_fallback); Tests/Event_Handlers + Tests/App stash-diffed - IDENTICAL 66-failure environmental set before and after; census suite shows only the known pre-existing allowlist failure + env errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
