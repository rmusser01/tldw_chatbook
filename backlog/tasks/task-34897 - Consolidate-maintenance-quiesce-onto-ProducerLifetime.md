---
id: TASK-34897
title: Consolidate maintenance quiesce onto ProducerLifetime
status: To Do
created_date: 2026-10-12 17:52
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ProducerLifetime in Backup_Recovery/runtime_producer_lifetime.py already expresses the close-admission drain resume machine with depth tracking but ~20 owners hand-roll private quartets and three near-duplicate decorator variants have already diverged.

Evidence (verified on dev df4985e): Hand-rolled copies verified at: TTS_Events/tts_events.py:586-620+735-780 STTS_Events/stts_events.py:673-740 Scheduling/scheduler/loop.py:201-240 Notes/notes_sync_runtime.py:1799-1877 Utils/db_status_manager.py:48-75 Chat/console_chat_controller.py:3569 Chat/console_prompt_queue_coordinator.py:177 Chat/console_image_edit_operations.py:99 UI/Console_Modules/dictation.py:873-921 UI/Navigation/audio_cpp_model_handoff.py:165 UI/stts_profile_library.py:1985 Evals/eval_orchestrator.py:92-107 Writing_Interop/writing_scope_service.py:164 Sync_Interop/server_sync_service.py:113 Logging_Config.py:269 app_ingest_queue.py:593. runtime_maintenance.py binds by name convention so the mixin must keep exposing the quartet names. No ADR pins the protocol. About 50 test files touch drain semantics; hooks preserve them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared mixin or decorator extends ProducerLifetime with an optional readiness hook and quartet name binding,15-20 hand-rolled quartets collapsed,Owner-specific readiness predicates remain hooks,TTS and STTS decorator variants unified with their kind gates preserved,Maintenance and backup settle tests pass
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
