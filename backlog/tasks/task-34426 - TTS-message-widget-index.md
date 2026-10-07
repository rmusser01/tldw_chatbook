---
id: TASK-34426
title: TTS message-widget index
status: Done
created_date: 2026-10-07 02:42
dependencies: []
updated_date: 2026-10-07 13:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 5 / F18d: TTS progress handlers run two full-app DOM queries plus linear id scans per event for widget types that are never mounted in production
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Handlers perform zero app.query calls when no widgets registered,Registered widget receives state update O(1)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 14 (T14)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Replaced the four TTS handlers' double full-DOM query walks + linear id scans with a weakref-backed message-widget index (new stdlib-only Widgets/Chat_Widgets/tts_widget_index.py — cycle-free shape, app_speech's module-scope tldw_chatbook.app import fences the direct direction): dict[str, list[weakref.ref]] keyed message_id_internal, identity-based register/unregister from both widget classes' on_mount/on_unmount (idempotent per instance, id-change watcher defensive), dead refs filtered on read with empty-key deletion on all mutation paths, event-loop thread-safety documented. Empty registry (the production case — only ConsoleChatMessage, a different class, is ever instantiated) early-returns before any DOM work; found-widget bodies byte-identical per handler; the old break removal is the sole semantic delta, mandated by the two-widgets-same-id requirement. One regression pin adapted to register via the public index API (assertions unchanged). Evidence: 10 TTS events with no mounted widgets 20 full-DOM queries -> 0 (cross-checked red-run). 9 new tests; 22 suites byte-identical to baseline. Files: tts_widget_index.py (new), app_speech.py, chat_message.py, chat_message_enhanced.py, Tests/App/test_tts_widget_index.py. Report: .superpowers/sdd/task-14-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
