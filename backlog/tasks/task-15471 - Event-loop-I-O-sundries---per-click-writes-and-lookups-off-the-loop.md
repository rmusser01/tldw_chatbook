---
id: TASK-15471
title: Event-loop I/O sundries: per-click writes and lookups off the loop
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch of small, individually-verified per-click blockers (July task-261 precedent), from the audit: Console conversation star toggle runs a sync read + write transaction (fsync) on the loop (`UI/Console_Modules/workspace.py:1946-2013`; browser refresh also reads starred ids sync at `chat_screen.py:9914`); Study create-card/add-topic write ChaChaNotes synchronously in handlers (`Event_Handlers/Study_Events/study_events.py:46-138`) and the Study dashboard fallback queries run on resume; emoji picker rewrites its recents JSON per selection (`Widgets/emoji_picker.py:74-113`); TTS export does `shutil.copy2` (MBs) + `json.dump` on the loop (`Event_Handlers/TTS_Events/tts_events.py:2510-2550`); collections keyword delete runs 2xN redundant SELECTs on the loop before its properly-threaded delete (`Event_Handlers/collections_tag_events.py:142-189`); `chat_message_enhanced.handle_save_image` writes multi-MB images sync (`:634-650`; the Console-side equivalent in `UI/Console_Modules/message.py:1501-1626` is already threaded — copy it); the enhanced file picker's search stats every directory entry twice per keystroke with no debounce (`Widgets/enhanced_file_picker.py:650/:686/:708-715`); CodeRepoCopyPaste reads whole files per tree-node click (`UI/CodeRepoCopyPasteWindow.py:716/:901`); ChatbookExportManagement globs + reads manifests per refresh (`:496`).

Fix direction: to_thread / debounce / dedupe per site, smallest-diff first; no behavioral changes. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed site is threaded, debounced, or deduped — or explicitly justified in the notes
- [ ] #2 Behavior unchanged across the touched surfaces (targeted tests where they exist)
- [ ] #3 Spot latency evidence recorded for the star toggle and file-picker typing
<!-- AC:END -->
