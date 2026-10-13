---
id: TASK-34937
title: Stop-spawning-webbrowser-open-and-pyperclip-on-the-UI-thread
status: To Do
assignee: []
created_date: '2026-10-13 14:57'
labels:
  - performance
  - ui-perf
  - wave-4-blocking-rebuilds
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Link handlers spawn the OS open command on the UI thread (settings_screen.py:23600, console_transcript.py:1964, library_media_controller.py:4204, library_collections_controller.py:1583) and the inspector copy spawns pbcopy (console_conversation_inspector.py:1910). Each spawn can cost 50-150ms of UI freeze.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Link and copy actions dispatch subprocess spawns via to_thread or executor,No subprocess spawn on the event loop in the named handlers
<!-- AC:END -->
