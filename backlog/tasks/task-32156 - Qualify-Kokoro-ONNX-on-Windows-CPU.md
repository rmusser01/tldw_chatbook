---
id: TASK-32156
title: Qualify Kokoro ONNX on Windows CPU
status: To Do
assignee: []
created_date: '2026-09-09 05:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
macOS and Linux ARM evidence does not establish Windows installation, path, handle or physical playback behavior. Existing task-13208 covers audio.cpp rather than Kokoro.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A fresh supported Windows Python installation loads explicit pinned Kokoro ONNX model and voice assets without undocumented manual fixes.
- [ ] #2 Real Lab and repeated Console speech play completely through the native Windows device and have independent content and provenance records.
- [ ] #3 Cancellation overlaps actual native inference, close joins native work, a successor succeeds, and paths or handles remain usable after cleanup.
<!-- AC:END -->
