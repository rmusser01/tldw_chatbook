---
id: TASK-34935
title: Move-image-gen-demo-PIL-decode-off-the-UI-thread
status: To Do
assignee: []
created_date: '2026-10-13 14:56'
labels:
  - performance
  - ui-perf
  - wave-4-blocking-rebuilds
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
image_gen_demo_screen._render_result (invoked via app.call_from_thread from the generation worker) does PILImage.open + thumbnail + Pixels.from_image on the UI thread (image_gen_demo_screen.py:132-135) — a 1024x1024 image can exceed 100ms, freezing the whole TUI exactly when the generated image appears. Demo screen, one-off freeze, hence low.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decode/thumbnail/pixel conversion run in a thread with only the widget update on the loop,Demo still displays the generated image
<!-- AC:END -->
