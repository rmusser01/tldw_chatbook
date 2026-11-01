---
id: TASK-34935
title: Move-image-gen-demo-PIL-decode-off-the-UI-thread
status: Done
assignee: []
created_date: 2026-10-13 14:56
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: low
updated_date: 2026-10-13 19:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
image_gen_demo_screen._render_result (invoked via app.call_from_thread from the generation worker) does PILImage.open + thumbnail + Pixels.from_image on the UI thread (image_gen_demo_screen.py:132-135) — a 1024x1024 image can exceed 100ms, freezing the whole TUI exactly when the generated image appears. Demo screen, one-off freeze, hence low.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Decode/thumbnail/pixel conversion run in a thread with only the widget update on the loop
- [x] Demo still displays the generated image
<!-- AC:END -->


## Implementation Notes

- The generation already ran on a thread worker; only the decode was marshaled to the loop via call_from_thread. The split: `_decode_result` (PIL open + thumbnail + Pixels.from_image + meta text) runs on the worker inside its existing try/except so decode failures surface as the same status error; `_apply_result` is the UI-thread half doing the three widget updates.
- Tests: `Tests/UI/test_image_gen_demo_decode_offloop.py` (2 tests: decode returns pixels + meta from a real generated PNG; apply touches exactly the three widgets). 2/2 pass.
- ADR required: no — worker/thread split inside a demo screen.