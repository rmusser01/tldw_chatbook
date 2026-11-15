---
id: TASK-41034
title: "Fresh-process app tests fail at random on 'TTS shutdown cleanup failed (CancelledError)'"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- backup-recovery
- tts
- flaky
priority: medium
references:
- tldw_chatbook/TTS/TTS_Generation.py
- Tests/Backup_Recovery/test_runtime_startup_handoff.py
- Tests/Backup_Recovery/test_rag_projection_capture.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Several Backup_Recovery tests start the app in a fresh child process and shut it down. On dev ecad404bd4 (2026-11-15) they fail at random, because the child's TTS shutdown raises `RuntimeError: TTS shutdown cleanup failed (CancelledError)` (`TTS/TTS_Generation.py:3896`, from `_sanitized_shutdown_error`) and the child exits non-zero:
- **`test_runtime_startup_handoff.py::test_actual_app_startup_returns_before_ordinary_admission`.** One of its three params (resume, cancel and audio) failed in each of two runs, a different one each time: `[cancel]` at `-n 4` and `[resume]` serially.
- **`test_rag_projection_capture.py::test_public_native_projection_capture`.** Three params failed this way at `-n 4`, two more timed out, and `test_rag_projection_lifetimes.py::test_native_admission_and_positive_retirement` failed in the same run.

Earlier reports agree:
- TASK-36948.1's implementer saw 1-2 startup-handoff params red in 4 of 4 runs at 5ca7ec6a03. They traced a second cause: `settle_producers` does not settle a console-attention preparation read (`list_console_unseen_marks` on an `asyncio.to_thread` worker), so the drain fails.
- The 2026-11-12 Phase 3 verification saw the rag_projection CancelledError on dev too.

Why it matters: a cancellation during TTS cleanup is reported as a shutdown failure. In the app, that may turn an ordinary exit into an error. In tests, every covering run has to subtract these failures by hand.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A CancelledError during TTS cleanup at app shutdown is handled as cancellation, not reported as a cleanup failure; the task notes say whether the real app exit was affected
- [ ] #2 test_actual_app_startup_returns_before_ordinary_admission and the two rag_projection files pass 5 of 5 runs at `-n 4`
- [ ] #3 The task notes name each cause found, including the attention-preparation read, and whether it was a product or a test defect
<!-- AC:END -->
