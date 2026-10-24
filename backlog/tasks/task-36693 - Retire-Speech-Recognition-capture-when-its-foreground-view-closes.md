---
id: TASK-36693
title: Retire Speech Recognition capture when its foreground view closes
status: To Do
created_date: 2026-10-24 17:33
labels:
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Leaving Speech Recognition removes its Stop control while the widget-owned recorder and real processing thread remain active, with callbacks still targeting the detached widget. End that foreground capture through the existing owner lifecycle.

### Evidence and scope

Verified after PR #3228 merged, against dev `96cae7eec00473bbfb5687ccfd0685eba2785add`. Actual STTSWindow request_view(effects) detaches the real DictationWindow. Injected recorder remains recording with stop_calls=0; real DictationProcessor remains alive and cancellation unset. Explicit Stop before the same navigation retires both. Physical microphone hardware was not opened.

Book basis: PDF145–146 / printed136–137 describes widget message ownership; PDF335 / printed326 demonstrates cancellation-aware publication. The foreground capture lifetime follows this application’s exclusive view ownership and disappearing Stop control, rather than an explicit book microphone policy.

Affected owners: [Dictation_Window_Improved.py](../../tldw_chatbook/UI/Dictation_Window_Improved.py), [STTS_Window.py](../../tldw_chatbook/UI/STTS_Window.py), [dictation_service_lazy.py](../../tldw_chatbook/Audio/dictation_service_lazy.py).

### Existing ownership and decisions

TASK-1283 is Done and repairs the Console capture owner’s processor retirement. TASK-15470 owns settings-write flushing; TASK-16266 owns partial-mount view switching. TASK-31993.17/.21 own backup-capture settlement. None owns ordinary departure from the widget-owned foreground Dictation session.

Related tasks: [TASK-1283](../tasks/task-1283%20-%20_release-frees-the-microphone-but-never-stops-the-DictationProcessor-thread-leaking-it-on-every-abandon.md), [TASK-15470](../tasks/task-15470%20-%20Config-persistence---batch-writes-and-take-them-off-the-click-path.md), [TASK-16266](../tasks/task-16266%20-%20Harden-Speech-view-switching-and-first-run-test-seam.md), [TASK-31993.17](../tasks/task-31993.17%20-%20Settle-dictation-dispatch-before-live-backup-capture.md), [TASK-31993.21](../tasks/task-31993.21%20-%20Settle-speech-handlers-and-shared-TTS-service-for-live-capture.md).

ADR required: no.
ADR paths: [025-shared-stt-artifacts-and-runtime-routing](../decisions/025-shared-stt-artifacts-and-runtime-routing.md), [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md).
Reason: Routine retirement of the existing view-owned service and callbacks, using existing session ownership. Re-evaluate ADR need if background capture or ownership policy changes.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leaving or unmounting Speech Recognition retires the capture and processor owned by that view, so returning cannot leave an inaccessible earlier recorder running.
- [ ] #2 Callbacks belonging to the departed view cannot publish into detached or replacement controls; teardown does not block the UI with a processing-thread join.
- [ ] #3 Ordinary explicit Stop retains its existing completion semantics; visiting another Speech view without starting capture has no recording side effect.
- [ ] #4 Mounted active-leave and explicit-Stop controls verify recorder retirement and the real owned processing thread using private profiles and injected audio I/O.
<!-- AC:END -->
