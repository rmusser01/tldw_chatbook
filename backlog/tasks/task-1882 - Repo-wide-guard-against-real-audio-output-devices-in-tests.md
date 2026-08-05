---
id: TASK-1882
title: 'Repo-wide guard against tests constructing real audio output devices'
status: To Do
assignee: []
created_date: '2026-08-02 12:00'
labels: [tests, audio, hygiene]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A structurally valid WAV fixture caused an automated test to open a REAL PortAudio output
device — and pass (streaming-sink Task-4 fix round; live callback confirmed). The narrow guard now
in `Tests/conftest.py` patches the sink's sounddevice import with a `real_audio_device` opt-out
marker, but nothing stops a future test reaching `sounddevice.OutputStream` through any other
path. Origin: streaming-sink Task-4 re-review ruling (narrow guard required now, repo-wide guard
as its own task).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Constructing a real `sounddevice.OutputStream`/`pyaudio` output stream in any test without the `real_audio_device` marker fails the test loudly.
- [ ] #2 The marker opt-out works and is documented for test authors.
- [ ] #3 The existing suite passes unchanged under the guard.
<!-- AC:END -->
