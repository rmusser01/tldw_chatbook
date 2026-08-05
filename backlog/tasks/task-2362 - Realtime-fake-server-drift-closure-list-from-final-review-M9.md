---
id: task-2362
title: 'Realtime: fake-server drift closure list from final review M9'
status: To Do
assignee: []
created_date: '2026-08-04'
labels: [realtime, test-quality]
dependencies: []
priority: low
---

## Description (the why)

The scripted fake WS server encodes the live-probed ground truth, but the V4 final review
(M9) listed residual drift: `input_audio_buffer.committed` is dispatched in production but
never emitted by any fake script (deleting the dispatch stays green); `voice` under
session.audio.output is sent but unasserted; `whisper-1` is documented live-confirmed but
the fake only checks transcription is enabled; the probe script's docstring still describes
a pre-GA session shape and sends no audio, so it cannot reproduce the input_audio_buffer
observations its header attributes to it; the singular `input_token_details` live-claim
lives only in provider_usage.py's comment.

## Acceptance Criteria (the what)

- [ ] Each listed drift item is either asserted by the fake/probe or explicitly documented
      as intentionally uncovered.
- [ ] Deleting the `input_audio_buffer.committed` dispatch fails a test.
- [ ] The probe script's docstring matches what it actually sends.
