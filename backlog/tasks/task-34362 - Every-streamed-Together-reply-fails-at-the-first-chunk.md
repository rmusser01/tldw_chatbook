---
id: TASK-34362
title: Every streamed Together reply fails at the first chunk
status: Done
assignee:
  - '@claude'
created_date: '2026-10-04 18:00'
updated_date: '2026-10-04 18:01'
labels:
  - providers
  - streaming
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live on dev 7d155170dc (2026-10-04): a non-streamed Together chat returns normally, but every streamed one raises "Hosted Chat stream choice is malformed." Together's stream choices carry a null logprobs key, which the Together record did not allow; and Together only sends a usage chunk when the request asks for it, which the record did not ask for, so a stream that got past the first problem would still end in "stream terminated before required metadata". Streaming is Together's default, so Together chat failed on every reply. Found by the TASK-33640 keyed capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A streamed Together reply completes, with a finish reason and usage, against the real API
- [x] #2 Non-streamed Together replies are unchanged
- [x] #3 The record change cites the live capture that proves it, and the capture replays offline
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Together record (provider_registry.py) gains choice_allowances={"logprobs"} and stream_include_usage=True, with a comment citing Tests/fixtures/cloud_live/together.json. Evidence, live on 2026-10-04 through the engine's real handler and transport: dev 7d155170dc gave non-stream 'ok' and stream HostedChatProtocolError 'Hosted Chat stream choice is malformed.'; this branch gave non-stream 'ok' and stream OK (finish=stop, usage present). A probe confirmed Together honours stream_options.include_usage with a final usage chunk that has empty choices. The fixture was recaptured with the flag (14 stream events, no uncovered keys) and replays in test_live_capture_replay.py (all pass); it holds no key fragment.
<!-- SECTION:NOTES:END -->
