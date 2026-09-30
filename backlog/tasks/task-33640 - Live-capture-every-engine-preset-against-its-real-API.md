---
id: TASK-33640
title: Live-capture every engine preset against its real API
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-09-30 04:00'
updated_date: '2026-09-30 04:08'
labels:
  - providers
  - live
  - engine
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
About 35 engine presets ship with allowances derived from public documentation, marked provisional until a live capture. The capture tool and live probe cover only Together, Cerebras and Fireworks. With keys arriving, every preset needs a raw capture of its real responses (model listing, plain chat, a tool call, a stream) so each record's allowances can be confirmed or amended from evidence rather than guessed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One command captures raw model-listing, plain, tool-call and streamed responses for every engine preset whose key is available, and cleanly skips the rest
- [x] #2 Keys come only from the environment or a user-owned keys file and never reach a fixture, log or printed line
- [x] #3 Per-account presets (Azure, Cloudflare, Databricks) take their URL and model from the environment
- [x] #4 Each capture replays through the engine's real parser offline, and a report names every preset that parses and every unknown field that does not
- [ ] #5 Allowance changes made from captures cite the fixture that proves them
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Registry-driven capture tool (Tests/fixtures/cloud_live/capture.py): every engine preset, requests shaped like the engine's, keys from env or ~/.config/tldw-live/keys.env, per-account URL/model/header overrides, raw fixtures + uncovered-key report.
2. Registry-wide offline replay test under each preset's real record; failures name the uncovered keys.
3. Retire the three-provider capture_cloud.py and its fixture flip.
4. Capture every preset with a key, amend allowances citing the fixture, re-replay.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Harness (PR 1 of the task): Tests/fixtures/cloud_live/capture.py replaces the three-provider longtail/capture_cloud.py. It is registry-driven (every engine preset except custom-hosted) and imports only provider_registry, never config. Requests mirror the engine per record: base URL + suffix, api-key vs Bearer, max_tokens_key, stream_options when asked, tool_choice only where payload_flags allow, extra_body_fields, record timeout. Keys come from the environment or a user-owned keys file (~/.config/tldw-live/keys.env) and exist only in the request header at call time. Per-account URLs are stored as <per-account>; listings keep a count + 20-entry sample. After each capture it prints the response key names outside the record's allowances. Tests/LLM_Calls/test_live_capture_replay.py replays every capture under its real record through the engine handler path, and a failing replay names the uncovered keys (checked with a synthetic capture). Tests/LLM_Calls/test_live_capture_tool.py drives the tool against a loopback server: Azure api-key header + max_completion_tokens + redacted URL, Ollama Cloud without tool_choice, Nous without a tool round, clean skips, and the canary key absent from output and fixture. The old cloud fixture flip in test_inference_cloud_presets.py is superseded and removed. Remaining (AC 5): run captures as keys arrive and amend allowances from the fixtures.
<!-- SECTION:NOTES:END -->
