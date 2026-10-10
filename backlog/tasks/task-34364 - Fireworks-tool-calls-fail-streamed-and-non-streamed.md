---
id: TASK-34364
title: 'Fireworks tool calls fail, streamed and non-streamed'
status: Done
assignee:
  - '@claude'
created_date: '2026-10-04 18:48'
updated_date: '2026-10-05 00:44'
labels:
  - providers
  - tools
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live on 2026-10-04 with a real key, every Fireworks reply that calls a tool failed in the app. Non-streamed: each tool call object carries index and name: null beside id/type/function, and built-in presets accept exactly id/type/function, so the reply failed as "Fireworks returned a malformed successful response." Streamed: continuation deltas repeat "id": null instead of omitting it, which the stream parser read as the call's identity changing ("Hosted Chat stream tool identity changed."). Plain replies worked both ways. Found by the TASK-33640 keyed capture. Its uncovered-keys report did not look at tool-call objects, so it reported none.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Fireworks tool call completes against the real API, streamed and non-streamed
- [x] #2 A provider entry can allow named extra keys on a tool call; they are validated and dropped, and any other extra still fails
- [x] #3 A streamed continuation whose id is null keeps its call; a continuation with a different id still fails
- [x] #4 The capture's uncovered-keys report covers tool-call objects
- [x] #5 The Fireworks capture replays offline and its entry cites it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two fixes, both proven live against Fireworks on 2026-10-04 through the engine's real handler and transport. Before: non-streamed tool call failed with 'Fireworks returned a malformed successful response.'; streamed tool call failed with 'Hosted Chat stream tool identity changed.'. After: both return get_weather({"city": "Tokyo"}); plain replies still work both ways, with usage.

1. Non-streamed: new ProviderRecord.tool_call_allowances (provider_registry.py), passed by hosted_provider_engine into normalize_hosted_chat_response, then to _normalize_tool_calls. A strict record now checks call-object extras with the existing _check_level_extras: named keys must pass the value rule and are dropped (never passed through); anything else still fails closed. The tolerant custom profile is unchanged. Fireworks allows {index, name}, citing Tests/fixtures/cloud_live/fireworks.json.
2. Streamed: hosted_chat._consume_tool_deltas treats an explicit "id": null on a continuation delta as not sent. Null claims no identity and the call is keyed by index. A different non-null id still raises 'identity changed'. This is a parser fix for every provider, not a Fireworks allowance.

The capture's uncovered_keys now reports a tool_call level (Tests/fixtures/cloud_live/capture.py). It printed 'none' for this capture because it never looked at call objects. Tests are in Tests/LLM_Calls/test_hosted_chat_allowances.py and test_live_capture_tool.py; the Fireworks fixture replays. Mutation-checked: removing the Fireworks allowance fails the replay, and restoring the old null-id check fails the stream test. Also answered the no-key probe's open question: a real-format wrong key reaches the user as 'Fireworks authentication failed. Check the API key.' The model list is one page of 20 serverless models (no pagination fields).

Independent review round (Qodo was out of credits and CodeRabbit skips dev-targeted PRs, so a fresh agent reviewed the PR at the owner's choice). It found no correctness or security bugs. Acted on:
- The streamed tool shape was never in a fixture: capture.py now records a streamed tool round (tool_stream_events, statuses.tool_stream; five requests per provider). Fireworks and Together were recaptured. Fireworks' fixture now holds 4 continuations with an explicit "id": null, and test_captured_tool_stream_replays_to_a_tool_call replays them to get_weather. Restoring the old null-id check fails that replay. Together's streamed tool calls work, tested for the first time.
- uncovered_keys now also reports a stream_tool_call level (delta.tool_calls[] keys).
- Added an Args docstring for normalize_hosted_chat_response; the census test now pins tool_call_allowances (fireworks only); added a test for the allowance subtraction in uncovered_keys; the metadata fallback test covers the 16 KiB serialized and long-key bounds.
Not changed (nits): a null type/name on a continuation still fails, since no provider has been seen sending that; the fallback still drops a model's whole metadata, so the 7 Vercel models lose the inferred-vision hint (documented trade-off).

Review follow-up (owner rule: fix every finding, minor included): a null type or function.name on a streamed continuation now counts as not sent, like a null id; a different non-null value still fails ('type changed' / 'name changed'). The downstream accumulator (console_provider_gateway._ToolCallAccumulator._merge) reads id, type and name by truthiness, so nulls in the visible frame are ignored. Mutation-checked.

Qodo round (2026-10-05): (1) uncovered_keys no longer crashes on a tool_calls value that is not a list; (2) a complete streamed tool round now counts as a successful round, in capture.py's write gate and in the replay's has_usable_round, so a tool-stream-only capture is written and recognised. Both have tests; removing the gate change fails its test.

Governance clarification for the reconciliation merge: existing ADR-179 now states the record-scoped non-streamed tool-call extra/value/drop contract and nullable established-stream identity rule. Provider-owned response normalization and all strict required-field/terminal/resource checks remain. No new ADR or duplicate transport is introduced.
<!-- SECTION:NOTES:END -->
