# Moonshot Live Native-Tool UAT Fix Design

Date: 2026-08-13
Status: User-approved; root-cause evidence incorporated
Backlog task: [TASK-16074](../../../backlog/tasks/task-16074%20-%20Make-Moonshot-live-native-tool-continuation-pass.md)
Foundation task: [TASK-15676](../../../backlog/tasks/task-15676%20-%20Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md)
Architecture decision: [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md)

## Purpose

Correct the post-merge Moonshot Kimi K3 defect found by paid UAT: Chatbook's
real Console-to-AgentService native-tool path reports HTTP 502 before the first
tool call, even though the provider accepts the request. The displayed status
is Chatbook's generic mapping of an internal streaming protocol error, not an
HTTP 502 response from Moonshot.

The fix must preserve the native-tool and durable-continuation contracts. It
must not make the UAT pass by removing tools, bypassing the joined product path,
or replacing the expected tool call with a text-only answer.

## Verified Evidence

The supplied credential was read only from the ignored local file
`moonshot-api-key.txt`. It was not printed, logged, copied into a fixture, or
committed.

| Probe | Result | What it rules out |
| --- | --- | --- |
| `GET /v1/models` | HTTP 200; `kimi-k3` present | Invalid credential, base URL, or unavailable model |
| Minimal Kimi K3 text request | HTTP 200 | General Kimi K3 request failure |
| Minimal non-stream function-tool request | HTTP 200 with one tool call | Basic function-tool support |
| Minimal streaming function-tool request with usage | HTTP 200 SSE with tool delta and `[DONE]` | Streaming tool calls and `stream_options.include_usage` |
| Streaming request with Chatbook's initial calculator, datetime, and subagent schemas | HTTP 200 | The initial three tool schemas in isolation |
| Exact joined first-turn messages with the original five and full captured eight-tool catalogs | HTTP 200 | The complete composed request and aggregate tool schemas |
| Exact request fully consumed through Chatbook's strict Moonshot adapter | `HostedChatProtocolError` at `HostedChatStream._consume_event()` | The failure is response normalization, not request rejection |
| Redacted event-key trace | `choices`, `created`, `id`, `model`, `object`, `system_fingerprint` | Streaming rejects Moonshot's standard `system_fingerprint` field |
| Real repository live harness through Console, AgentService, and Moonshot | Synthetic HTTP 502 before a recorded tool call | The parser defect is reachable through the product path |

The exact defect is an asymmetric neutral-hosted response allowlist:
`normalize_hosted_chat_response()` already admits `system_fingerprint` on a
non-streaming response, while `HostedChatStream._consume_event()` rejects the
same field on a streaming event. Moonshot currently emits it on the first Kimi
K3 SSE event. The strict parser raises before consuming the event's tool-call
delta; the Console gateway then presents its default safe 502 error copy. No
production code has been changed at this stage.

## Goals

- Admit Moonshot's bounded `system_fingerprint` on hosted streaming events,
  matching the existing non-streaming contract.
- Pin that exact live event shape in deterministic parser and joined Console
  native-tool regressions.
- Correct the narrowest owner of the invalid value while preserving Moonshot,
  AgentService, Console, and durable continuation contracts.
- Prove the joined paid path performs exactly one calculator call, sends its
  result back to Kimi K3, and receives the required final marker.
- Keep credentials and complete provider payloads out of persistent artifacts
  and diagnostic output.

## Non-Goals

- No new provider, model, API mode, endpoint, SDK, or configuration surface.
- No change to the durable checkpoint schema or Sync-v2 contract.
- No weakening of native tool selection, approval, execution, or replay.
- No provider-hosted tools and no expansion of the default Chatbook tool set.
- No broad provider refactor or unrelated provider behavior change.
- No paid network request in the default test suite.

## Root-Cause Method

The investigation captured the final outbound Moonshot request object in
memory at the existing provider preparation/transport seam, after all Console,
AgentService, gateway, and Moonshot builder transformations but before network
I/O. It then compared only allowlisted structure and non-reversible content
digests with passing direct requests.

The captured object was compared programmatically with a known-passing request
assembled from the same user prompt and tool schemas. Diagnostics emitted only
an allowlisted structural summary: request key names, value types, collection
counts, bounded lengths, and non-reversible digests where equality had to be
checked. They did not print message content, tool arguments, the full payload,
headers, the credential, or raw provider error bodies.

The evidence proceeded from the smallest high-information boundary:

1. Model, streaming flags, usage options, messages, and the initial tool schemas
   matched already-passing direct requests.
2. Each additional runtime tool schema, their combined five-tool catalog, and
   the exact full captured catalog returned HTTP 200.
3. Fully consuming the same request through Chatbook reproduced
   `HostedChatProtocolError` at the stream event allowlist.
4. A keys-only trace isolated `system_fingerprint` as the sole unexpected
   top-level field. No response values or raw provider body were printed.

The comparison was evidence collection, not a permanent generic
payload-logging feature. Its temporary `/tmp` harness was deleted after the
root cause was isolated; no repository seam was introduced.

## Correction Boundary

The production change belongs only in
`tldw_chatbook/LLM_Calls/hosted_chat.py`, whose neutral stream parser owns the
top-level OpenAI-shaped event allowlist. It will accept
`system_fingerprint` only as optional bounded metadata, preserve strict
rejection of unknown keys and malformed/oversized values, and leave tool,
reasoning, usage, finish, provider, and request construction behavior
unchanged.

The outer regression will add the exact metadata field to Moonshot's scripted
SSE in the existing real Console HTTP/native-tool test. No downstream layer
will discard the field merely to bypass validation, and unrelated providers
retain their current prepared requests.

## Error And Privacy Contract

- Provider HTTP/network/malformed-response failures continue to use the
  existing typed, redacted error path.
- Authorization headers and API keys are never part of captured request data.
- Complete outbound payloads, prompts, reasoning, tool arguments, and raw
  provider bodies are never written to logs, tracebacks, fixtures, snapshots,
  or committed files.
- Automated assertions use synthetic canaries and structural comparisons so a
  failure cannot reveal the paid UAT prompt or credential.
- The paid test remains doubly gated by the explicit live-test opt-in and a
  nonblank credential.

## Verification

The implementation must use strict RED-to-GREEN evidence:

1. A deterministic parser regression first reproduces rejection of a bounded
   `system_fingerprint` streaming event without network access.
2. A focused joined scripted-HTTP test proves Console/AgentService consumes
   that exact Moonshot event, executes the native calculator call, continues
   with its result, and completes.
3. Existing focused Moonshot hosted transport, provider gateway, AgentService,
   Console continuation, and privacy/redaction tests remain green.
4. The paid Moonshot harness runs only after local evidence is green and proves
   exactly one calculator execution, provider continuation with the tool
   result, and the required final marker.
5. A final repository search and diff audit prove the local key and full live
   payload were not added to tracked files.

Per the user's test-scope instruction, verification is limited to touched files
and directly related Moonshot/hosted/AgentService/Console continuation paths;
no full-suite or broad-directory sweep is required.

## ADR Check

ADR required: no.

ADR-063 already assigns provider policy, neutral hosted wire mechanics,
durable continuation ownership, redaction, and replay boundaries. This task is
a compatibility bug fix inside those accepted boundaries and introduces no new
architecture decision.
