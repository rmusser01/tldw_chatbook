# Moonshot Live Native-Tool UAT Fix Design

Date: 2026-08-13
Status: Written-spec review approved; awaiting user approval for planning
Backlog task: [TASK-16074](../../../backlog/tasks/task-16074%20-%20Make-Moonshot-live-native-tool-continuation-pass.md)
Foundation task: [TASK-15676](../../../backlog/tasks/task-15676%20-%20Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md)
Architecture decision: [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md)

## Purpose

Correct the post-merge Moonshot Kimi K3 defect found by paid UAT: Chatbook's
real Console-to-AgentService native-tool path receives a provider HTTP 502
before the first tool call, even though smaller requests against the same
credential, endpoint, model, streaming protocol, and tool schemas succeed.

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
| Real repository live harness through Console, AgentService, and Moonshot | HTTP 502 before a recorded tool call | The joined composed request remains defective |

The exact offending field or interaction is not yet established. The remaining
surface is the final first-turn request as composed by the joined product path,
especially its messages and any request policy added above the already-proven
tool-schema builder. This uncertainty is the investigation target, not an
implementation placeholder. No production code has been changed at this stage.

## Goals

- Identify the exact difference between Chatbook's failing first provider
  request and the smallest equivalent passing request.
- Pin that difference in a deterministic automated regression at the real
  provider preparation/transport boundary.
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

## Diagnostic Design

The investigation will capture the final outbound Moonshot request object in
memory at the existing provider preparation/transport seam. Capture occurs
after all Console, AgentService, gateway, and Moonshot builder transformations,
but before authorization headers are attached or network I/O begins.

The captured object will be compared programmatically with a known-passing
request assembled from the same user prompt and tool schemas. Diagnostics may
emit only an allowlisted structural summary: request key names, value types,
collection counts, bounded lengths, and non-reversible digests where equality
must be checked. They must not print message content, tool arguments, the full
payload, headers, the credential, or raw provider error bodies.

The comparison proceeds from the smallest high-information boundary:

1. Assert that model, streaming flags, usage options, tool choice, and tool
   schema structure match the already-passing direct request.
2. Compare message roles, counts, content shapes, and bounded sizes without
   logging their values.
3. If needed, remove or restore one captured request component at a time in a
   local deterministic harness to identify the first discriminating field.
4. Use another paid request only after the candidate is isolated locally, to
   confirm that one difference changes the provider outcome.

The comparison is evidence collection, not a permanent generic payload-logging
feature. Any test seam introduced must remain narrow, injectable, and inert in
normal execution.

## Correction Boundary

The regression will enter through the outermost practical product path that
reproduces the composition defect and assert the final prepared Moonshot
request. The production correction belongs to the earliest existing component
that semantically owns the invalid value:

- Moonshot-only policy stays in the Moonshot request builder.
- Provider-neutral hosted Chat-Completions normalization stays in the neutral
  hosted wire boundary.
- Agent/runtime message or tool-catalog composition stays in its current
  provider-neutral owner.

No downstream layer will silently discard a valid upstream value merely to
make Moonshot accept the request. Unrelated providers must retain their current
prepared requests.

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

1. A deterministic regression first reproduces the exact invalid prepared
   request property without network access.
2. A focused joined test proves Console/AgentService reaches the Moonshot
   preparation boundary with the corrected request and still exposes native
   function tools.
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
