---
id: TASK-33640
title: Live-capture every engine preset against its real API
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-09-30 04:00'
updated_date: '2026-10-04 18:49'
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
- [x] #6 Without keys, each preset's shipped URL is probed: its chat route exists, a bad key reaches the user as an authentication failure, public listings parse through discovery, and seeded models are still listed
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

No-key probe (user request): capture.py --no-auth probed 29 presets with shipped URLs (Azure/Cloudflare/Databricks need an account URL) with no key and a fake key, at no token cost. The results are committed under Tests/fixtures/cloud_live/noauth/ and pinned by Tests/LLM_Calls/test_noauth_probe_evidence.py. Findings, 2026-09-30:
- (1) Cloudflare blocks urllib's default User-Agent with 403 "error code: 1010" at Together, Cerebras, GMI, W&B, OpenCode Zen and Command Code. The app sends python-requests/2.32.5 and is NOT blocked (verified), but the retired capture_cloud.py would have been. The tool now sends the app's User-Agent.
- (2) Every chat route exists at its shipped URL.
- (3) A bad key answers 401/403, which maps to "authentication failed. Check the API key.", everywhere except Fireworks and GMI. Both check the model before the key (404 "model not found / inaccessible"); whether a real but bad Fireworks key surfaces as a 404 needs a live key.
- (4) All 12 public listings (SambaNova, NVIDIA, DeepInfra, Novita, Vercel, ZenMux, Kilo, Ollama Cloud, Nous, Venice, OpenCode Zen, Command Code; 7 to 429 models) parse through normalize_models_response.
- (5) All 30 seeded ids for OpenCode Zen and Command Code are still listed.
- (6) Venice answers a missing key with 402 (x402 payment protocol), and the others with 401/403.

Qodo round (9 findings, all fixed): (1) a capture with no successful round is no longer written or counted, and the replay test fails any such fixture; (2) Google-style docstrings on every public function; (3) HTTP error responses are closed via `with error:`; (4) isolated uncovered_keys unit tests cover every level plus allowances; (5) a 200 stream that is empty or lacks [DONE] fails replay (stream_verdict), and the capture flags it; (6) TLDW_LIVE_<KEY>_API_KEY_ENV_VAR names an alternate key variable that is read before the registry candidates, as the engine does with api_key_env_var (StepFun STEP_API_KEY, Meta MODEL_API_KEY); (7) the serialized fixture is scanned for the live credential and every occurrence redacted before writing; (8) overrides are validated by a Pydantic CaptureOverrides model (http(s) URL with no query/userinfo, single-line model and header values, env-var-name check), and a bad value skips the provider without echoing the value; (9) --keys-file goes through Utils/path_validation.validate_path_simple. A mutation of each of 1/6/7/9 turns its test red.

2026-10-04: Together captured with a real key (Tests/fixtures/cloud_live/together.json). It surfaced three bugs, fixed in the same PR: discovery rejected Together's bare-array listing and its 16 KB chat-template metadata (TASK-34361), and every streamed reply failed on logprobs and missing usage (TASK-34362). The record changes cite the fixture (AC #5 holds for Together). The other 15 key-only listings are still unseen.

2026-10-04: Fireworks captured (Tests/fixtures/cloud_live/fireworks.json). Plain and stream rounds were clean, and streamed usage arrives without asking. Tool calls failed both ways (TASK-34364, fixed in the same PR): extra index/name on non-streamed calls, and "id": null on streamed continuations. A real-format wrong key maps to 'authentication failed', closing the no-key probe's open Fireworks question.
<!-- SECTION:NOTES:END -->
