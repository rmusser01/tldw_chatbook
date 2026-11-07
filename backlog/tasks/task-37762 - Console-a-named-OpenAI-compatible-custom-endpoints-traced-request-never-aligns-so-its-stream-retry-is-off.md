---
id: TASK-37762
title: >-
  Console: a named OpenAI-compatible custom endpoint's traced request never
  aligns, so its stream retry is off
status: Done
assignee:
  - '@claude'
created_date: '2026-10-29 17:45'
updated_date: '2026-11-07 01:59'
labels:
  - console
  - trace
  - streaming
  - local-models
  - p2
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A named custom endpoint of the OpenAI-compatible family (`custom-ep:<slug>`, executed by the hosted engine under the `custom-hosted` key) records every provider call's traced request surface as one `provider_overlay` omission with reason `alignment_mismatch`, instead of the saved user revision that Custom, vLLM and llama.cpp calls record. Measured 2026-10-29 with trace capture on (the default), reading `console_trace_surface_nodes` in an isolated profile: the same on origin/dev 38c9d2edcf (a plain send) and on TASK-33620.10's branch. So the trace never holds what was actually sent to such an endpoint.

It also blocks any second traced call that has to compare against that surface. TASK-37421's one-time non-streaming retry did: live on the branch, a custom-ep endpoint whose server refused `"stream": true` ended "Run: Blocked — capture failed", because the retry's trace reservation raised `ValueError("surface_replacement_checkpoint_unavailable")` in `prepare_current_surface_delta` (the incoming saved revision cannot replace the omission without a checkpoint for the `llama_fallback` route). TASK-37421 therefore leaves the `custom-hosted` key out of the retry (`refused_stream.retries_refused_stream`); such an endpoint's refused stream fails once with the Streaming advice, and an empty or unreadable stream fails as "unexpected provider error".

Evidence: the live captures, front logs and DB reads were kept with that session's working files, not in the repo.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A custom-ep (OpenAI-compatible family) call traced with capture on records the saved user revision in its surface, as a Custom slot call does, or the reason it cannot is documented and the retry's reservation tolerates it
- [x] #2 With that in place, the custom-hosted key is retried once without streaming like the Custom slots, and a refused or empty custom-ep stream completes with capture on and its retry traced as its own call reserved on the llama_fallback route (amended 2026-11-06: since PR3304's integration that reservation is recorded as tool_loop in a Console agent run, as the Custom slots' retry is; recording custom-hosted's retry as llama_fallback instead blocks capture with trace_turn_unavailable / trace_tool_chain_unavailable, measured by mutation)
- [x] #3 A test with the real trace boundary factory drives a custom-ep call through the hosted engine and asserts the recorded surface; it fails before the change
- [x] #4 The User Guide's chat-basics Streaming section and the CHANGELOG no longer carve named custom endpoints out of the retry
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure on origin/dev 8ae979d81b with the real ConsoleTraceBoundaryFactory, controller, bridge, gateway, chat_api_call and the hosted engine against a loopback OpenAI-compatible server: a custom-ep send's traced surface, and its refused/empty/whole-JSON streams; compare with the Custom slot and with the 2026-10-29 commit 38c9d2edcf; bisect any difference.
2. RED tests: a new capture-on surface test file (custom-ep vs Custom: plain send records the saved user revision; refused and empty streams retried as their own traced call); add custom-ep to the bridge retry tests, the automatic-budget and capture-on accounting tests, the tool-chain retry test and the mounted ChatScreen send; drop it from the cloud test.
3. Fix: retries_refused_stream no longer excludes custom-hosted; stream_fallback_route_provenance projects its agent retry onto tool_loop like Custom; restore the hosted engine's unreadable-stream retry and the clean-empty-terminal-turn retry dropped by 5be5b655b3, with gateway unit tests.
4. Mutation-check each change; docs (chat-basics Streaming, CHANGELOG); ratchet row by exact growth.
5. Live: real llama-server behind a front that refuses stream=true (and one that streams nothing), custom-ep, capture on, branch vs origin/dev; read console_trace_calls / surface nodes / messages.usage_json.
6. Rebase onto newest origin/dev, re-run gates and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**Measured first (origin/dev 8ae979d81b, 2026-11-06).** A probe drove a plain send and refused/empty/whole-JSON streams through the real `ConsoleChatController`, `ConsoleAgentBridge`, gateway, `chat_api_call` and adapters, with the real `ConsoleTraceBoundaryFactory` and capture on, against a loopback OpenAI-compatible server, for Custom and `custom-ep:box`. On dev the custom-ep call's traced surface already records the saved user revision (`message`/`revision`, integrity complete, no omission), like Custom. The same probe on 38c9d2edcf (the commit the finding was measured on) reproduces the `provider_overlay` `alignment_mismatch` omission. `git bisect` names bf1886c95c (multi-model review delivery, 2026-10-28): it moved the custom-hosted base-URL projection (strip `/chat/completions`, the engine adds its route) into the shared `adapter_api_base_url` and used it in `reconstruct_provider_gateway_kwargs` too; before, the gateway sent the stripped base and the independent reconstruction expected the unstripped one, so `api_base_url` never aligned. So AC#1 was met on dev by that commit; this task pins it (AC#3) and restores the retry. On dev the remaining behaviour was: refused stream -> one request + Streaming advice; empty stream or whole JSON -> three streamed requests (the agent runtime retries the hosted engine's protocol error as "unexpected provider error"), each traced as an error.

**Fix, reusing TASK-37421's path.** `retries_refused_stream` no longer excludes `custom-hosted`. `stream_fallback_route_provenance` no longer keeps `custom-hosted` on `llama_fallback`: the retry reserves its call on the fallback route, which a Console agent run projects onto `tool_loop` (keeping the run's actor/chain), as the Custom slots' retry has since PR3304's integration; kept on `llama_fallback`, capture blocks (`trace_turn_unavailable` / `trace_tool_chain_unavailable`), so AC#2's route wording is amended. The two paths 5be5b655b3 dropped with the exclusion are back: `refused_stream.is_unreadable_stream` (the hosted engine's `HostedChatProtocolError` before anything arrived: no finish record, or a whole JSON answer to a streamed request) and a clean empty terminal turn (finish reason, no text, no tool call) in the tools path, both retried once without streaming. Failure evidence goes through the existing `_note_provider_failure` (an unreadable stream has no response-owned status, so it stays unknown, never a refusal) and `_note_provider_outcome`.

**Tests, red for the right reason.** New `Tests/Chat/test_custom_endpoint_trace_surface.py` (real factory, capture on, Custom vs custom-ep): a plain send's surface holds the saved user revision (custom-ep red on the pre-fix commit d999e7f4a9 = bf1886c95c^, where the send fails with `surface_provenance_mismatch`, and red on dev with the reconstruction's base URL mutated back); refused and empty streams are retried, one user row and one reply, calls `[error agent_first, complete tool_loop]`, the retry's usage once, both calls on the saved revision (custom-ep cases red on dev). Added custom-ep to the loopback bridge tests (refused, empty; plus finish-record-only and whole-JSON shapes), to the automatic-budget, capture-on usage and tool-chain retry tests (`Tests/Chat/test_self_hosted_retry_accounting.py`) and to the mounted ChatScreen send (`Tests/UI/test_console_refused_stream_retry_screen.py`: one user row, one reply, usage once in the store and `messages.usage_json`); removed it from the cloud not-retried test. Gateway unit tests restored for the two hosted-engine paths. The loopback server gained `finish_records` (a finish record before the usage record, what a real server streams) for the hosted-engine cases; its short form is byte-identical for the legacy cases. Mutations, each reverted by edit: re-excluding custom-hosted (6 bridge/surface reds, the screen send red, the terminal-turn unit test red), keeping its route on llama_fallback (5 capture-on reds), dropping the unreadable retry (3), dev's terminal-turn branch order (4 + the unit test), dropping the `not delivered` guard (the after-text unit test), and the pre-bf1886c95c base URL in the reconstruction (3 surface reds).

**Live (2026-11-06, capture on, isolated profile, null keyring).** No server on this host refuses streaming, so the stand-in is the one TASK-37421 used: a real llama-server (qwen2.5-0.5b-instruct q4_k_m, port 9762) behind a local front (port 9763) that answers `"stream": true` with HTTP 400 (mode refuse) or with an empty stream (mode empty) and forwards everything else. Console used a named endpoint `[custom_endpoints.box]` (`openai_compatible`) as `custom-ep:box`. Branch: refuse -> front `stream=true -> 400`, `stream=false -> 200`; reply shown, no System row; `messages.usage_json` 96 uncached / 10 output = llama-server's 96 prompt / 10 eval; `console_trace_calls` agent_first error (no usage) + tool_loop complete (96/10), both surfaces the saved user revision, integrity complete. Empty -> `stream=true -> 200 empty`, `stream=false -> 200`; reply "Red."; usage 105 cache read + 14 uncached / 3 output = llama-server's 14 prompt (105 cached) / 3 eval; same trace shape. origin/dev 8ae979d81b, same front and prompt: one refused request, "Response failed" and the Streaming advice. Evidence (captures, front log, llama-server log, DB reads) kept with the session's working files, not in the repo.

**Gates (branch vs a detached origin/dev 8ae979d81b, by test id, plain command).** New/changed files on the branch: test_console_refused_stream_retry 17 passed (dev's copy 14), test_self_hosted_retry_accounting 53 (dev 47), test_custom_endpoint_trace_surface 6 (new), test_console_refused_stream_retry_screen 2 (dev 1), test_console_provider_gateway 488 passed + the same 2 base-URL-pin reds as dev (qwencloud, deepseek), module-size ratchet 201, CI queue-pressure contract 118. Neighbours with the same reds as dev: trace provenance (1 red on both: test_explicit_prepared_request_constructor_rejects_missing_on_provenance), provider failure copy (1 on both: test_stream_chat_provider_400_reports_one_consistent_status), agent bridge 339 passed + the same 4 usage-observation reds; all green on both: trace runtime 86, retry recovery proof 29, failed-call retry ownership 40, model retries 4, call lifecycle 15, settlement 36, surface identity 9, surface recovery 8, refusal diagnostics 3, custom endpoint engine swap 49 and registry 20, automatic provider budget 51, first-reply failure copy 35, typed failure copy 48, retry JSON wire 6 and identity 32. test_console_project_instruction_traces dies at setup on both trees with RecoveryRequired (raw_source_selection_changed); under a throwaway bootstrap_profile plugin (deleted) both trees pass 61 and fail the same 6 (test_transformed_project_sends_preserve_source_and_context, all six parametrizations). ruff: the changed files report nothing new (the gateway pair's 7 findings are dev's). The new surface file joins pr-fast-lane-admission-2 with the TASK-37421 files.

**Files.** tldw_chatbook/Chat/refused_stream.py, tldw_chatbook/Chat/console_provider_gateway.py, tldw_chatbook/Chat/console_trace_provenance.py; Tests/Chat/test_custom_endpoint_trace_surface.py (new), Tests/Chat/test_console_refused_stream_retry.py, Tests/Chat/test_self_hosted_retry_accounting.py, Tests/Chat/test_console_provider_gateway.py, Tests/UI/test_console_refused_stream_retry_screen.py, Tests/Architecture/test_module_size_ratchet.py (gateway row 8487 -> 8495, +8, dated owner-decision comment), Tests/CI/test_ci_queue_pressure_contract.py, .github/workflows/derived-artifacts.yml; Docs/User_Guide/console/chat-basics.md, CHANGELOG.md; backlog/docs/lessons-testing-evidence.md.

**Not changed.** A whole JSON answer to a streamed request is normalized in one request for Custom and vLLM (PR3304) but retried once for a named endpoint, because the hosted engine cannot read it as a stream; normalizing it in the engine would be a separate change. TASK-37700 (OpenAI's status loss), TASK-37760 (shared first-token window) and TASK-37181 (llama.cpp direct usage) stay open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->
