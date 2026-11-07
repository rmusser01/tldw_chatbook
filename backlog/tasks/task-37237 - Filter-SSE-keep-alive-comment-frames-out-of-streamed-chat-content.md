---
id: TASK-37237
title: Filter SSE keep-alive comment frames out of streamed chat content
status: Done
assignee:
  - '@dsh'
created_date: '2026-10-30 08:30'
updated_date: '2026-10-30 08:30'
labels:
  - console
  - streaming
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Proxies and gateways in front of OpenAI-compatible endpoints legally inject
Server-Sent Events comment frames (lines whose first character is `:`, e.g.
`: keep-alive`) between records as connection heartbeats. The Console streamed
that transport noise into the transcript as assistant content, spamming
"keep-alive" repeatedly during responses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The `chat_with_openai` streaming relay drops non-`data:` lines instead of passing every non-empty raw line through.
- [x] #2 The shared local OpenAI-compatible relay (llama.cpp/vllm/ollama/mlx/ooba/tabby/aphrodite/custom-openai) drops non-`data:` lines.
- [x] #3 The Console gateway content sink drops SSE transport frames (comments and `event:`/`id:`/`retry:` fields) defensively, even if a relay regresses; a comment-only stream degrades to the no-content fallback copy, never to spam.
- [x] #4 Regression tests reproduce the spam on unpatched HEAD (red) and pass with the fix (green).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: routine bug fix in stream-frame filtering; no storage, permission, provider-contract or UX-structure decision.
1. Trace the spam end to end: provider stream generator → `chat_api_call` relay → `ConsoleProviderGateway.normalize_provider_response` → transcript.
2. Audit every provider streaming loop for non-`data:` line handling (openai, anthropic, cohere, google, huggingface, hosted engine, qwencloud, local relay).
3. Fix the two leaking relays and add a defensive sink filter.
4. Prove red/green against a clean HEAD worktree; confirm the touched suites' other failures are pre-existing machine state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Cause** (three cooperating leaks):
  1. `LLM_Calls/LLM_API_Calls.py` `chat_with_openai` stream generator relayed *every* non-empty raw line, so `: keep-alive` comment frames from any OpenAI-compatible gateway/proxy flowed downstream.
  2. `LLM_Calls/LLM_API_Calls_Local.py` `_chat_with_openai_compatible_local_server` — the one generator every local provider family shares — did the same raw pass-through. **Independently hardened on dev** (PR #3288 lineage) to `yield line + "\n\n" if line.startswith("data:") else ""`; this task's relay change is therefore dropped from the PR and the regression test pins dev's behavior instead.
  3. `Chat/console_provider_gateway.py` `_content_from_provider_item` returned any non-`data:` string/bytes item *verbatim as assistant content*, which rendered the comment text into the Console transcript.
- **Fix**: the `chat_with_openai` relay now forwards only `data:` lines, and the gateway sink gained `_is_sse_transport_frame` (SSE `:` comments plus `event:`/`id:`/`retry:` control fields → `_EMPTY_RESPONSE`) as defence in depth — covering every relay, including any future one that regresses. All hosted providers were already safe via the strict `SSERecordDecoder`, which ignores comment frames by spec.
- **Also caught while testing**: `event: ping`-style control lines leaked through the sink exactly like comments; both are covered.
- **Evidence**: `Tests/Chat/test_console_keepalive_comment_filtering.py` (4 tests) fails 4/4 on unpatched HEAD in a clean worktree and passes 4/4 with the fix. The 28 failures in the touched suites (`test_local_reasoning_gateway.py`, `test_streaming_stop_closes_transport.py`, 2 gateway tests) reproduce identically on clean HEAD — machine backup-recovery state (`RecoveryRequired`), pre-existing and unrelated.
- **Notable**: a pre-existing stash (`stash@{0}`, branch `fix/library-stt-url-scheme-collapse`) contains an earlier partial fix for this same bug (local module only, `:`-comments only). It was never applied and is insufficient (misses the `chat_with_openai` relay, `event:` lines, and the sink); left untouched for its owner.
- **Manual UAT helper**: `scripts/mock_keepalive_sse_server.py` (stdlib-only, default port 8799) serves an OpenAI-compatible `/v1/chat/completions` that injects `: keep-alive` comments and `event: ping` frames between every token chunk (50 + 25 frames vs 28 data records per reply). Verified live end-to-end: the real `chat_api_call("openai", streaming=True)` chain against this server relays 29 items with 0 transport frames leaked, and `normalize_provider_response` renders only the clean reply.
- **Modified files**: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`, `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py`, `tldw_chatbook/Chat/console_provider_gateway.py`, `Tests/Chat/test_console_keepalive_comment_filtering.py` (new), `scripts/mock_keepalive_sse_server.py` (new).
<!-- SECTION:NOTES:END -->
