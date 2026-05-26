---
id: TASK-71
title: Console native chat core
status: Done
assignee: []
created_date: '2026-05-22 08:12'
updated_date: '2026-05-24 06:45'
labels:
  - console
  - chat
  - ux
dependencies: []
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console uses native chat core for send and streaming
- [x] #2 Local llama.cpp streaming path is verified
- [x] #3 Transcript message selection and actions work
- [x] #4 Unavailable action paths show WIP reasons
- [x] #5 Actual CDP screenshots are captured and approved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add native chat contracts
2. Add provider gateway
3. Add stores and controller
4. Wire Console send/streaming
5. Replace transcript rendering
6. Add selected-message actions
7. Capture CDP QA evidence
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Console native chat core with native session/store/controller/gateway plumbing, local llama.cpp streaming, transcript selection/actions, WIP action reasons, and repo-tracked CDP screenshot evidence.

PR review follow-up addressed all Qodo findings: draft text is validated and sanitized at the controller boundary; llama.cpp base URLs are normalized and validated before HTTP requests; streaming chat now falls back to non-streaming completions when SSE is unavailable or yields no chunks; provider blocked copy is normalized in the controller so the UI no longer duplicates system messages or mutates run state directly; transcript polling is throttled and coalesced; streamed assistant content is buffered and materialized on UI/terminal boundaries to avoid per-chunk string concat.

Verification:
- Red regression run with the newly added review tests failed before implementation.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_chat_controller.py::test_submit_draft_sanitizes_user_text_before_storage_and_provider_send Tests/Chat/test_console_chat_controller.py::test_submit_draft_blocks_unsafe_markup_before_storage_or_provider_send Tests/Chat/test_console_chat_controller.py::test_blocked_provider_wip_copy_is_normalized_once_in_controller Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_normalizes_scheme_less_llamacpp_base_url_before_http Tests/Chat/test_console_provider_gateway.py::test_resolve_for_send_blocks_invalid_llamacpp_base_url_before_http Tests/Chat/test_console_provider_gateway.py::test_llamacpp_stream_chat_falls_back_to_non_streaming_when_stream_rejected Tests/Chat/test_console_provider_gateway.py::test_llamacpp_stream_chat_falls_back_when_sse_has_no_content_chunks Tests/Chat/test_console_chat_store.py::test_store_buffers_stream_chunks_until_messages_are_materialized Tests/UI/test_console_native_chat_flow.py::test_console_llamacpp_base_url_normalizes_openai_compatible_endpoints Tests/UI/test_console_native_chat_flow.py::test_console_transcript_sync_timer_polls_at_coarse_interval Tests/UI/test_console_native_chat_flow.py::test_console_unsupported_provider_block_renders_one_normalized_system_message --tb=short` passed with 18 tests.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_internals_decomposition.py --tb=short` passed with 175 tests.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py` passed; generated output only changed the timestamp, so that churn was removed.
- `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console native chat is first-class and the PR review follow-up is complete. The branch now validates/sanitizes draft input, validates and normalizes llama.cpp base URLs, supports non-streaming fallback completions, avoids duplicate blocked transcript messages, reduces transcript sync churn, and buffers stream chunks without losing visible streaming updates.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests and verification recorded
- [x] #3 Documentation updated in task record
- [x] #4 Self-review complete
- [x] #5 No known blockers
<!-- DOD:END -->
