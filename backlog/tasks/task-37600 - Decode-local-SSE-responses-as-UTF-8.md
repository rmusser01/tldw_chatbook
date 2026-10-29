---
id: TASK-37600
title: Decode local SSE responses as UTF-8
status: Done
created_date: 2026-10-29 20:22
assignee:
- '@codex'
documentation:
- Docs/Architecture/llm-providers.md
- Docs/superpowers/plans/2026-10-29-local-sse-utf8.md
updated_date: 2026-10-29 20:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent mojibake in local-provider streamed replies, including llama.cpp agent requests with native tools, when event-stream responses omit or misstate their charset.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local SSE replies preserve Unicode punctuation, accented text, non-Latin scripts and emoji with missing, UTF-8 or misleading response charset headers, including multibyte characters split across transport chunks.
- [x] #2 Regression coverage exercises raw response bytes through the actual shared local adapter, with ASCII/escaped-JSON compatibility and response cleanup retained.
- [x] #3 Relevant targeted tests, lint and formatting pass; provider transport documentation and verification evidence describe the scope and limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-10-29-local-sse-utf8.md: reproduce with raw Requests response bytes before changing the shared SSE decoder, verify Unicode/ASCII and stream cleanup, then run targeted provider checks and update the transport reference.
ADR required: no
ADR path: N/A (existing ADR-063/179 boundaries apply)
Reason: routine decoding bug fix within the current local-provider contract; no architectural boundary or dependency changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The shared local OpenAI-compatible streaming sender selects UTF-8 before Requests decodes SSE lines. This prevents charset-less text/event-stream responses from taking Requests' Latin-1 default and also ignores misleading charset declarations, as required by SSE. The source change is one setting plus its explanatory comment; payloads, routing, non-streaming replies and stream cleanup retain their existing owners. Existing malformed saved replies are not reconstructed.

Tests/LLM_Calls/test_local_sse_encoding.py uses a real Requests Session and HTTPAdapter.build_response with an in-memory byte transport. Its 36 cases cover literal/escaped Unicode and ASCII, missing/UTF-8/Latin-1 charset headers, one-byte/whole-body reads, normal exhaustion and consumer Stop. Before the fix, eight literal-Unicode cases failed and 28 controls passed. After the fix, the targeted run passed all 104 cases, including local argument routing, thinking dispatch, first-token timeouts, native-tool content/fragments and iterator cleanup. Ruff lint, Ruff format check and git diff --check passed.

Validation command: .venv/Scripts/python.exe -m pytest Tests/LLM_Calls/test_local_sse_encoding.py Tests/LLM_Calls/test_dispatch_local_argument_routing.py Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/test_first_token_window.py Tests/Chat/test_console_provider_gateway.py::test_stream_accumulates_sse_tool_call_fragments Tests/Chat/test_console_provider_gateway.py::test_non_streaming_message_tool_calls_surface Tests/Chat/test_console_provider_gateway.py::test_tools_run_with_real_content_and_no_calls_stays_a_normal_answer Tests/Chat/test_console_provider_gateway.py::test_tee_tool_calls_closes_underlying_iterator_once Tests/Chat/test_streaming_stop_closes_transport.py::test_no_streaming_handler_yields_inside_a_finally -q

Native verification used a fresh TEMP/TMP under the user's private .codex directory, avoiding the earlier shared-writable Temp/checkout ancestor refusals without changing existing permissions. The run reports the existing Pydantic exclude_if deprecation and pytest-asyncio loop-scope warning. These are offline transport checks, not a live llama.cpp or full-suite qualification.

Docs/Architecture/llm-providers.md records the decoding boundary and backlog/docs/lessons-testing-evidence.md records why pre-decoded fixtures miss this defect. ADR required: no; this is a routine protocol fix within existing ADR-063/179 boundaries.
Independent review found no actionable issues. It verified that the UTF-8 override precedes incremental decoding, that regression cases exercise real fragmented bytes and response closure, and that provider routing/lifetime logic is preserved. All acceptance criteria are complete under the targeted verification scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
