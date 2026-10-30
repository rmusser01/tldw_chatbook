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
updated_date: 2026-10-31 01:59
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
Latest-dev integration onto 134108dd84: retain both the UTF-8 transport explanation and upstream Strata documentation, reproduce regression-fixture compatibility with upstream empty keep-alive checkpoints, and adjust only fixture consumption if needed. Run targeted encoding/provider/Strata and affected Console checks, lint and independent review, then force-with-lease and re-arm PR3308 under its existing Qodo-only waiver. ADR required: no; no protocol or provider ownership changes.
Subsequent latest-dev integration onto7536ac3b87: retain upstream TASK33621.51 lessons and all three PR lessons, keep the four reviewed code/test files byte-identical, verify with range-diff and targeted provider/Console checks, then republish and re-arm with the existing gates. ADR required:no; documentation conflict resolution only.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The shared local OpenAI-compatible streaming sender selects UTF-8 before Requests decodes SSE lines. This prevents charset-less text/event-stream responses from taking Requests' Latin-1 default and also ignores misleading charset declarations, as required by SSE. The source change is one setting plus its explanatory comment; payloads, routing, non-streaming replies and stream cleanup retain their existing owners. Existing malformed saved replies are not reconstructed.

Tests/LLM_Calls/test_local_sse_encoding.py uses a real Requests Session and HTTPAdapter.build_response with an in-memory byte transport. Its 36 cases cover literal/escaped Unicode and ASCII, missing/UTF-8/Latin-1 charset headers, one-byte/whole-body reads, normal exhaustion and consumer Stop. Before the fix, eight literal-Unicode cases failed and 28 controls passed. After the fix, the targeted run passed all 104 cases, including local argument routing, thinking dispatch, first-token timeouts, native-tool content/fragments and iterator cleanup. Ruff lint, Ruff format check and git diff --check passed.

Validation command: .venv/Scripts/python.exe -m pytest Tests/LLM_Calls/test_local_sse_encoding.py Tests/LLM_Calls/test_dispatch_local_argument_routing.py Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/test_first_token_window.py Tests/Chat/test_console_provider_gateway.py::test_stream_accumulates_sse_tool_call_fragments Tests/Chat/test_console_provider_gateway.py::test_non_streaming_message_tool_calls_surface Tests/Chat/test_console_provider_gateway.py::test_tools_run_with_real_content_and_no_calls_stays_a_normal_answer Tests/Chat/test_console_provider_gateway.py::test_tee_tool_calls_closes_underlying_iterator_once Tests/Chat/test_streaming_stop_closes_transport.py::test_no_streaming_handler_yields_inside_a_finally -q

Native verification used a fresh TEMP/TMP under the user's private .codex directory, avoiding the earlier shared-writable Temp/checkout ancestor refusals without changing existing permissions. The run reports the existing Pydantic exclude_if deprecation and pytest-asyncio loop-scope warning. These are offline transport checks, not a live llama.cpp or full-suite qualification.

Docs/Architecture/llm-providers.md records the decoding boundary and backlog/docs/lessons-testing-evidence.md records why pre-decoded fixtures miss this defect. ADR required: no; this is a routine protocol fix within existing ADR-063/179 boundaries.
Independent review found no actionable issues. It verified that the UTF-8 override precedes incremental decoding, that regression cases exercise real fragmented bytes and response closure, and that provider routing/lifetime logic is preserved. All acceptance criteria are complete under the targeted verification scope.
Latest-dev integration (2026-10-30): GitHub reported a new merge conflict after dev advanced. Disabled auto-merge before pulling/rebasing; rebased onto 134108dd846f4af85f5efb8252e12cb7f61ee750. The only conflict was Docs/Architecture/llm-providers.md; retained both this task's UTF-8 explanation and upstream Strata/PolyStrata documentation. Range-diff preserves the two reviewed UI harness commits unchanged.
Upstream changed the shared sender to yield empty SSE checkpoints for Stop handling. The unchanged encoding fixture failed 18 exhaustion cases with JSONDecodeError on empty chunks while 18 Stop cases passed. The fixture now excludes only falsy chunks while preserving exact content, DONE and response closure assertions. Production still differs from dev by only the UTF-8 setting and its comment.
Fresh native GREEN: 152 targeted provider/transport/Strata cases passed (29.30s), and all five Console restyle tests passed (64.09s). An isolated temporary decoder-negative probe reinstated header-based encoding immediately before real Requests iter_lines; the original eight Unicode cases failed and 28 controls passed. This confirms the checkpoint filter does not hide charset corruption. Scoped Ruff lint/format and git diff --check pass. Existing dependency warnings only; no full suite or live-server qualification.
Focused independent latest-dev integration review found no actionable issues. It confirmed exact data-content comparisons, preservation of upstream Stop/control handling, cleanup and both documentation sections. The pre-existing Unicode-separator splitlines/SSE framing edge (U+0085/U+2028/U+2029, also present with correctly labeled UTF-8 on dev) remains outside this charset repair, as already ruled in the original review. No parser migration is claimed. ADR required: no; existing provider boundaries remain unchanged. Hosted checks were all green on the prior head and must pass again after publication.
Subsequent dev rebase (2026-10-30): all 38 hosted jobs were green on b5227809a9, then dev 7536ac3b87 introduced a conflict only in lessons-testing-evidence.md. Disabled auto-merge before pulling/rebasing; retained both upstream TASK-33621.51 lessons and all three PR lessons. Range-diff changes only first-commit lesson context; the other three patches and all four reviewed code/test files are unchanged. Fresh targeted qualification on the new base: 152 provider/transport/Strata cases passed in 53.74s; five Console restyle tests passed in 82.18s. Scoped Ruff lint/format and diff checks pass. Read-only independent conflict review found no issues and verified no upstream lesson loss. ADR required: no; documentation conflict resolution only. Required hosted CI must pass on the republished head.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
