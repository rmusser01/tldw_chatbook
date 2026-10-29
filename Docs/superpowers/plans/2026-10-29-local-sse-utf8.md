# Local SSE UTF-8 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this single task in the current session.

**Goal:** Preserve Unicode in local SSE replies when the response omits or misstates its charset.

**Architecture:** Set UTF-8 on the shared local Requests response before its SSE iterator reads bytes. Keep provider routing, payloads, retries and stream lifetime under their existing owners.

**Tech Stack:** Python, Requests, pytest.

**Spec:** TASK-37600 and the user's approved fix following the screenshot investigation.

**ADR required:** no
**ADR path:** N/A; existing ADR-063/179 provider boundaries apply.
**Reason:** Routine protocol-decoding bug fix within the existing local adapter; no new storage, dependencies, service contract or runtime boundary.

## Global Constraints

- Change only the shared local SSE decoding boundary; do not migrate providers or alter saved messages.
- Use real Requests response construction and byte iteration in regression coverage; replace only the network transport.
- Run targeted tests only, using fresh private profiles. Do not change existing directory permissions.

## Task 1: Preserve local streamed Unicode

**Files:** Modify `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py` and `Docs/Architecture/llm-providers.md`; create `Tests/LLM_Calls/test_local_sse_encoding.py`.

- [x] Add `test_local_sse_preserves_text_and_closes_response` with literal Unicode/ASCII expectations, raw and escaped JSON, missing/UTF-8/Latin-1 charset headers, one-byte/full-buffer transport reads, and exhaustion/consumer Stop. Exercise the actual adapter through a Requests Session with a local fixture HTTPAdapter.
- [x] Run `pytest Tests/LLM_Calls/test_local_sse_encoding.py -q`; raw Unicode with missing or misleading charset must fail on corrupted content, while UTF-8-header/escaped-JSON/ASCII controls pass.
- [x] Select UTF-8 before local SSE consumption and document why the response header is insufficient.
- [x] Run the regression file and relevant local-dispatch, native-tool, stream-lifetime and provider tests; run scoped Ruff lint/format and diff checks. Document evidence limits and the existing private-profile test startup issue.
- [x] Self-review, request focused independent review, and record results in TASK-37600 before closeout.

## Review Focus

Missing and misleading charsets must both preserve Unicode. One-byte transport reads must preserve multibyte characters. Escaped JSON and ASCII must remain compatible. Normal stream exhaustion must release the real response body. Existing Stop/cleanup behavior must retain its targeted lifecycle checks.
