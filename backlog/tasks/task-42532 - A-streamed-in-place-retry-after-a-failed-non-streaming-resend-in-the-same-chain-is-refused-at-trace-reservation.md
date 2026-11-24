---
id: TASK-42532
title: "A streamed in-place retry after a failed non-streaming resend in the same chain is refused at trace reservation"
status: To Do
assignee: []
created_date: 2026-11-24 09:14
updated_date: 2026-11-24 09:14
labels:
- follow-up
- console
- agents
- trace
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_provenance.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Chat/refused_stream.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by code reading during TASK-42472's round-24 work (2026-11-24); **not probed**. Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** Between a failed call and its retry, `Chat/console_trace_service.py::_retry_generation_matches` allows only one change in generation settings: streaming True -> False, proven by the stream-fallback overlay. Two paths make a failed call a non-streaming resend *inside the run's own chain* (a TOOL_LOOP call with `streaming=False`). After either one, the agent's next in-place retry streams again, is compared with `streaming=False`, and would be refused with `trace_tool_chain_unavailable`, ending on the trace card instead of the provider's error:
- TASK-37421: a self-hosted or Custom OpenAI-compatible endpoint that refuses a stream (400/406/415/422/501, or an empty stream; `Chat/refused_stream.py`) is sent once more without streaming. If that resend then fails transiently (a 429, a 5xx, a timeout), the agent retries.
- llama.cpp with tools in the request (`has_tools`): `Chat/console_trace_provenance.py::stream_fallback_route_provenance` projects the resend onto TOOL_LOOP in the chain instead of its own `llama_fallback` run.

TASK-42472 AC#8 covered only the no-tools llama.cpp resend, which has its own run id.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A probe or test establishes, for each of the two paths, whether a streamed in-place retry after a failed non-streaming resend is refused today
- [ ] #2 Where it is refused, a captured run retries the streamed call in place and ends on the provider's own error when retries run out; tests are red before the fix
- [ ] #3 A retry that changes target, settings other than this streaming flip, or surface is still refused
<!-- AC:END -->
