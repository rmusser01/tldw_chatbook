---
id: TASK-42534
title: "After a captured agent run fails before any response, the user's next message is blocked at trace reservation"
status: To Do
assignee: []
created_date: 2026-11-24 09:14
updated_date: 2026-11-24 10:23
labels:
- follow-up
- console
- agents
- trace
priority: high
references:
- tldw_chatbook/Chat/console_trace_continuation.py
- tldw_chatbook/Chat/console_trace_runtime.py
- Tests/Chat/test_console_project_instruction_traces.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-42472 (2026-11-24) and reproduced on dev af4b13ea75 and on fu/r24-diag. Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** A Capture-On agent run whose model calls all fail before any response ends `failed` on the provider's error. Examples: a 429 or 5xx that outlasts the in-place retries, or a llama.cpp stream whose non-streaming resend fails too. The user's next message in the same chat is then refused at trace reservation with `completed_tool_turn_unavailable`. The run ends `blocked` on 'Trace capture could not start', and the provider is never contacted. A run that recovers on a retry does not do this.

**Cause.** The new turn's reservation builds a completed-turn witness over the failed turn, whose assistant reply failed (`closed_assistant_message_id` is set). `Chat/console_trace_continuation.py::validate_completed_tool_turn` refuses a closed turn whose terminal call has no `response_started_at` (`closed and not explicit_discard and terminal.response_started_at is None`). A run that failed before any response never has one. This is not TASK-42472 AC#8's cause (a llama.cpp resend's own run id): a hosted run with no resend blocks the same way.

**Evidence.** Probe `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/3f117384-6f37-40d1-bdd4-c465d56c3f56/scratchpad/fu/r24fx-nextsend-probe.py`. Copy it into `Tests/Chat/` and run it through `$FU/run.sh`; the scratch path may have been cleaned. It uses the real controller, bridge, gateway and trace factory in `Tests/Chat/test_console_project_instruction_traces.py::_project_console`, with Capture On and project instructions present.
- dev af4b13ea75, deepseek through the gateway's handler seam (`functools.wraps(chat_api_call)`), failing with 503:
  - failing 3 times (retries spent): the run fails; the next send is `blocked`, `['completed_tool_turn_unavailable']`;
  - failing once: the retry recovers, and the next send completes.
- dev af4b13ea75, llama.cpp streaming, stream and resend both 503 (dev does not retry llama.cpp): lineage `agent_first=error`, `llama_fallback=error`; the next send is `blocked`, `completed_tool_turn_unavailable`.
- fu/r24-diag: the same for deepseek, and for OpenAI's real handler (`requests.Session.post` returning 503 three times).

Whether the block needs project context (the harness always has an `AGENTS.md`) was not checked.

**The cause is confirmed.** In the second review of TASK-42472, disabling `and terminal.response_started_at is None` (dev `console_trace_continuation.py:637`) unblocked the deepseek, llama.cpp and OpenAI next sends on af4b13ea75.

**The probe, inlined** (the scratch copy may be cleaned). Copy it into `Tests/Chat/` of a worktree and run it through the scratch-profile wrapper with `-s`. Each test prints `NEXT ...` with the first run's status, the second send's status and the reservation errors.
- `test_hosted_failed_then_next` replaces the gateway's handler seam with a `functools.wraps(chat_api_call)` stand-in that raises `ChatProviderError(503)` for the first `fail` calls.
- `test_llama_stream_failed_then_next` patches `httpx.MockTransport.handle_async_request` to return 503 for the first two requests (the stream and its non-streaming resend); later requests reach the harness's own transport.

```python
"""TEMPORARY probe (deleted after use): the send after a captured run that failed."""
import functools

import httpx
import pytest

from Tests.Chat.test_console_project_instruction_traces import _project_console
from tldw_chatbook.Chat import Chat_Functions
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("fail", [1, 3])
async def test_hosted_failed_then_next(tmp_path, monkeypatch, fail):
    """deepseek through the gateway's handler seam; `fail` calls fail with 503."""
    async with _project_console(tmp_path, monkeypatch) as app:
        calls = []

        @functools.wraps(Chat_Functions.chat_api_call)
        def handler(*_a, **_k):
            calls.append(1)
            if len(calls) <= fail:
                raise ChatProviderError("busy", provider="deepseek", status_code=503)
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        async def resolve(_selection):
            return ConsoleProviderResolution(
                ready=True, provider="deepseek", execution_key="deepseek",
                model="deepseek-chat", base_url="https://api.deepseek.com",
                api_key="fixture", streaming=False,
                resolved_destination=ConsoleResolvedDestination(
                    provider="deepseek", model="deepseek-chat",
                    endpoint_identity="https://api.deepseek.com",
                    egress_class=ConsoleEgressClass.PUBLIC_NETWORK),
            )

        monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
        monkeypatch.setattr(app.gateway, "_chat_api_call_fn", handler)
        await app.controller.submit_draft("Hello", session_id=app.session.id)
        first = app.controller.run_state.status.value
        n = len(calls)
        await app.controller.submit_draft("And now?", session_id=app.session.id)
        print("NEXT hosted fail", fail, first, n, "->", app.controller.run_state.status.value, app.reservation_errors, len(calls))


async def test_llama_stream_failed_then_next(tmp_path, monkeypatch):
    """llama.cpp streaming: the stream and its non-streaming resend both get 503."""
    real = httpx.MockTransport.handle_async_request
    seen = []

    async def failing(self, request):
        seen.append(1)
        if len(seen) <= 2:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return await real(self, request)

    monkeypatch.setattr(httpx.MockTransport, "handle_async_request", failing)
    async with _project_console(tmp_path, monkeypatch, response_mode="stream") as app:
        await app.controller.submit_draft("Hello", session_id=app.session.id)
        first = app.controller.run_state.status.value
        with app.db.transaction() as cursor:
            calls = app.factory.repository.read_conversation_call_lineage(cursor, app.session.persisted_conversation_id)
        lineage = [(c.route_identity, c.state.value) for c in calls]
        await app.controller.submit_draft("And now?", session_id=app.session.id)
        print("NEXT llama stream", first, lineage, "->", app.controller.run_state.status.value, app.reservation_errors, len(seen))
```
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a Capture-On agent run that ended `failed` before any response, the user's next message is sent and answered; tests cover a hosted run whose retries ran out and a llama.cpp stream whose non-streaming resend also failed, and are red before the fix
- [ ] #2 The next turn's trace never records the failed turn's calls as having borne a response
- [ ] #3 The notes record whether the block depends on project context
<!-- AC:END -->
