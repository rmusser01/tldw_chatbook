---
id: TASK-42534
title: "After a captured agent run fails before any response, the user's next message is blocked at trace reservation"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 09:14
updated_date: 2026-11-25 01:22
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

**The cause is confirmed.** In the second review of TASK-42472, disabling `and terminal.response_started_at is None` (dev `console_trace_continuation.py:637`) unblocked the probe's next sends on af4b13ea75: both deepseek cases (fail 1 and fail 3) and the llama.cpp case. The probe has no OpenAI case. The OpenAI block (OpenAI's real handler, `requests.Session.post` returning 503 three times) was seen on fu/r24-diag only, not on dev, and is not covered by that confirmation.

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
- [x] #1 After a Capture-On agent run that ended `failed` before any response, the user's next message is sent and answered; tests cover a hosted run whose retries ran out and a llama.cpp stream whose non-streaming resend also failed, and are red before the fix
- [x] #2 The next turn's trace never records the failed turn's calls as having borne a response
- [x] #3 The notes record whether the block depends on project context
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce at the round base with the inlined probe (hosted deepseek, retries spent; llama.cpp stream + resend failing, retries spent), with and without project context, plus a run whose first call answered with a tool call and whose later calls all failed (AC#3).
2. In `validate_completed_tool_turn`, replace the "a closed turn's terminal call bore a response" requirement with the contract: a closed turn's terminal may be a settled ERROR that never bore a response (it authorized at most an unchanged retry), and every removed tool node must have been appended by a call whose run predecessor bore a response. The second check is what keeps a failed call from ever authorizing tool history now that the terminal need not be response-bearing.
3. Tests through the real Console harness (`_project_console`): hosted run whose retries ran out, llama.cpp stream whose resend failed (retries spent), and the tool-then-failure run; each red before, green after, the next message answered, and the failed calls still carry no response start and no response link (AC#2). Negatives: a run that recovered on a retry and a completed tool run still validate; a tampered predecessor state (a failed call before appended tool history) and a closed terminal that completed without a response are still refused.
4. Record the project-context finding (AC#3), update the ADR-337 amendment text and the User Guide if it describes the block.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** As filed. `validate_completed_tool_turn` (`Chat/console_trace_continuation.py`) refused a closed turn (the run's assistant settled Failed and empty) whose terminal call never started a response. The 2026-09-09 closed-run witness (ADR-337) assumed the closed run's terminal call bore a response; the 2026-10-04 integration then let a run retry calls that fail with no response, so a run whose retries all fail ends on exactly such a terminal.

**Fix (contract level, every route).** A failed call authorizes an unchanged retry, never tool history.
- The witness now closes a turn on a settled ERROR terminal with no response. A closed terminal that completed or stopped without a response is still refused (`test_closed_history_is_revalidated_before_dispatch[no_response]` still pins the COMPLETE case).
- In exchange the witness checks the tool history directly: every removed tool-loop node must have been appended by a call whose predecessor in the run bore a response (RESPONSE_STARTED or COMPLETE). The factory already refused any append after a failed call at reservation; the witness no longer relies on a response-bearing terminal to stand for that.
- No call is relabeled and no response link is written (AC#2): the tests compare the failed turn's call rows (state, outcome, `response_started_at`, `settled_at`) and its response links before and after the next send.
- The fix is in the shared witness, so it covers every route that reaches it: hosted handlers, the llama.cpp stream and its resend, tool history, and a dictionary-transformed turn's restored source. `Tests/Chat/test_console_trace_current_turn_transforms.py::test_next_send_after_failed_transformed_call[error]` (TASK-32032 AC#5, item 1, red on dev with the same `completed_tool_turn_unavailable`) is red at this branch's base and green after; its `[stopped]` case stays red for its own reason (TASK-32032 item 2: the first call persists `error`, not `stopped`).
- ADR-337 gained "Amendment recorded 2026-11-24 (owner-approved): a closed run whose calls failed before any response" (the owner approved the relaxation on 2026-11-24), with an inline note on the 2026-09-09 sentence it relaxes; the User Guide's semantic-trace-capture page names the case.

**AC#3: does the block depend on project context?** Not on project context alone. It happens whenever the next request must replace the finished run's suffix, which is either project context (`AGENTS.md`) or tool history from earlier in the run. Probe at the round base 75bac413f5 (temporary file, deleted), real controller, bridge, gateway and trace factory, Capture On:

| first run | next send, `AGENTS.md` | next send, no project file |
|---|---|---|
| deepseek, 503 three times (retries spent) | blocked, `completed_tool_turn_unavailable` | completed |
| deepseek, 503 once (recovered on the retry) | completed | completed |
| llama.cpp stream + resend 503, three rounds | blocked | completed |
| llama.cpp (Streaming off), a tool call, then every later call 503 | blocked | blocked |

The test's `no_project` cases pin the same matrix.

**Tests** (`Tests/Chat/test_console_failed_call_retry_ownership.py`, the file that pins this contract; it runs in "Run the Dreams tracking tests and unlaned riders").
- `test_the_message_after_a_captured_run_that_failed_is_sent`, `_project_console` with Capture On, `project`/`no_project` for each run:
  - `hosted_exhausted`: deepseek through the gateway's handler seam (`functools.wraps(chat_api_call)`), three 503s;
  - `hosted_tool_then_exhausted`: a tool call, then three 503s;
  - `llama_stream_exhausted`: llama.cpp streaming, stream and resend 503 for three rounds;
  - controls that bore a response: `hosted_recovered` (one 503, then a reply) and `hosted_tool_completed`.
  Each asserts the next message is answered ("Fixture reply"), no reservation error, one more provider call, project context present or absent as set, and the failed turn's calls and links unchanged.
- `test_a_failed_call_never_authorizes_the_tool_history_a_turn_closes`: the call that answered with the tool call is made to read as failed (a `get_call_by_logical_identity` stand-in); the next message is refused with `completed_tool_turn_lineage` before the provider is asked, and the earlier trace is unchanged. Cases: `hosted_tool_completed`, `hosted_tool_then_exhausted`.
- Red before: at the base source, 4 of the 10 next-send cases fail (`hosted_exhausted-project`, `hosted_tool_then_exhausted-project` and `-no_project`, `llama_stream_exhausted-project`), each `assert ['completed_t..._unavailable'] == []`; the other 6 pass at base (controls, and the no-project runs that need no replacement). Both negative cases fail at base: `hosted_tool_completed` is admitted (`assert [] == ['completed_t...turn_lineage']`), `hosted_tool_then_exhausted` is refused with the old code. With the terminal relaxation alone (the recheck removed, temporary edit), both negatives are admitted (`assert [] == [...lineage]`) and the 10 next-send cases pass: the recheck is what keeps the contract.

**Files.** `tldw_chatbook/Chat/console_trace_continuation.py`; `Tests/Chat/test_console_failed_call_retry_ownership.py`; `backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md`; `Docs/User_Guide/console/semantic-trace-capture.md`.

**Runs (round 26, branch `fu/r26-trace1` off dev 75bac413f5, all three tasks; other agents ran beside these, load 7-22).**
- *Soak*, the touched file (`Tests/Chat/test_console_failed_call_retry_ownership.py`, now 92 tests) at `-n 4`: 20/20 rounds, 1,840/1,840 passed, load 6.9-14.8.
- *Lane cost.* The file runs in "Run the Dreams tracking tests and unlaned riders". Serially with that step's flags (`-p no:xdist --timeout=180 --tb=short`): 92 passed in 106.3 s at load 9 (TASK-42472 recorded 59 passed in 26.9 s at load 6). The 35 new cases are Console harness runs of 1-6 s each.
- *Config reads.* The `RecoveryRequired.__init__` recorder over the file (serial, with its `bootstrap_profile` marker): 92 passed, `RR-HITS 0`.
- *Covering*, `-n 4`: every `Tests/Chat/test_console_trace_*.py`, `Tests/Chat/test_console_project_instruction_traces.py`, `Tests/UI/test_console_captured_model_retry.py`, the touched file, `Tests/Chat/test_console_refused_stream_retry.py`, `Tests/UI/test_console_refused_stream_retry_screen.py`, `Tests/Chat/test_rate_limit_reads_as_rate_limit.py` and `Tests/Chat/test_console_provider_failure_copy.py`: 1,630 passed, 33 failed. The 33 are TASK-42513's 32 (`sidecar_replay` 21, `sidecar_redaction` 5, `proprietary_settlement` 4, `graph_gc` 1, `project_turn_service` 1) plus TASK-32032's `test_next_send_after_failed_transformed_call[stopped]`; all 33 fail the same way with the three changed modules at 75bac413f5 swapped in (33 failed). TASK-32032's `[error]` case, red on that base, passes on this branch.
- `Tests/CI` `-n 4`: 1,946 passed, 2 skipped. `Tests/Architecture`, serial: 1,507 passed, 5 skipped. `preflight.sh`: all checks passed. `ruff check tldw_chatbook --select F`: clean. `Tests/lint/ruff_gate.py`: ratchet OK.
- *Lane steps*, via `r19-step.sh`, serially with each step's flags:
  - "Run the Dreams tracking tests and unlaned riders" (`--timeout=180`): 1,558 passed, 6 skipped in 1,050 s;
  - "Run the admission-sensitive Console P0 regression tests" (`--timeout=300`): 529 passed in 1,524 s;
  - "Run admission-sensitive suites 2" (`--timeout=300`; the refused-stream retry file): 1,529 passed in 876 s;
  - "Run fast PR contract" (`--timeout=180`; the size ratchet): 4,871 passed, 4 skipped in 1,131 s.

**Review round 1 (2026-11-24).**
- *C1: a restart turned the evidence into INTERRUPTED.* The tool-history recheck counted only RESPONSE_STARTED and COMPLETE predecessors. Startup recovery (`recover_open_calls`) settles an open RESPONSE_STARTED call as INTERRUPTED; such a call did bear a response, so a tool run left open (third call refused), recovered after a restart, then discarded, was refused at the next send (`completed_tool_turn_lineage`) where dev sends it. `bore_response` (`Chat/console_trace_continuation.py`) now also accepts INTERRUPTED with a response start. Verified in code: `_PERMITTED_CALL_TRANSITIONS` and the `console_trace_calls_lifecycle_guard` trigger allow INTERRUPTED only from RESPONSE_STARTED; the one other way to an INTERRUPTED row is a promoted voice import's insert (`console_trace_calls_insert_reserved`, v72), which is why the state also needs `response_started_at`. The answering-resend check uses the same rule. Test: `test_a_tool_run_interrupted_by_a_restart_and_discarded_still_closes` (project and no project), red before (`assert ['unsupported...turn_lineage'] == ['unsupported_surface_change']`), green after, green at base 75bac413f5.
- *I2: AC#1 was not met for a streamed llama.cpp run with tool history and no project context.* Two new runs in `test_the_message_after_a_captured_run_that_failed_is_sent`: `llama_stream_tool_then_exhausted` (a streamed tool call, then stream and resend 503 for three rounds) and `llama_resend_tool_then_exhausted` (TASK-42530's shape: the resend answers with the tool call, then every call fails). With `NOTES.md` both were refused with `surface_replacement_checkpoint_unavailable`: the factory took the witness origin from the latest call's run, the last resend's own run, so the replacement range never started where the turn's run did, and the validator's resend remap ran only under project context and required the origin to be the resend. Fix: the factory builds the witness from the run origin of the call the resend repeated (`failed_stream_fallback_parent(..., answered=True)`); the validator remaps a resend terminal with or without project context and requires the witness origin to be that call's run origin (it now uses `adjacent_call` instead of its own boundary query). `test_console_trace_project_turn_service.py::test_project_fallback_requires_the_exact_preceding_failed_stream` built its witness with the resend as origin; its `valid` case now names the stream's run origin, and a new `fallback_origin` case (the old shape) is refused.
- *Earlier overclaim, corrected.* The first notes' AC#3 table and test list did not cover a streamed llama.cpp tool run; the next-send test now has 7 runs (14 cases), and the matrix above holds for all of them.
- Red/green and mutations for this round, all three tasks: see TASK-42530's review notes.

**Runs (review round 1, all three tasks; other agents ran beside these, load 7-15).**
- *Soak*, the touched file plus `test_console_trace_project_turn_service.py::test_project_fallback_requires_the_exact_preceding_failed_stream` at `-n 4`: 20/20 rounds, 2,260/2,260 passed (113 each), load 9.3-15.0.
- *Lane cost.* The touched file serially with the Dreams step's flags: 107 passed in 171 s at load 10-13, with the soak and two lane steps running beside it (92 in 106 s at load 9 in the first round). The project-fallback function: 6 passed in 3.8 s at load 7. The `RecoveryRequired.__init__` recorder: `RR-HITS 0` for both.
- *Covering*, `-n 4`, the same file set as the first round: 1,646 passed, 33 failed, the same 33 ids as the first round (TASK-42513's 32 plus TASK-32032's `[stopped]`), all red at 75bac413f5.
- `Tests/CI` `-n 4`: 1,946 passed, 2 skipped. `Tests/Architecture`, serial: 1,507 passed, 5 skipped. `preflight.sh`: all checks passed. `ruff check tldw_chatbook --select F`: clean. `Tests/lint/ruff_gate.py`: ratchet OK.
- *Lane steps*, via `r19-step.sh`, serially with each step's flags:
  - "Run the Dreams tracking tests and unlaned riders" (changed: one new target): 1,579 passed, 6 skipped in 1,066 s;
  - "Run the admission-sensitive Console P0 regression tests": 529 passed in 1,537 s;
  - "Run fast PR contract" (the size ratchet): 4,871 passed, 4 skipped in 1,182 s.
- *Size row.* `console_trace_service.py` 6,809 -> 6,829 (+20), exact measurement, dated comment.

**Runs (re-review, 2026-11-24).** The touched file plus the project-fallback function at `-n 4`: 5/5 rounds, 585/585 passed (117 each), load 12-25. The touched file with `Tests/Chat/test_console_project_instruction_traces.py` and `Tests/Chat/test_console_refused_stream_retry.py`, `-n 4`: 230 passed. `Tests/CI` `-n 4`: 1,946 passed, 2 skipped. `preflight.sh`: all checks passed. `ruff --select F`: clean. `ruff_gate`: ratchet OK.
<!-- SECTION:NOTES:END -->
