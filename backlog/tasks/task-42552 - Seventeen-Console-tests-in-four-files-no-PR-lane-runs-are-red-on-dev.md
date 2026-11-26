---
id: TASK-42552
title: "Seventeen Console tests in four files no PR lane runs are red on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-26 04:50
updated_date: '2026-11-26 06:10'
labels:
- follow-up
- console
- dev-red
priority: medium
references:
- Tests/Chat/test_console_continuation_group_merge.py
- Tests/Chat/test_console_named_agent_model.py
- Tests/Chat/test_console_redirect_stream_cut.py
- Tests/Chat/test_console_request_provenance_routes.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 26 (TASK-42525's wide runs, 2026-11-25). 17 tests in `Tests/Chat/test_console_continuation_group_merge.py`, `Tests/Chat/test_console_named_agent_model.py`, `Tests/Chat/test_console_redirect_stream_cut.py` and `Tests/Chat/test_console_request_provenance_routes.py` fail identically on the round-26 stack and with the stack's tests on the base's source. No PR lane runs these four files, so nothing reports them, and a real regression in continuation grouping, named-agent model choice, redirect stream cuts or request provenance routes would read as old noise. It is not yet known whether the product or the tests are wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For each red test, the notes say whether the product or the test is wrong, with the cause and the commit that introduced it
- [x] #2 A product defect is fixed and its test shown red before and green after; a drifted test is repaired with every behaviour assertion kept
- [x] #3 All four files pass on dev at `-n 4`, twice in a row
- [x] #4 The four files run on a PR lane that blocks merges to dev, timed with that lane's flags
<!-- AC:END -->

## Implementation Plan

1. Run the four files on the base at `-n 4` and record the exact red set.
2. Per red node, read the traceback, find the cause and the commit that introduced it (blame, `git log -S`, a static replay of the census check at old commits), and decide product or test.
3. Repair each: sandbox trap -> module `bootstrap_profile`; drifted doubles -> match the shipped contract, assertions kept; the route/callsite censuses -> describe the product as it is, keyed by function rather than line.
4. Lane the four files in recovered-suites (sandboxed census file at the end of the sandboxed block, the bootstrap ones after the variant-stream file), time each with the step's flags, update `RECOVERED_SUITES_TARGETS`.
5. Two `-n 4` runs, soak, the brief's before-reporting checks.

## Implementation Notes

All 19 nodes in the four files pass. The red set at base 6e41919880 (`-n 4`) was 17 failed, 2 passed: all 10 in `continuation_group_merge`, 2 in `named_agent_model`, 3 in `redirect_stream_cut` and 2 in `request_provenance_routes`. Per node:

- **`continuation_group_merge`, all 10: test drift, 45cf929ac6 (TASK-42472, 2026-11-23).** Before each primary call, `chat_call_impl` now reads the message it streams into (`self._store.get_message(...)`) so a retried call can roll back what a failed call streamed. The double was `SimpleNamespace(session_id_for_message=...)`, so every case died with `AttributeError: ... no attribute 'get_message'` before it reached the merge. The double now has `get_message={}.__getitem__`. Its KeyError is the store's own "nothing streamed yet". Assertions are unchanged.
- **`redirect_stream_cut` (3) and `named_agent_model` (2): the sandbox trap.** Every adapter call asks `_is_subagent`, which resolves the `agents.subagent_system` internal prompt through `get_cli_setting` in-process. That read dates from 4f2c0db065 (2026-07-21). Under the per-test config sandbox it raises `RecoveryRequired(raw_source_selection_changed)`. That guard arrived with TASK-32628 (b5251e9a6e, 2026-09-16); `git log -S` finds the error string's first appearance there, and `Tests/conftest.py`'s note dates this signature to it. The fix is a module `pytestmark = pytest.mark.bootstrap_profile` with the reason beside it, as in TASK-42513. Behind the trap, `redirect_stream_cut` passed. `named_agent_model` still had two drifted doubles:
  - `test_concurrent_worker_override_is_call_local_and_keeps_parent_continuation`: test drift, bf1886c95c (2026-10-28). The primary now filters its constructor continuation sidecar by `owner_message_id` before preparing, and the double's sidecar was the string `"parent-sidecar"` (`AttributeError: 'str' object has no attribute 'owner_message_id'`). The sidecar is now an object with an owner id that no request row selects. The assertion that it reaches `prepare_chat_request` unchanged is kept.
  - `test_capture_on_worker_uses_its_model_and_keeps_parent_private_history`: test drift, 45b4494127 (TASK-41310 B, ADR-337/092 amendment, 2026-11-17). A Capture-On child call's route now names the spawning turn's saved user revision, and the factory fails closed without one (`ValueError: trace_turn_unavailable`). The test's admitted parent request had no saved user row. That row is now a `SavedRevisionTraceProvenance`, as every real parent turn's is. Assertions are unchanged.
- **`request_provenance_routes` (2): the product's census was wrong, and the test keyed it by line number.** `CONSOLE_REQUEST_ROUTE_CENSUS` and `CONSOLE_GATEWAY_CALLSITE_CENSUS` have not changed since c78f641adf (2026-08-30), and they were red the day they landed. A static replay of both checks at c78f641adf (AST over `git cat-file` blobs; it matches the real test's diff exactly at the base) puts all 13 route markers off their pinned lines and finds `summarize_manual`'s auxiliary call undocumented. `route=ConsoleRequestRoute.MANUAL_SUMMARY` never existed in any dev commit (`git log -S`). Later drift added to that:
  - the agent call moved to `console_streaming_request.py` (7ebe7e585a);
  - compaction moved into `_summary_completion` behind `**kwargs` (7d587a6d64);
  - new callsites arrived: speculative voice's Capture-On fresh dispatch (b7dd8e53f2), three response-rules helper calls (d24c4c04b2) and the Conversation-settings generation probe (939dee8dc2).

  The fix makes the census describe the shipped code (`Chat/console_trace_provenance.py`):
  - The `source_line` fields are gone; rows are keyed by module and function.
  - Manual summary is recorded as Capture Off on `_summary_completion`, beside automatic compaction. It has always dispatched there, and `_complete_auxiliary` refuses any route but automatic compaction.
  - Voice is a second FRESH callsite. Response rules are Console-owned and Capture Off; the probe is excluded.

  The test (`Tests/Chat/test_console_request_provenance_routes.py`) changes in step:
  - It compares a Counter of (module, function, gateway, binding).
  - A route's marker must be a line inside its named function.
  - Discovery records `route=` text, `**name` for a splat, or `""`.
  - A new assertion requires every traced callsite to bind a route.
  - Route uniqueness admits exactly FRESH's documented second callsite.
  - The capture assertion names the two Capture-Off routes.

  Every other assertion is kept. Red/green: the repaired test against the base census fails both census tests and passes the other two; with the new census, all 4 pass. `Docs/Development/console-request-provenance-route-census.md` says the same.

Lane (AC#4): all four files go to recovered-suites ("Run the recovered suites"). The census file is sandboxed: a `RecoveryRequired.__init__` recorder counted 0 over it, so it ends the sandboxed block. The three bootstrap_profile files follow `test_console_variant_stream.py`. `RECOVERED_SUITES_TARGETS` is updated (`Tests/CI/test_ci_queue_pressure_contract.py`). Timings, each file serially with the step's flags (`--storage-lease-guard --timeout=180`) at load 34-46:

| File | Time | Tests |
| --- | --- | --- |
| request-provenance routes | 30.7 s | 4 (the AST walk of the whole package dominates) |
| continuation group merge | 3.6 s | 10 |
| named agent model | 4.1 s | 2 |
| redirect stream cut | 3.5 s | 3 |

The whole step, serially with its flags, gave 2041 passed, 1 skipped and 1 deselected in 2978.7 s, at load 26-60 with other runs beside it. The four files took about 59 s of that.

Runs:
- the four files at `-n 4`, twice: 19 passed each;
- a soak of 20 runs at `-n 4` (the four files, the provider-gateway file and the CI contract file) at load 35-60: 20/20 green, 678 passed each;
- `Tests/CI` at `-n 4`: 2009 passed, 2 skipped.

Files:
- `Tests/Chat/test_console_{continuation_group_merge,named_agent_model,redirect_stream_cut,request_provenance_routes}.py`;
- `tldw_chatbook/Chat/console_trace_provenance.py`;
- `Docs/Development/console-request-provenance-route-census.md`;
- `.github/workflows/derived-artifacts.yml`;
- `Tests/CI/test_ci_queue_pressure_contract.py`;
- `backlog/docs/lessons-testing-evidence.md` (the line-keyed census lesson).
