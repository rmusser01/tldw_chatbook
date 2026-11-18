---
id: TASK-41508
title: "About 198 Console trace test nodes are red on dev and on no lane: PreparedProviderRequest rejects cache_boundary"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 08:31
labels:
- follow-up
- tests
- console
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_prepared_request.py
- Tests/Chat/test_console_project_instruction_traces.py
- Tests/Chat/test_console_trace_failed_tool_run.py
- Tests/Chat/test_console_trace_discarded_tool_run.py
- Tests/Chat/test_console_agent_call_thinking.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the trace stream (round 15). On dev these fail with one cause, a `TypeError` for `PreparedProviderRequest(cache_boundary=...)`: `cache_boundary` is declared `field(default=None, init=False, repr=False)` (`console_prepared_request.py:708`, added by 7970fbcada), so test code constructing it with the keyword breaks.

**Counts.** `test_console_project_instruction_traces.py` 67, `test_console_trace_failed_tool_run.py` 51, `test_console_trace_discarded_tool_run.py` 31, `test_console_agent_call_thinking.py` 15 failed plus 8 errors. None of the four is on a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The four files pass under a plain run, by changing the test builders to the current constructor
- [x] #2 The four files are on a PR lane, with their time recorded
- [x] #3 No `init=False` field is added or removed to make tests pass without a decision recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Run the four files plainly and under the bootstrap profile; classify every failure (the description's cause is a guess from TASK-41133's notes).
2. Fix each cause in the tests: profile marker, test builders/doubles to the current signatures, stale identity assertions.
3. Fix the one real `PreparedProviderRequest(cache_boundary=)` builder to the current constructor (no init=False change).
4. Lane the four files with their measured serial time under the lane's flags.

## Implementation Notes

**The cause was mostly not `cache_boundary`.** Plain run at base (`-n 4`, the four files plus `test_console_trace_provenance.py`): 165 failed, 8 errors, 55 passed. 155 of them were `RecoveryRequired('raw_source_selection_changed')`: the files had no profile marker and the Console bridge reads its tool gates through config. The one `PreparedProviderRequest(cache_boundary=)` TypeError was in a fifth file, `test_console_trace_provenance.py::test_explicit_prepared_request_constructor_rejects_missing_on_provenance` (TASK-41133's covering-run note listed both causes together). Under the bootstrap profile 14 stayed red, each a stale test double or assertion:
- `test_transformed_project_sends_preserve_source_and_context` (6): the dictionary-applier double took two arguments; the controller passes the turn's frozen inputs as a third. The TypeError was swallowed and the send kept "alias".
- `test_resumed_child_retains_its_final_call_thinking_only` (8): it asserted `is envelope`, but retention deep-copies the transcript (827b622638, a day after the test). It now asserts equality. Plain, these 8 also ended `'error' == 'done'` with teardown errors.

**Changes (tests only).**
- All four files: `pytestmark = pytest.mark.bootstrap_profile`, like their sibling trace files (`test_console_trace_model_retries.py`, `test_console_trace_surface_identity.py`). The agent-call thinking file's single per-test marker became the module marker.
- The applier double takes the current signature.
- The provenance builder passes only `init` fields (`dataclasses.fields`).

**AC#1, met with a deviation.** The method the AC names (changing builders to the current constructor) fixes only 7 nodes: the applier double, and the provenance builder in the fifth file. Most failures needed the profile marker. **AC#3, met.** No `init=False` field was added or removed.

**Red/green.** Base as above. After the changes, the same five files plain at `-n 4`: 220 passed in 202 s (load ~16).

**AC#2, lane.** All four files run whole in `console-p0-gate`'s admission-sensitive step (a bootstrap invocation), placed before the PR3023 nodes, which stay last. The agent-call thinking file's two nodes moved there from `pr-fast-lane-admission-2`, which runs at 29-35 min of its 35. Serially with the step's flags (`--timeout=300 --tb=short`, no xdist) at load 18-35: 175 passed in 578 s. Per file (setup+call+teardown): failed tool run 212 s, project instruction traces 186 s, discarded tool run 155 s, agent-call thinking under 2 s. The step measured 9.7-13.1 min on CI in the five latest runs of its 35. The whole step, run serially as CI runs it (one process, in its order, with the skill-discovery capture file of TASK-41511): 453 passed in 1464 s at load 15-27, of which about 11 min is the five new files. `Tests/CI/test_response_rules_critical_gate.py`'s `INHERITED` and `test_ci_queue_pressure_contract.py`'s `ADMISSION_2_TARGETS` follow the move.

Files: `Tests/Chat/test_console_project_instruction_traces.py`, `test_console_trace_failed_tool_run.py`, `test_console_trace_discarded_tool_run.py`, `test_console_agent_call_thinking.py`, `test_console_trace_provenance.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_response_rules_critical_gate.py`, `Tests/CI/test_ci_queue_pressure_contract.py`.
