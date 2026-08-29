# TASK-19506 runaway-pytest diagnostic evidence

Date: 2026-08-21
Base under test: rebased latest dev at `5f720a40417eaa78f33619d5cbc82effc470104b`

## Outcome

The historical 14-hour pytest process could not be reproduced or attributed to a
current application/fixture lifecycle defect. Its process had already exited before
the active node or stacks were captured. No AgentService or fleet runtime change was
made.

The instrumented suspected surface completed **380/380 tests in 78.06 seconds** with
a 30-second per-test timeout. Every teardown returned the classified project-owned
thread inventory (`tool-*` and `fleet-*`) to the session baseline within a three-second
settle bound.

## Suspected surface and command

The surface combines the AGENTS.md project-instruction tests merged immediately before
the incident with the AgentService timeout, review-scope, and fleet lifecycle suites
that own the only production-named agent threads:

- `Tests/Agents/test_project_instruction_concurrency.py`
- `Tests/Agents/test_project_instruction_path_targets.py`
- `Tests/Agents/test_project_instruction_performance.py`
- `Tests/Agents/test_project_instruction_resolver.py`
- `Tests/Agents/test_project_instruction_resolver_properties.py`
- `Tests/Agents/test_project_instruction_runtime.py`
- `Tests/Agents/test_agent_runtime_preparation.py`
- `Tests/Agents/test_agent_runtime_review_hook.py`
- `Tests/Agents/test_agent_service_review_state_scope.py`
- `Tests/Agents/test_agent_service.py`
- `Tests/Agents/test_fleet_runtime.py`

The exact command was:

```bash
.venv/bin/python -m pytest -p Tests.Performance.pytest_thread_diagnostics \
  Tests/Agents/test_project_instruction_concurrency.py \
  Tests/Agents/test_project_instruction_path_targets.py \
  Tests/Agents/test_project_instruction_performance.py \
  Tests/Agents/test_project_instruction_resolver.py \
  Tests/Agents/test_project_instruction_resolver_properties.py \
  Tests/Agents/test_project_instruction_runtime.py \
  Tests/Agents/test_agent_runtime_preparation.py \
  Tests/Agents/test_agent_runtime_review_hook.py \
  Tests/Agents/test_agent_service_review_state_scope.py \
  Tests/Agents/test_agent_service.py \
  Tests/Agents/test_fleet_runtime.py \
  -vv --tb=short --timeout=30 \
  --thread-diagnostics-jsonl=Docs/superpowers/qa/runaway-pytest-2026-08-21/suspected-surface.jsonl \
  --thread-diagnostics-interval=0.25 \
  --thread-diagnostics-settle=3 \
  --thread-diagnostics-stack-after=20 \
  --thread-diagnostics-strict
```

Raw report: `suspected-surface.jsonl`

## Measurements

| Metric | Result |
|---|---:|
| Tests | 380 passed |
| Wall time | 78.06 s |
| JSONL records / periodic samples / teardown assertions | 1,829 / 306 / 380 |
| Recorded call outcomes | 380 passed, 0 skipped/failed/xfail |
| RSS at session start | 86.17 MiB |
| Observed peak RSS | 226.62 MiB |
| RSS at session finish | 226.16 MiB |
| Observed peak total Python threads | 5 |
| Observed peak project-owned threads | 1 (`tool-slow_tool`) |
| Teardowns with a surviving project-owned thread | 0 |
| Ownership failures | 0 |
| Test phases exceeding the 20 s stack threshold | 0 |

Observed peak RSS occurred during setup for
`test_child_tool_call_binds_the_childs_own_run_id`. The five
timeout/cancellation tests intentionally leave their
daemon `tool-*` worker alive after the wrapper returns; the strict diagnostic waited
for those bounded fake workers to finish. Their longest observed teardown settle was
2.25 seconds, and all returned to baseline. This is expected test behavior, not an
unbounded thread leak.

## Historical Event-wait control

The TASK-3316 test name recorded in the old task was subsequently renamed to
`test_file_notes_collections_source_transition_blocks_mutation_through_targeted_reconcile`.
The current node and the AST guard for unbounded background-signal waits ran with a
10-second per-test timeout:

- 8 passed in 5.81 seconds;
- observed peak RSS 384.55 MiB;
- observed peak total threads 3;
- zero project-owned survivors;
- zero phases reached the five-second stack threshold.

Raw report: `known-event-wait-control.jsonl`

The stack-capture path was separately exercised against the guard's intentionally slow
AST scan with a 0.2-second threshold. It retained two Python stack snapshots (setup and
call), each including `MainThread`, before the test completed normally. Repository,
virtual-environment, interpreter-install, and home prefixes are normalized to
`$REPO`, `$VENV`, `$PYTHON`, and `$HOME` in retained stacks.

Raw report: `stack-capture-control.jsonl`

## Host residue cleanup

Read-only process inspection found no surviving copy of the original runaway pytest.
It did find one unrelated application process launched from the obsolete
`.worktrees/rag-15810-hang` checkout:

- PID `79581`;
- started 2026-08-15 14:31:49 PDT;
- age 5 days 22 hours at inspection;
- 604 minutes cumulative CPU, 2.8% current CPU, 233,632 KiB RSS;
- cwd `.worktrees/rag-15810-hang`.

After revalidating the exact PID and cwd, the process was sent `SIGTERM`; a subsequent
`ps -p 79581` returned no process. No files or worktrees were removed. Recovery is to
launch the application again if that obsolete checkout is intentionally needed.

## Decision

There is no current reproduction to prefix-bisect, no producer task whose state needs
repair, and no deterministic evidence justifying an AgentService/fleet shutdown change
or a new bound around application code. The opt-in diagnostic remains available for a
future reproduction; because it writes and flushes JSONL samples continuously, a
pytest-timeout process termination retains the last current-test, RSS, thread inventory,
and any threshold-triggered Python stack snapshot.

The retained reports were regenerated after review hardening changed baseline ownership
from reusable native thread identifiers to thread-object identity, attributed the final
inventory to the session-finish phase, and fenced monitor writes so `session_finish` is
always the terminal JSONL record. All three final records report `monitor_stopped=true`.

ADR required: no. This task adds diagnostic test tooling and evidence only; it does not
change runtime ownership or application lifecycle.
