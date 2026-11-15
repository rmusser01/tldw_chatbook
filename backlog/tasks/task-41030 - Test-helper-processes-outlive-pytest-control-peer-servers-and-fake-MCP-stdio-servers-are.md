---
id: TASK-41030
title: "Test helper processes outlive pytest: control-peer servers and fake MCP stdio servers are orphaned to launchd"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- ci
priority: medium
references:
- Tests/Chat/test_console_ssh_custody.py
- Tests/Backup_Recovery/test_activation_mcp.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two test helpers start child processes that nothing stops when the test fails, times out, or loses its pytest process. These children are reparented to ppid 1 and run until someone kills them by hand:
- **The control-peer server.** `Tests/Chat/test_console_ssh_custody.py::test_unknown_peer_proof_keeps_original_model_worker_and_storage[*]` reruns itself in a child pytest (`subprocess.run(..., timeout=40)`). In the `ambiguous_master` mode, the fake ssh `launch` starts `control-peer.py serve` with `start_new_session=True`, and that server exits only when it receives an `exit` message.
- **The fake MCP stdio server.** The `_REQUEST_SETTLEMENT` / `_FAILED_STOP` family in `Tests/Backup_Recovery/test_activation_mcp.py` (for example `test_mcp_failed_termination_retains_admission_until_natural_exit`) loops on `while not Path(sys.argv[3]).exists(): time.sleep(.01)`. It never exits if the test does not write the release file.

Evidence:
- On 2026-11-15 this machine held 12 such orphans with ppid 1 from other sessions' runs. The oldest was 1 day 20 h old.
  - 10 were `control-peer.py serve` processes under `test_unknown_peer_proof_keeps_*` basetemps, each still holding its socket under `/tmp/tldw-cs-test-*`.
  - 2 were MCP stubs under `test_mcp_failed_termination_re*`.
- The architecture-review coordinator killed 4 leaked MCP stubs on 2026-11-07, and 7 more orphans from its own runs on 2026-11-13.

On 2026-11-15 a verification run of the six unknown-peer-proof params on dev left one more `control-peer.py serve` behind (all six children hit their 40 s timeout; TASK-40716's parked workers are a likely reason a child cannot exit). Each orphan keeps its basetemp directory alive, and the control-peer servers also leave `/tmp/tldw-cs-test-*` directories. They were also alive each time the Agents covering set or its 85 outside files stalled (TASK-41029).

Why medium, not high: a GitHub Actions runner kills a job's leftover processes when the job ends, so these orphans do not hang CI. The cost falls on developer machines and long local runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After either file runs, whether it passes, fails or is interrupted, no process it started is still running
- [ ] #2 A case that fails or times out still stops its helper process, shown by forcing one failure
- [ ] #3 The task notes list any other test helper found that starts a detached or new-session process, and each one is fixed or filed
<!-- AC:END -->
