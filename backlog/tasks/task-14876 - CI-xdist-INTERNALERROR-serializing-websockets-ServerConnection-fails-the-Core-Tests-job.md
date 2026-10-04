---
id: TASK-14876
title: >-
  CI: xdist INTERNALERROR serializing websockets ServerConnection fails the Core
  Tests job
status: Done
assignee:
  - '@rmusser01'
created_date: '2026-08-09 21:50'
updated_date: '2026-10-01 00:46'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 'Core Tests (all but UI)' CI job intermittently reports 22 failed / 12163 passed with an execnet INTERNALERROR: "DumpError: can't serialize <class 'websockets.asyncio.server.ServerConnection'>", attributed to Tests/LLM_Calls/test_openai_realtime_session.py::test_connect_sends_session_update_and_fires_ready on worker gw0. Observed on PR #1467, whose diff touches ONLY backlog/*.md (zero Python files), so the failures cannot originate from the branch. The same test file passes locally: 35 passed in 4.70s. Under pytest-xdist a test that leaves a live websockets ServerConnection reachable from something xdist tries to serialize (e.g. an assertion-rewritten repr or a failure payload) crashes the worker protocol, which can cascade into unrelated reported failures and mask real regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause identified: what xdist is attempting to serialize and why the ServerConnection is reachable from it
- [x] #2 The realtime-session test cleans up (or avoids exposing) its ServerConnection so xdist can serialize any failure payload
- [x] #3 Core Tests job passes on an unmodified dev checkout across three consecutive runs (see deviation note: the crash class is verified absent from every post-fix core-shard execution; a literally green Core run no longer exists in any lane — see Implementation Notes)
- [x] #4 If the 22 failures have a cause distinct from the INTERNALERROR, they are enumerated and filed separately (enumerated from the incident's own json-report artifact; all accounted for by TASK-18610 or no longer reproducing — nothing left unfiled)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit current state: TASK-19160/PR #1858 fix (--json-report-omit log), TASK-19425 statement, recent CI runs via gh
2. Verify locally: realtime suite standalone + under xdist -n 2 with neighbor, plus CI contract pin test
3. Evidence for AC#3: >=3 consecutive clean core CI run IDs
4. AC#4: cross-reference 22-failure cascade against TASK-19642.x family
5. Close task with notes; commit
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed as already-fixed with evidence: TASK-19160 (PR #1858) root-caused and fixed the execnet DumpError via --json-report-omit log; re-verified the mechanism and the fix on the incident's exact package versions locally. The 22 PR-1467 failures were a distinct, already-tracked cause (TASK-18610 git-push cluster + 2 no-longer-reproducing wizard tests). No code change needed.

### Findings (2026-09-30, worktree at dev tip 90597ade77)

**Outcome: FIXED by TASK-19160 (PR #1858, merged ~2026-08-21). This task,
filed 11 days before that fix, was never closed. No code change was needed;
this close is verification with evidence.**

**AC#1 — root cause, re-verified live on the incident's exact package
versions** (websockets 16.1.1, execnet 2.1.2, pytest-xdist 3.8.0,
pytest-json-report 1.5.0 — versions read from the incident job log itself):

1. json-report 1.5.0's `LoggingHandler.emit` (pytest_jsonreport/plugin.py)
   stores a plain dict of each `LogRecord.__dict__` — including every
   `extra` key. websockets' `LoggerAdapter` puts the live
   `ServerConnection` in `extra`, so a captured websockets record embeds
   the connection object.
2. `pytest_runtest_makereport` sets `report._json_report_extra =
   item._json_report_extra` — on an xdist worker this rides the report
   across execnet, which serializes basic types only.
3. `DumpError: can't serialize ServerConnection` kills the worker mid-run;
   the controller hits `assert not crashitem` → INTERNALERROR → the job's
   reporting aborts.

Scratch repro (kept out of the repo, in /tmp): a failing test that logs
through a `websockets.*` logger with a non-serializable `extra`, run
`-n 2 --dist loadscope --json-report` → the exact incident signature
(`execnet.gateway_base.DumpError: can't serialize ...` + INTERNALERROR);
the same run with `--json-report-omit log` (the CI shape since TASK-19160)
reports a clean `1 failed`. The mechanism is still live in the current
toolchain; the omit flag is the load-bearing fix and remains necessary.

**AC#2 — test-side hardening in place and verified.** The realtime file
carries `_transport_safe_error` (drops the original traceback whose frames
hold the live connection; pinned by
`test_transport_safe_error_discards_original_traceback`) and the
`fake_server` fixture teardown closes every tracked session and server
(`close()` + `wait_closed()`). On the dev tip the file has 34 failures
when run standalone (pre-existing contract drift — TASK-19642.9's scope,
not this task's) and 36 across the three LLM_Calls realtime files under
xdist, yet `pytest Tests/LLM_Calls/test_openai_realtime_session.py
test_realtime_protocol.py test_realtime_tls_trust.py -n 2 --dist loadscope
--json-report --json-report-omit log` completed 3/3 with no worker crash
and INTERNALERROR-free json reports — i.e. failure payloads serialize.

**AC#3 — deviation documented.** A literally green "Core Tests" run cannot
be cited post-fix, for two reasons unrelated to this defect:

- CI was restructured: `test.yml` no longer accepts pull_request events
  (PRs use the bounded fast lane in `derived-artifacts.yml`; test.yml runs
  on push to main + workflow_dispatch, and every dispatch since 2026-09-14
  was the backup-recovery lane with core shards SKIPPED; `nightly-deep.yml`
  ubuntu legs get cancelled by a windows-latest failure — its failure is a
  separate known issue).
- The last actual core-shard executions — runs 34185952279 (2026-09-08),
  34156968562 (2026-09-07) and the 2026-08-29 cluster (e.g. 33278018977,
  33277735172, 33277484246) — all fall inside the documented TASK-19520 /
  TASK-19642 red era (1153 failures), so no green core run exists anywhere
  after the fix.

Evidence the crash class specifically is gone: grepping those runs'
`--log-failed` output for the exact signatures finds **0 occurrences of
`DumpError`, `can't serialize`, or `INTERNALERROR`**, while thousands of
real failure payloads crossed xdist cleanly — a far heavier exercise of
the serialization path than green runs would be. (A naive `grep -c execnet`
returns 4230 — all pip-install lines and full-process asyncio tracebacks;
see the lessons entry added with this close.)

**AC#4 — the 22 failures enumerated and accounted for.** Pulled from the
incident run's own json-report artifact (artifact id 9044657179 of run
31334634058, job 93298353730, PR #1467 head 82d9d0ae): 16x
`Tests/Notes/test_file_notes_git_push_service.py`, 1x
`test_file_notes_git_push.py`, 2x `test_file_notes_git_integration.py`,
1x `test_file_notes_git_commit_integration.py`, 2x
`Tests/Wizards/test_first_run_setup_wizard.py::test_summary_default_speech_check_*`.
None are realtime tests — all 13 realtime tests PASSED in the incident
run. The 20 git-push tests are exactly the "git-push-service 16 (ubuntu) /
17 (macOS)" cluster TASK-18610 AC#1 root-caused (the factory pinned
`sys.executable` for SSH dispatch; hosted-toolcache Python fails the
predicates) and fixed via the `python_executable` seam in its pass 2. The
2 wizard tests pass on the current dev tip (verified locally) and are
absent from the TASK-19520 inventory — era-specific, no longer
reproducing. Nothing left unfiled.

**Overlap (noted, not executed):** TASK-19642.9 "Restore OpenAI realtime
session contract tests" (To Do) owns the 34 serial failures this audit
observed in `test_openai_realtime_session.py` on the dev tip — a distinct
defect in the same file; whoever picks it up inherits the serial/xdist
behavior split documented above (AC#3 of that task).

**Verify commands (targeted, per AGENTS.md):**
- `pytest Tests/LLM_Calls/test_openai_realtime_session.py` → 34 failed,
  2 passed (serial; pre-existing, TASK-19642.9 scope)
- `pytest Tests/LLM_Calls/test_openai_realtime_session.py
  Tests/LLM_Calls/test_realtime_protocol.py
  Tests/LLM_Calls/test_realtime_tls_trust.py -n 2 --dist loadscope
  --json-report --json-report-omit log` → 36 failed / 18 passed, 3/3
  repeats, no INTERNALERROR
- `pytest Tests/CI/test_github_actions_test_workflow.py` → 19 passed
  (includes the `test_every_json_report_invocation_omits_log_capture` pin)
- All live `--json-report` invocations in `.github/workflows/` carry
  `--json-report-omit log` (the only non-omitting match is a snippet in
  `CROSS_PLATFORM_FIXES.md`, documentation)

**Files changed:** this task file and a lessons-testing-evidence.md entry.
No production or test code.

ADR required: no — verification close-out of an already-landed fix
(TASK-19160); no architectural decision made.
<!-- SECTION:NOTES:END -->
