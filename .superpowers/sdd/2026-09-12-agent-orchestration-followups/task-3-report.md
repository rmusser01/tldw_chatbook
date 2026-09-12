# Task 3 report: old orchestration harness reconciliation

## Scope and outcome

- Test-only changes; no production files changed.
- TASK-2155: both dictionary send harnesses now create and hydrate the real
  durable Console Library policy before Send. The agent branch gained the same
  raw persisted USER-text assertion as the provider branch.
- TASK-22720: removed only the stale non-strict xfail. All replacement,
  completion, repair-dispatch, and visible-output assertions remain intact.
- TASK-19642.8.3: the four inventory nodes required no behavior change. Updated
  the headless test docstring to describe the frozen `run_budget` argument and
  ADR-134's additional automatic-chain allowance.
- Task 2 review follow-up: the revoked-arm regression now requires
  `observed == []`, proving the configuration callback is never entered.
- All three backlog tasks remain **In Progress** for root's independent review.

## Root-cause evidence

The pre-edit dictionary module produced 2 failures. Both durable commits were
refused because the manually bound conversations had no persisted policy row.
Root's probes further isolate the setup defect:

- `dictionary-durable-probe.log`: `RuntimeError: Durable Console Library policy
  no longer matches acceptance.`
- `dictionary-hydration-probe.log`: hydration alone still failed.
- `dictionary-policy-probe.log`: repository insert plus hydration passed.
- `stale-harness-probes.log`: the same dictionary failure, four passing wake
  nodes, and the citation regression XPASS.

The test helper uses the real repository seam, asserts policy revision 1, and
hydrates the real session holder. Durable commit is neither bypassed nor made
ephemeral. Dictionary substitutions remain limited to provider/agent payloads;
stored transcript content remains the literal raw draft.

## Verification

Interpreter for every pytest command:
`.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`.

1. Before edit:
   `-m pytest -q Tests/UI/test_console_dictionary_send_integration.py`
   — **2 failed**, one expected dependency warning.
2. Citation selection after edit:
   `-m pytest -q Tests/Chat/test_console_local_citation_boundary.py -k citation_repair_agent`
   — **7 passed, 96 deselected**, one dependency warning.
3. Final dictionary, revocation, and four exact wake nodes:
   `-m pytest -q Tests/UI/test_console_dictionary_send_integration.py Tests/UI/test_console_mcp_approval.py::test_revoked_arm_is_refused_before_configuration_read Tests/Chat/test_console_fleet_wake_safety.py::test_a_wake_defers_behind_a_pending_card_and_cannot_resolve_it Tests/Chat/test_console_fleet_wake_safety.py::test_a_wake_dispatches_run_reply_under_the_same_authority_as_manual Tests/Chat/test_console_fleet_wake_safety.py::test_a_woken_turns_gated_tool_still_raises_the_approval_card Tests/Chat/test_console_headless_wake_invariants.py::test_a_headless_wake_takes_the_same_agent_dispatch_and_budget`
   — **8 passed**, one dependency warning. The eight cases are 2 dictionary, 2
   parametrized revocation, and 4 wake cases.
4. `scripts/check_backlog_task_ids.py` — no duplicate IDs across 3,852 task
   files; all paths Windows-compatible.
5. `git diff --check` — clean.

## Static baseline debt

Ruff was run against all four changed Python modules. It reported 30 existing
findings across the legacy modules: import ordering, unused tuple bindings,
quoted annotations, broad cleanup exception handling, one implicit collection
string concatenation, and one list-index preference. None point at a newly
added or behavior-edited line. `ruff format --check` reports all four modules
already require whole-file formatting; its displayed hunks are pre-existing
format debt outside this batch's changed lines. The batch does not broaden
scope into the repository's Ruff cleanup tasks.

Pytest also emits the existing RequestsDependencyWarning for the environment's
urllib3/chardet combination and intermittent pytest temp-directory cleanup
warnings for `test_kokoro_constructor_direct3`; neither affects the focused
results.

## ADR decision

- TASK-2155 and TASK-22720: ADR required: no. These are test harness and stale
  marker repairs under existing production contracts.
- TASK-19642.8.3: no new ADR. Existing
  `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md` governs
  the additional automatic-chain allowance.
