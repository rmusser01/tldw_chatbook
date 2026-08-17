---
id: TASK-16788
title: Run-log tools are offered regardless of allowed_tools
status: Done
assignee: []
created_date: '2026-08-16'
updated_date: '2026-08-16 16:59'
labels:
  - agents
  - tools
dependencies: []
priority: low
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify how run-log CALLS are dispatched (does `invoke_tool`'s
   allow-list check ever see them?) before choosing the arm.
2. Record the contract where `allowed_tools` is documented; point at it
   from the filter seam in `agent_service.py`.
3. Pin both halves (offered under an empty allow-list; dispatched without
   reaching `invoke_tool`), RED-first by mutation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decision (AC#1): DOCUMENT, do not filter** — pre-registered in
`Docs/superpowers/specs/2026-08-16-expansion-residue-design.md` and
verified against the code before it was written. The run-log tools are not
a special case: `search_run_log`/`run_log_stats`/`run_log_slice` are
appended to `runtime_schemas` in `AgentService._run_one` AFTER the
allow-list filter, exactly like `spawn_subagent`, `wait_agents`/
`check_agents`, `find_tools`/`load_tools`, `skill_file`, `install_skill`
and `run_skill_script` — the eleven names in `RUNTIME_TOOL_NAMES`, each
offered under its OWN gate. Filtering only the run-log tools would make one
runtime tool behave unlike its family; filtering the family would break
skills and sub-agents (a skill fork's `allowed_tools` is its skill's
intersected tool list, which never contains `find_tools` or the skill-file
tool). Consumers checked: `console_agent_bridge` builds
`allowed_tools` as catalog names + `RUNTIME_TOOL_NAMES` (it already
assumes the runtime layer is separate), and `settings_agents_panel`
excludes runtime names from a user-authored allow-list for the same
reason.

**The dispatch fact that settles it.** `run_agent_loop` gives every
runtime name its own `elif` branch ahead of the generic `deps.invoke_tool`
fallback, so `invoke_tool`'s `call.name not in config.allowed_tools`
refusal *structurally never sees* a run-log call — the allow-list could
not govern the call even if it governed the offer. The single exception is
`spawn_subagent`, whose branch re-checks the allow-list itself and refuses
before dispatch (Q6); that asymmetry is now stated rather than left to be
rediscovered.

**Where it is recorded.** The contract lives on
`AgentConfig.allowed_tools` (`Agents/agent_models.py`) — what it governs
(initial disclosure, `find_tools`, `load_schemas`, `invoke_tool`), what it
does not (the runtime names and their gates), why calls are not caught
later, and what it means for a caller: narrowing `allowed_tools` to
isolate an experiment's tool set is NOT an exhaustive restriction; close
the runtime tool's own gate instead. A pointer comment sits at the filter
seam in `agent_service.py` so an editor there reads it first.

**Pins (AC#2), both RED-proved by mutation and reverted via Edit.**
`Tests/Agents/test_run_log_service_wiring.py`:
`test_run_log_tools_are_offered_under_an_empty_allow_list` runs a primary
turn with `allowed_tools=()` against a native endpoint and asserts the
captured `tools=` payload contains all three run-log names AND that every
offered name is a runtime name — so the test cannot pass by the allow-list
being inert. RED: filtering `runtime_schemas` by `config.allowed_tools`
after assembly → `assert 'search_run_log' in []`.
`test_a_run_log_call_dispatches_although_the_allow_list_is_empty` drives
`run_agent_loop` with an explicit empty allow-list and an `invoke_tool`
that records everything handed to it: the injected handler runs and
`invoke_tool` stays untouched. RED: adding `and call.name in
config.allowed_tools` to the `SEARCH_RUN_LOG_TOOL_NAME` branch → the call
fell through to `invoke_tool` and the handler saw nothing.

**AC#3.** The docstring cites
`Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/report.md` at the
point it explains the consequence for embedders, so the oracle run's
confound (its tool-OFF arm's q3 spending steps on run-log tools and ending
`stuck`) stays discoverable from the parameter a future experiment would
reach for.

**Trade-off accepted.** A programmatic caller still cannot express "no
tools at all" through `allowed_tools` alone. That is now documented with
the alternative (close the gate: `max_subagents=0`, an inactive run-log
writer, etc.) rather than repaired, because the repair would have to be
family-wide to be coherent and the family is load-bearing.

**Modified:** `tldw_chatbook/Agents/agent_models.py`,
`tldw_chatbook/Agents/agent_service.py` (comment only),
`Tests/Agents/test_run_log_service_wiring.py`. Batteries: `Tests/Agents`
1458 passed; ruff on the three files reports only the 4 findings that are
byte-identical at HEAD.
<!-- SECTION:NOTES:END -->

## Description (the why)

Found during TASK-16174's oracle run (PR #1712): `search_run_log` and
`run_log_slice` are appended to the offered tool set as `runtime_schemas`
AFTER the `allowed_tools` filter is applied, so a caller that restricts
`allowed_tools` still sees them. Not a user-permission bypass (the
permission gate is a separate layer) — but it silently widens any
programmatic restriction: in the oracle run's tool-OFF arm, question q3
spent its agent steps on run-log tools and ended `stuck`, a confound for
any experiment that isolates arms by tool set, and a surprise for any
future embedder that passes `allowed_tools` expecting it to be exhaustive.

## Acceptance Criteria (the what)

- [x] A decision is recorded: either `allowed_tools` also filters
      runtime schemas (with existing consumers checked for reliance on
      the current behaviour), or the parameter's docstring states
      explicitly that run-log tools are always offered
- [x] A test pins whichever behaviour is chosen
- [x] The oracle-run harness note in
      Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/report.md is
      referenced so the confound stays discoverable
