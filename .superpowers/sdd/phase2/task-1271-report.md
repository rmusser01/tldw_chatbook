# TASK-1271 — Run log Phase 2: aggregation, slicing, cross-run search

## Status
Implemented and committed. Task marked Done in backlog (`backlog task 1271`).

## What shipped

- `run_log_stats` — bounded aggregation over the current run's log: counts,
  error counts, and content-byte totals grouped by `tool`/`type`/`status`/
  `kind`, plus optional pre-filters (`tool`, `type`, `status`, `kind`,
  `from_record`, `to_record`). Output is one line per DISTINCT group value,
  never one line per record — bounded by construction, `group_by` is
  restricted to the four metadata fields whose cardinality cannot grow
  with the log. No token totals: `RunLogRecord` carries no per-record
  token count (only a whole-run total, in MANIFEST, after the run ends),
  and fabricating an estimate would silently disagree with the run's own
  authoritative accounting — `content_bytes` is the honest substitute.

- `run_log_slice` — retrieves a contiguous record-number range as one
  unit. Capped at `MAX_SLICE_RECORDS` (50) regardless of requested range
  width or log size. Reuses `format_results` for per-record rendering
  (no second renderer, so the TASK-1250 "match not in the rendered body"
  defect class can't be reintroduced in a new place).

- Both follow the established runtime-tool pattern exactly, mirroring
  `search_run_log`: name constants in `agent_models.py` +
  `RUNTIME_TOOL_NAMES`, schemas in `tool_catalog.py`, optional `LoopDeps`
  fields defaulting to `None` in `agent_runtime.py` plus dispatch branches
  guarded by `deps.X is not None`, and service wiring in
  `agent_service.py` gated identically to `search_run_log` — schema
  disclosure under the three-part `log_active` gate (primary agent, log
  active, at least one other disclosable schema), `LoopDeps` wiring under
  `agent_kind == AGENT_KIND_PRIMARY`.

- Cross-run search: investigated, **deferred to task-1273**. Run
  directories are named by bare `run_id` under
  `<root>/.agent-runs/<run_id>/` with no per-conversation grouping on
  disk; MANIFEST records only run-level metadata (no `conversation_id`,
  no resolved root). `AgentRunsDB.agent_runs` maps `conversation_id` ->
  `run_id` (needed), but has no column for which root `resolve_log_root()`
  chose for a given run — that choice is resolved fresh every run and
  never persisted. Locating a historical run's log from its `run_id`
  alone therefore requires either assuming the current workspace binding
  matches every historical run (silently wrong when it doesn't) or a
  schema addition. Per spec §2.2 and this task's own instruction not to
  improvise a DB change, filed as task-1273 rather than built here.

## Files changed

- `tldw_chatbook/Agents/agent_models.py` — name constants, `RUNTIME_TOOL_NAMES`
- `tldw_chatbook/Agents/run_log_search.py` — `compute_stats`, `format_stats`,
  `slice_records`, `format_slice`, `STATS_GROUP_BY_FIELDS`,
  `DEFAULT_SLICE_WIDTH`, `MAX_SLICE_RECORDS`
- `tldw_chatbook/Agents/tool_catalog.py` — `RUN_LOG_STATS_TOOL_SCHEMA`,
  `RUN_LOG_SLICE_TOOL_SCHEMA`
- `tldw_chatbook/Agents/agent_runtime.py` — `LoopDeps` fields, dispatch
  branches
- `tldw_chatbook/Agents/agent_service.py` — real closures, schema/deps
  wiring, `RUN_LOG_PROMPT_SECTION` extended
- `Tests/Agents/test_run_log_search.py` — pure-function coverage, extended
  (18 new tests)
- `Tests/Agents/test_run_log_stats_slice_runtime_tools.py` — new (49 tests):
  schema registration, loop dispatch, sub-agent isolation, junk-args via
  the real closures, empty/single/multi-segment logs, boundedness on a
  large synthetic log
- `Tests/Agents/test_agent_models.py`,
  `Tests/Agents/test_install_skill_runtime_tool.py` — `RUNTIME_TOOL_NAMES`
  exhaustive-set pins updated (same maintenance every prior runtime-tool
  addition required in these two files; both were confirmed to be the
  established pattern before touching them)
- `backlog/tasks/task-1271` — plan, notes, ACs checked, status Done
- `backlog/tasks/task-1273` — new, cross-run search follow-up

## Mutation test (primary-agent gate)

Temporarily removed the `agent_kind == AGENT_KIND_PRIMARY` conditional on
`run_log_stats`/`run_log_slice` in `agent_service.py`'s `LoopDeps` wiring
(wired both unconditionally). Ran
`test_subagent_cannot_call_run_log_stats_or_run_log_slice` — it FAILED
(the child's calls were no longer refused; assertion on
`"Tool not permitted: run_log_stats"` in the child's tool_result steps
came back false). Restored the gate; the test passes again. Confirmed via
`git diff` that no mutation-test artifact remains in the committed diff.

## Test results

- `Tests/Agents/` (venv pytest, PLAIN): 601 passed (baseline was 533; +68
  new/updated tests). Zero failures, zero errors.
- `Tests/Chat/` (venv pytest, PLAIN): see final report to caller for exact
  numbers — run against the known pre-existing baseline (4 failures in
  `test_chat_functions.py`, 13 errors in `test_scope_picker_listers.py`).
  Nothing outside those two files should be new.

## Open items

- task-1273 (cross-run search via `AgentRunsDB` root tracking) — filed,
  not started.
- No pre-existing test was edited to make a genuine failure disappear.
  Two pre-existing tests (`test_agent_models.py::test_runtime_tool_names`,
  `test_install_skill_runtime_tool.py::
  test_install_skill_name_in_runtime_tool_names`) pin an EXHAUSTIVE
  `RUNTIME_TOOL_NAMES` set that has been updated once per prior
  runtime-tool addition (spawn, find_tools, load_tools, skill_file,
  install_skill, run_skill_script, search_run_log) — updating them for
  the two new members is the same established maintenance, not a
  workaround.
