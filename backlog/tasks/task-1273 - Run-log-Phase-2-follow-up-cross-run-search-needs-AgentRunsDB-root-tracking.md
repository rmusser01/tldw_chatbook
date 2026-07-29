---
id: TASK-1273
title: 'Run log Phase 2 follow-up: cross-run search needs AgentRunsDB root tracking'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-29 00:20'
updated_date: '2026-07-29 17:31'
labels:
  - agents
  - run-log
dependencies:
  - TASK-1271
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-1271 (run-log Phase 2: aggregation, slicing, cross-run search) investigated cross-run search and deferred it here rather than build it against a guess. The log is per-run: a run's directory is named by its bare run_id under <root>/.agent-runs/<run_id>/, with no per-conversation grouping on disk, and each run's MANIFEST records only run-level metadata (run_id, model, api_endpoint, allowed_tools, budget, status, superseded_run_id, total_tokens) -- neither conversation_id nor the resolved root the log was written under. AgentRunsDB.agent_runs does map conversation_id -> run_id (indexed), which cross-run search needs, but it has no column recording which root (a bound workspace folder, or the sandbox fallback) resolve_log_root() chose for that run -- that choice is resolved fresh on every run and never persisted. So a historical run's log directory cannot be located from its run_id alone without either assuming the current workspace binding matches every historical run (silently wrong whenever it does not: workspace folder reconfigured, different session, different sandbox root) or adding a schema column. The design spec's own §2.2 deliberately kept AgentRunsDB out of Phase 1's retrieval path, and this is exactly the kind of DB-schema change task-1271 was told not to improvise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cross-run search remains primary-agent only, mirroring search_run_log/run_log_stats/run_log_slice's own isolation gate
- [x] #2 search_run_log gains a scope argument (default "run", byte-identical to prior behaviour; "conversation" also searches the conversation's earlier runs) that composes AgentRunsDB.list_runs with run_log.resolve_existing_log_dir, per the task's Revised analysis option (a) -- no agent_runs schema column or migration added
- [x] #3 A run whose log cannot be located under the current root (workspace folder bound/rebound/unbound since, or predates run-log) degrades gracefully -- reported as unresolved/not-attempted with an explicit count, never raising and never silently indistinguishable from "no matches"
- [x] #4 Cross-run search output stays bounded regardless of how many runs are searched -- one shared hit limit and one shared wall-clock deadline across the whole call, mirroring MAX_STATS_GROUPS/MAX_SLICE_RECORDS' own bounded-output guarantee
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read run_log_search.py, run_log.py (resolve_existing_log_dir), agent_service.py's search_run_log closure + three-part gate, and AgentRunsDB.list_runs.
2. Implement option (a), best-effort, no schema change: add MAX_CROSS_RUN_RUNS, CrossRunHit/CrossRunSearchResult, search_across_runs(), format_cross_run_results() to run_log_search.py -- pure functions taking already-resolved (run_id, log_dir|None) pairs, sharing ONE hit `limit` and ONE wall-clock `deadline_seconds` across every run searched (not reset per run).
3. Extend search_run_log's args with `scope` ("run" default / "conversation") in agent_service.py: scope="conversation" enumerates the conversation's PRIMARY runs via self.db.list_runs (capped at MAX_CROSS_RUN_RUNS), resolves each via run_log.resolve_existing_log_dir (current run's own log_dir reused directly, never re-resolved), and renders via format_cross_run_results. The scope="run" code path is left byte-for-byte untouched below an early branch.
4. Add `scope` to SEARCH_RUN_LOG_TOOL_SCHEMA in tool_catalog.py.
5. New test file Tests/Agents/test_run_log_cross_run_search.py: default-scope byte-identical proof, older-run hit attribution, unresolved-run reporting, mixed resolvable/unresolvable, zero-prior-runs, junk scope values, sub-agent gating (incl. a temporary gate-removal mutation to confirm the gating test actually fails), plus direct unit tests of search_across_runs/format_cross_run_results for the shared-limit and shared-deadline bounds.
6. Update task-1273's ACs to match the actually-agreed scope (best-effort, no schema column -- see the task's own "Revised analysis" section), run both Tests/Agents and Tests/Chat, mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented option (a) from the Revised analysis: best-effort cross-run search, no
AgentRunsDB schema change. `search_run_log` gained a `scope` arg ("run" default /
"conversation") rather than a new tool -- the model already knows this tool and the
per-run gate/schema-disclosure logic is reused unchanged.

run_log_search.py (pure, no DB/root resolution of its own -- mirrors load_records'
explicit-Path contract): MAX_CROSS_RUN_RUNS=10, CrossRunHit/CrossRunSearchResult,
search_across_runs() and format_cross_run_results(). Both the hit `limit` and the
wall-clock `deadline_seconds` are SHARED across every run scanned in one call, not
reset per run -- resetting per run would let a scope="conversation" call cost
N_runs * MAX_SEARCH_SECONDS in the worst case, defeating the single-run "cheap,
in-process" guarantee (F6). CrossRunSearchResult carries three distinct buckets
that must never be conflated: searched (log found and scanned, even if 0 of its
hits made the cut), unresolved (log not locatable under the current root -- the
one honest limitation), and not_searched (log may well exist; there was no room
in this call's run-count cap or its shared deadline to check it).

agent_service.py: search_run_log's closure branches on `scope` immediately after
computing log_dir/contains/pattern; the scope=="run" code below the branch is the
literal untouched original block (proven byte-identical by a new test comparing
"no scope" vs. explicit scope="run" output, plus the full pre-existing test suite
passing unmodified). scope=="conversation" lists the conversation's PRIMARY runs
via self.db.list_runs (capped at MAX_CROSS_RUN_RUNS), resolves each via
run_log.resolve_existing_log_dir (the current run's own log_dir is reused
directly, never re-resolved), and never raises -- a DB/resolution failure
degrades to a ToolResult like every other failure mode in this closure.
Primary-agent isolation needed no new gate: scope is just an argument on a tool
still wired under the existing agent_kind==AGENT_KIND_PRIMARY LoopDeps gate.
Verified behaviourally, not just by inspection: temporarily changed that gate to
`search_run_log if True else None`, confirmed both the new
test_subagent_cannot_call_search_run_log_with_conversation_scope AND the
pre-existing test_subagent_cannot_call_search_run_log failed, then restored the
gate exactly (git diff empty on that hunk afterward).

tool_catalog.py: SEARCH_RUN_LOG_TOOL_SCHEMA gained a `scope` property describing
both values and the coverage-reporting contract.

Tests: Tests/Agents/test_run_log_cross_run_search.py (19 new tests) -- default-
scope byte-identical proof, an older run's hit found and attributed by run id,
an unresolvable older run counted and reported (not silently skipped), a mixed
resolvable/unresolvable scenario, zero-prior-runs graceful degradation, junk
scope values (7 parametrized cases) never raising, sub-agent gating (schema-
disclosure half + dispatch-refusal half, scope="conversation" explicitly
requested), and direct unit tests of search_across_runs/format_cross_run_results
proving the shared limit and shared deadline bounds.

Full suite: Tests/Agents/ 767 passed (748 baseline + 19 new), zero regressions.
Tests/Chat/ unaffected (not touched): 4 failed / 13 errors, matching this
programme's documented pre-existing baseline exactly.

Files changed: tldw_chatbook/Agents/run_log_search.py,
tldw_chatbook/Agents/agent_service.py, tldw_chatbook/Agents/tool_catalog.py,
Tests/Agents/test_run_log_cross_run_search.py (new).

--- PR #1088 review fixes ---

Finding A (Performance, agent_service.py ~:934): the conversation-scope
path called AgentRunsDB.list_runs(conversation_id) unbounded, materialising
every run (primary + every sub-agent) a conversation has ever had before
capping to MAX_CROSS_RUN_RUNS client-side. Fixed by pushing the cap INTO
the query: list_runs gained an optional agent_kind filter (SQL-level, not
client-side), called with limit=MAX_CROSS_RUN_RUNS, agent_kind="primary".
A new count_runs() method (single COUNT(*) query, no rows materialized)
gets the exact total so the coverage line still reports a precise omitted
count -- CrossRunSearchResult's not_searched_run_ids (exact ids, from the
deadline bucket) and the new omitted_run_count (a count only, from the cap)
are folded into the same "not attempted" note by format_cross_run_results.

Finding B (Reliability, run_log_search.py ~:963): the shared deadline was
checked before load_records(), but loading itself is unbounded I/O not
counted against the budget. Fixed BOTH ways the finding offered: (1)
load_records() itself is now deadline-aware -- checked between segment
files (never mid-segment-read), raises RunLogSearchTimeout if exceeded
before every segment is read, `None` (default) preserves prior behaviour
exactly; (2) search_across_runs recomputes the remaining deadline again
immediately after each load_records call, before search_records. Either
exhaustion routes that run to not_searched -- never a partial scan
silently presented as complete, never a hard failure of the whole call.

Tests added: Tests/Agents/test_run_log_cross_run_search.py (+4:
more-runs-than-cap reports exact excess; load_records raises on an
exhausted deadline; a load that alone exhausts the budget lands in
not_searched via search_across_runs, not scanned or dropped; format_cross_
run_results folds omitted_run_count correctly) and Tests/DB/test_agent_
runs_db.py (+6: list_runs agent_kind filtering and composition with limit,
count_runs correctness including superseded exclusion, and a trace-
callback proof that count_runs is a single COUNT(*) query, never len(list_
runs(...)) in disguise).

Full suite: Tests/Agents/ 771 passed (767 + 4), Tests/DB/ 609 passed + 1
skipped (no regressions), zero pre-existing tests edited. Tests/Chat/
unaffected: 4 failed / 13 errors, matching the documented baseline exactly.
<!-- SECTION:NOTES:END -->
