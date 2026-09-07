---
id: TASK-1250
title: Run-log recovery cannot return content past max_tool_result_chars
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 00:00'
updated_date: '2026-07-28 18:57'
labels:
  - agents
  - run-log
  - correctness
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When an agent's tool result is truncated in history, `_truncate_tool_result` now
appends a trailer naming the run-log record holding the full copy:

> The full result is recorded at record 000412 — `search_run_log(from_record=412, to_record=412)`.

Following that pointer cannot return anything the model has not already seen.
`run_log_search.format_results` renders `content[:max_chars]` from **offset 0**,
and the service closure sets `max_chars` to the run's
`budget.max_tool_result_chars` (16,000 by default) — the same ceiling that
truncated the result in history in the first place. So for any result larger than
that ceiling, the "recovered" render is byte-identical to the truncated view the
model already had.

Two consequences:

1. **The trailer overpromises.** It says the full result is recoverable. For the
   results most worth recovering — the large ones — it is not.
2. **A match can render a body without the match.** Because rendering always
   starts at offset 0, `contains="THE_ANSWER"` can legitimately *match* a record
   whose match sits at character 40,000, and then render characters 0–16,000,
   which do not contain it. The agent is told the record matches and shown text
   that contradicts that. This is the same silent-wrong-answer class as the
   `limit`/`context` and negative-`context` defects fixed earlier in this branch.

This is a scope boundary rather than a regression — the Phase 1 fix it came from
removed a much worse 400-character cap, and the design spec defers slicing and
aggregation tools to Phase 2. But the trailer's wording asserts a capability that
does not exist yet, and the match-without-showing behaviour is actively
misleading.

Discovered by the final whole-branch re-review of the run-log Phase 1 branch
(`feat/agent-run-log-spec`); see
`Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md` §6.1
and §11.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An agent can retrieve content from a logged record beyond `max_tool_result_chars` — e.g. via an offset/length parameter, a windowed render centred on the match, or a dedicated slice tool
- [x] #2 When a record matches a `contains=`/`pattern=` query, the rendered body contains the match, or states explicitly that the match lies outside the rendered window and how to reach it
- [x] #3 The truncation trailer's wording matches what following it can actually deliver
- [x] #4 Retrieval remains bounded so a single call cannot blow the context window
- [x] #5 Tests cover a result substantially larger than `max_tool_result_chars`, asserting that content near its END is retrievable and that a match beyond the ceiling is either shown or explicitly located
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add match-centred rendering to run_log_search.format_results: new keyword-only
   contains/pattern/offset params; a helper locates each record's first match
   (contains unbounded, pattern bounded to MAX_REGEX_SCAN_CHARS, matching
   search_records' own decision) and another picks the window start (explicit
   offset > 0 wins; else centre on the match; else 0), clamped into bounds.
2. Make elision visible: when the rendered window doesn't cover the whole
   record, append the shown character range, total size, and the offset to
   pass next.
3. Add "offset" to the search_run_log tool schema (tool_catalog.py) and thread
   it from args through agent_service.py's search_run_log closure into
   format_results, coerced the same defensively-numeric way as
   from_record/to_record/context.
4. Reword _truncate_tool_result's trailer (agent_runtime.py) so it no longer
   implies a bare from_record/to_record call recovers everything -- it now
   names contains=/offset= as how to actually reach content past the ceiling.
5. Add tests: the live-failure reproduction (2,925-char record, marker at
   2,646, max_chars=500), offset paging to a record's end, negative/past-end
   offset clamping, junk offset via the real closure, and a no-query control.
   Verify the reproduction test fails pre-fix (TypeError: unexpected keyword
   argument 'contains') and passes post-fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed by adding match-centred rendering, an explicit `offset` paging
parameter, and visible/actionable elision to run_log_search.format_results,
threaded end to end through the search_run_log tool.

Approach: format_results gained keyword-only contains/pattern/offset params
(existing max_chars=400 default untouched, existing tests unmodified). Per
record, a window start is chosen: explicit offset>0 always wins (deterministic
paging); otherwise a match position (contains searched unbounded, pattern
bounded to MAX_REGEX_SCAN_CHARS, mirroring search_records' own match
decision) centres the window; otherwise it starts at 0 as before. Centring
math is proven (see _window_start's docstring) to always keep the match
inside the window once found. When the window doesn't cover a record's whole
content, the block states the shown range, total size, and the `offset` to
continue with -- this is what lets an agent actually page past the render
ceiling across multiple calls, since the ceiling itself did not change (AC #4:
retrieval stays bounded per call).

`offset` was added to SEARCH_RUN_LOG_TOOL_SCHEMA and threaded through
agent_service.py's search_run_log closure with the same defensive
int(args.get(...) or 0) coercion already used for from_record/to_record/
context -- a non-numeric value returns an error ToolResult, never raises.

_truncate_tool_result's trailer (agent_runtime.py) was reworded: it no
longer implies a bare `search_run_log(from_record=X, to_record=X)` call
recovers everything (it renders at the same ceiling that did the original
cut) -- it now names `contains=`/`offset=` as how to actually reach content
beyond that cut. The record-number threading itself is unchanged.

Tests added (Tests/Agents/test_run_log_search.py,
test_search_run_log_runtime_tool.py):
- test_match_beyond_max_chars_is_rendered_not_silently_dropped: reproduces
  the live failure exactly (2,925-char record, marker at char 2,646,
  max_chars=500). Verified failing pre-fix (TypeError: format_results() got
  an unexpected keyword argument 'contains') and passing post-fix.
- test_offset_pages_through_a_large_record_to_its_end
- test_offset_negative_or_past_end_is_clamped_not_empty
- test_no_query_render_still_starts_at_offset_zero
- test_name_is_registered_as_a_runtime_tool: extended to assert "offset" in
  the schema
- test_real_closure_never_raises_on_a_junk_offset: full AgentService.run_turn
  flow proving a non-numeric offset never raises into the run

Full suite: Tests/Agents/ 447 passed, 0 failed (442 baseline + 5 new tests).

Modified: tldw_chatbook/Agents/run_log_search.py,
tldw_chatbook/Agents/agent_service.py, tldw_chatbook/Agents/agent_runtime.py,
tldw_chatbook/Agents/tool_catalog.py, Tests/Agents/test_run_log_search.py,
Tests/Agents/test_search_run_log_runtime_tool.py.
<!-- SECTION:NOTES:END -->
