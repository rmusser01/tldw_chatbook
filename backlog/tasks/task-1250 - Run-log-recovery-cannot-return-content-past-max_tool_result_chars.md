---
id: TASK-1250
title: Run-log recovery cannot return content past max_tool_result_chars
status: To Do
assignee: []
created_date: '2026-07-28 00:00'
labels: [agents, run-log, correctness]
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
- [ ] #1 An agent can retrieve content from a logged record beyond `max_tool_result_chars` — e.g. via an offset/length parameter, a windowed render centred on the match, or a dedicated slice tool
- [ ] #2 When a record matches a `contains=`/`pattern=` query, the rendered body contains the match, or states explicitly that the match lies outside the rendered window and how to reach it
- [ ] #3 The truncation trailer's wording matches what following it can actually deliver
- [ ] #4 Retrieval remains bounded so a single call cannot blow the context window
- [ ] #5 Tests cover a result substantially larger than `max_tool_result_chars`, asserting that content near its END is retrievable and that a match beyond the ceiling is either shown or explicitly located
<!-- AC:END -->
