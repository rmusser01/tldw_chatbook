---
id: TASK-32342
title: >-
  Tool-call continuation crashes on mappingproxy JSON and is reported as a
  provider HTTP 400
status: To Do
assignee: []
created_date: '2026-09-10 19:09'
labels:
  - agents
  - llm-calls
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a native tool call, the next provider request carries a MappingProxyType (ToolCall.arguments became immutable in 4ed4757d53, 2026-09-04) and the requests-based OpenAI-compatible handler cannot serialise it. The failure is shown to the user as 'provider returned HTTP 400 ... The provider rejected this request. Confirm the model is still available', blaming the provider for a client-side bug. Reproduced live with the custom provider; other handlers unverified. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A multi-step tool-call turn (find_tools, load_tools, tool) completes through the custom/OpenAI-compatible handler without a serialisation error.
- [ ] #2 Every provider handler that re-sends tool-call history serialises immutable argument mappings; a regression test covers the serialisation seam.
- [ ] #3 A client-side serialisation failure is reported to the user as an app error, never as a provider rejection or HTTP status.
<!-- AC:END -->

## Renumbering provenance

<!-- SECTION:PROVENANCE:BEGIN -->
This task was filed as TASK-32273 on 2026-09-10 at 12:19 PT (19:19 UTC) in the
approval-card / MCP-permissions UX review wave (PR #2574), which makes it the
OLDER arrival against the TASK-32273 that `dev` minted later the same day
("Integrate template-aware local reasoning history with canonical Console
thinking", landed via PR #2575).

It renumbered to TASK-32342 anyway. The bare 2026-08-21 owner rule of TASK-19601
quoted in `scripts/check_backlog_task_ids.py` (older arrival keeps the id) is
superseded by the 2026-09-08 refinement recorded in
`backlog/docs/lessons-backlog-hygiene.md`: **landed-keeps-id trumps
older-keeps-id -- a task already on origin/dev never renumbers; the unmerged
side moves regardless of timestamps, because renumbering landed ids breaks
external references.** The dev-side TASK-32273 is merged and cited from
`backlog/decisions/090-console-thinking-block-ownership-and-replay.md`,
`backlog/docs/lessons-live-verification.md` and two plan/QA records; this task
was cited only from its own unmerged plan. `dev` applied the same refinement
earlier on 2026-09-10 when it renumbered its archive-lifecycle task from
TASK-32273 to TASK-32300 for exactly this collision.

Renumbered 2026-09-10. Commit messages on the wave branches
`approval-wave-a/b/c` written before this date that cite `task-32273` refer to
THIS task; the dev-side TASK-32273 keeps the id.
<!-- SECTION:PROVENANCE:END -->
