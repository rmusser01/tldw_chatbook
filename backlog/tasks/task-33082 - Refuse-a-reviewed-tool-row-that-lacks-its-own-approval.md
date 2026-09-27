---
id: TASK-33082
title: Refuse a reviewed tool row that lacks its own approval
status: To Do
created_date: 2026-09-27 16:40
dependencies:
- TASK-32956
labels:
- agents
- permissions
- security
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-32956, a probe found a latent fail-open in the Console's tool-batch review path. When the approval card returns a map in which one row is approved and a sibling row is marked `timeout` (or any other value that is neither an approval nor `deny`), the sibling runs as if approved. It runs on the approved row's name-keyed stamp, because the review hook turns every non-deny row into a `proceed` verdict, and the runtime also defaults missing rows to `proceed` (`agent_runtime.py`, `verdicts.get(call.name, "proceed")`).

The shipped approval bridge never produces such a map today: it answers every row, and it fills a missing answer with deny, or with timeout for the whole batch on a deadline. So this cannot currently be reached. But the safety of every mutating tool, `character_save` included, rests on that invariant in a different module rather than on the hook itself. The hook should fail closed on its own.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reviewed tool call runs only when its own row carries an approval decision. A row that is missing, or marked timeout or any unknown value, is refused, whatever its siblings were answered.
- [ ] #2 A test drives a mixed approval map (one row approved, a sibling timed out) through the real review path and shows the sibling does not run.
- [ ] #3 Existing approve/deny/timeout behaviour for fully answered batches is unchanged. The failure-name set matches dev's.
<!-- AC:END -->
