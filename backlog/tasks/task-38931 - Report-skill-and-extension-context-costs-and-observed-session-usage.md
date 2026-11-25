---
id: TASK-38931
title: Report skill and extension context costs and observed session usage
status: In Progress
created_date: 2026-11-05 16:08
updated_date: 2026-11-23 21:10
labels:
- console
- skills
- diagnostics
priority: high
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
documentation:
- Docs/superpowers/specs/2026-11-23-console-skill-cost-report-design.md
- Docs/superpowers/plans/2026-11-23-console-skill-cost-report.md
- backlog/decisions/230-console-send-preparation-and-io-ownership.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users understand which advertised skills and tool definitions consume the current conversation's input budget and which skills they have actually used during the observed session. Make setup costs explainable before users decide what to disable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A read-only diagnostic report separates advertised skill/catalog costs from full skill bodies loaded during the observed run and other tool-schema costs, without counting the same contribution twice.
- [x] #2 Costs use the prepared request's accounting where attribution exists; estimates, unavailable attribution, and the snapshot's age are explicitly labeled, with totals reconciled to the existing context breakdown.
- [x] #3 Each observable skill's invocation count and last observed use have an explicit session/observation window; missing history is reported as unknown rather than never used.
- [x] #4 Duplicate or conflicting invocation names and unavailable or disabled components have actionable explanations linked to their existing management surface.
- [x] #5 The report never automatically disables components, triggers model calls, expands path/provider authority, or exposes hidden instruction bodies outside the authorized Next Send preview.
- [x] #6 Targeted checks cover advertised-but-unused skills, invoked skills, ambiguous cost attribution, catalog changes, and workspace/session switching without stale results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/230-console-send-preparation-and-io-ownership.md
Reason: extend the existing Console preparation decision to define session-local diagnostic metadata ownership.

1. Test session fencing, invocation observation, and reconciliation against a prepared request.
2. Attach content-free observations to existing Console request and skill paths, and open a read-only report from the Console command.
3. Verify the mounted command and targeted regressions. Document the behavior and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [x] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [x] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Console now opens a read-only /skill-cost report for the current live chat. It shares the prepared request accounting with Context, including native and fenced tool estimates, and records accepted skill uses independently of trace Capture. It keeps rendered child text estimates separate from the parent request and labels manual body attribution and earlier history as unavailable. The report uses existing Library, Plugins, and MCP management links and keeps only bounded metadata for the active session. ADR-230 governs ownership and visibility. Targeted Chat, agent, Console UI and design-token tests cover the implementation. The task remains In Progress while the pull request is under review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
