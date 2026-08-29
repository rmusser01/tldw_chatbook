---
id: TASK-22868
title: Close out the Console-driven Watchlists workflow UAT
status: In Progress
assignee:
  - codex
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:17'
labels:
  - watchlists
  - console
  - uat
  - docs
dependencies:
  - TASK-613
  - TASK-22859
  - TASK-22860
  - TASK-22861
  - TASK-22862
  - TASK-22863
  - TASK-22864
  - TASK-22865
  - TASK-22866
  - TASK-22867
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-console-watchlists-workflow-uat-closeout.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the complete latest-dev Console workflow from source registration through briefing consumption, confirm the already-landed First Run fixes remain intact, and publish user-facing documentation and reproducible UAT evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a disposable fresh profile from then-current `origin/dev`, a Console-driven user can register several feeds, create a Watchlist, check sources, generate a briefing, schedule every 24 hours, follow exact receipts, and have the agent read the completed briefing with provenance.
- [ ] #2 The same sources, memberships, runs, briefing, and schedule state are independently inspectable in Watchlists and Settings without product hunt/ATHF integration.
- [ ] #3 External MCP contract UAT proves approved metadata/receipts are consumable while Console-only commands, article snippets/bodies, and full briefing content are absent.
- [ ] #4 A generic non-skill framework repository is classified accurately, and installable skill input still reaches trust review; TASK-613's superseded-submit scenario is included.
- [ ] #5 First Run regression checks confirm latest-dev's already-landed detected/selected/configured, atomic readback, focus, and blocked-Console intent behavior without duplicating those fixes.
- [ ] #6 User guides document Console prompts, approvals/effects, receipt polling, bulk authoring, every-24-hours/app-open semantics, agent briefing consumption, and skill/framework classification without implying briefing-to-hunt handoff.
- [ ] #7 Reproducible targeted automated and production-shaped live evidence is recorded; a full repository sweep is run only with explicit user opt-in.
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: This closes the approved programme through deterministic UAT, documentation,
and disposable-profile evidence while preserving the existing storage, scheduler,
permission, Console, MCP, First Run, and Library boundaries.

1. Build one deterministic public-seam QA harness for the Console source → Watchlist
   → check → briefing → every-24-hours schedule → agent-read workflow, using only
   temporary SQLite/profile state and scripted model/briefing content.
2. Prove exact receipt following, cross-surface Watchlists/Settings corroboration,
   and the external-MCP metadata/receipt-only privacy boundary.
3. Retain First Run and generic skill/framework behavior as targeted regression
   prerequisites without reopening or duplicating those implementations.
4. Update the user guides and publish redacted, reproducible automated and
   production-shaped UAT evidence at the supported terminal sizes.
5. Run only the plan's targeted gates, obtain an independent review, and close the
   task after every acceptance criterion is evidenced.
