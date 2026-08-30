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

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`

Reason: This closes the approved programme through deterministic UAT,
documentation, and disposable-profile evidence while preserving the existing
storage, scheduler, permission, Console, MCP, First Run, and Library boundaries.
ADR-032 already governs the Console-local Watchlists, briefing privacy, durable
receipt, and external-MCP boundary, so no duplicate ADR is required.

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

## Review-fix round 1 (2026-08-29)

- Replaced the no-preset import-time provider fallback with one call-time
  persisted provider/model resolver shared by manual generation, scheduled
  generation, schedule receipts, and durable `model_used` provenance. The
  collection/preset remains first; active conversation state is never consulted;
  unavailable persisted defaults fail closed.
- Added a disposable-profile mounted-app UAT using the real
  `ConsoleChatController`, app-owned provider composition, visible approval flow,
  durable receipts, and public Watchlists/Settings/Library navigation with local
  scripted fixtures and no public network.
- Fixed identical resume-state approval synchronization so it preserves mounted
  row/control identity while changed calls, phase, or round still render.
- Corrected the three independently reported Library test failures and recorded
  a 204-pass changed-file gate.
- Added a committed fail-closed redaction checker, truthfully separated service,
  mounted, and seeded rendering evidence, and added actual mounted Console
  captures at 180x50 and 160x42.
- Review-fix verification is green for all non-network targeted gates. The QA
  file is 3-pass plus one sandbox-only loopback-bind failure; the exact
  local-bind rerun and reconciliation onto observed `origin/dev`
  `c2939400be1138ed92fb1a92e81b908548c31642` remain for root after this isolated
  commit. Task status and acceptance criteria intentionally remain unchanged
  until independent re-review.

## Latest-dev reconciliation (2026-08-29)

- Rebased the complete branch onto `origin/dev`
  `91e5340e347e7db21c3f4f19ba3d14fb4da61f85`; the merge base now equals that
  exact revision.
- Preserved latest dev's pinned workspace executor in external-MCP provider
  composition and updated the scripted QA model to wait for durable readiness
  before querying each distinct receipt exactly once, without weakening the
  production loop detector.
- Reconciled integration fixtures with the canonical user-denial result and the
  complete read-only Watchlists service shape.
- Fresh targeted evidence includes the mounted UAT, the loopback service UAT,
  First Run, external MCP, provider approvals, durable scheduling, Library,
  documentation, and CSS contracts. Task status and acceptance criteria remain
  unchanged until independent re-review.

## Review-fix round 2 (2026-08-29)

- Kept a confirmed cadence write authoritative when the optional persisted
  provider/model projection is unavailable: the command now returns an honest
  `ok` receipt, requests scheduler reload, marks `briefing_route_ready: false`,
  and provides fixed Settings recovery instead of claiming storage failed.
- Added anti-vacuous coverage for one write, one reload request, acknowledged
  reload, null unresolved route fields, scrubbed exception details, and the
  unchanged successful-route receipt.
- Rebased again onto `origin/dev`
  `667f8168e15940fb80b1d8812891ce0f48f4fd53`; focused command/briefing/scheduler,
  no-preset, mounted UAT, upstream notification, Ruff, compile, and diff gates
  are green. Status and acceptance criteria remain unchanged for round-3 review.
