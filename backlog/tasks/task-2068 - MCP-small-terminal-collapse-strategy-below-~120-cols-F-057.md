---
id: TASK-2068
title: 'MCP: small-terminal collapse strategy below ~120 cols (F-057)'
status: In Progress
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 20:46'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100x30 the servers table loses Tools/Auth columns with no scroll affordance, summary clips, rail rows truncate. No collapse strategy. Evidence: mcp-100x30.png. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All primary content and actions remain reachable at 100x30,No mid-word clipping at 100x30,Rendered-layout test at 100x30
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (TCSS layout/copy behavior; no widget contract change). Steps: 1. Reproduce at 100x30 in a rendered test: identify what clips (servers table columns, summary line, rail rows). 2. RED rendered-layout test at 100x30 asserting primary content/actions reachable + no mid-word clipping. 3. Lightest TCSS fix consistent with existing patterns (candidate: narrow rail min-widths at small widths, wrap the summary, hide lowest-priority table columns) via existing responsive idioms in the bundle. 4. Run MCP layout/parity/workbench tests + ruff.
<!-- SECTION:PLAN:END -->
