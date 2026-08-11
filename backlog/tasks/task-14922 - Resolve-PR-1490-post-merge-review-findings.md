---
id: TASK-14922
title: Resolve PR 1490 post-merge review findings
status: In Progress
assignee: []
created_date: '2026-08-11 04:46'
labels:
  - mcp
  - review
dependencies:
  - TASK-2512
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/1490'
priority: high
---

## Description

Reconcile every substantive finding raised after PR #1490 merged so the new
standalone MCP boundary is not left with known correctness, packaging,
observability, or maintainability questions. Preserve the protocol and privacy
contracts that were deliberately established during TASK-2512 while correcting
findings that reproduce against the merged tree.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every substantive PR #1490 review finding is reproduced and either fixed with regression evidence or rejected with code- and protocol-backed rationale
- [ ] #2 Payload-free diagnostic and MCP 2025-03-26 compatibility contracts remain intact
- [ ] #3 Focused MCP regression, packaging metadata, formatting, lint, type, security, and diff checks pass for changed scope
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read PR #1490 review threads and verify each claim against current dev and ADR-053.
2. Capture failing tests for each valid behavioral finding before production edits.
3. Apply the smallest fixes while preserving payload-free diagnostics and 2025-03-26 protocol compatibility.
4. Add Google-style public runtime documentation and validate dependency/import ownership decisions.
5. Run focused and proportional regression/static/security gates, reply to each GitHub thread with evidence, and record implementation notes.

ADR required: no
ADR path: backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md
Reason: This is a post-merge correctness/documentation review of the runtime boundary already decided by ADR-053; no new architectural boundary is introduced.
<!-- SECTION:PLAN:END -->
