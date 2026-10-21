---
id: TASK-35931
title: Allow mixed Console grounding without forced Library attribution
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 04:23'
updated_date: '2026-10-21 05:30'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-guidance.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console answers grounded in workspace files must coexist with staged Library evidence without acquiring unrelated Library citation labels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console automatic and retry evidence permits workspace grounding while dedicated Library answers remain strict.
- [x] #2 Canonical Console file-only answers require no Library marker repair; actual unknown or malformed markers remain invalid.
- [x] #3 Actual handoff staging and capture preserve exact source text, evidence identity and ownership.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow G2 in Docs/superpowers/plans/2026-10-20-workspace-project-guidance.md. Write RED regressions for Console mixed formatter, canonical capture optional markers and real consumer/handoff flow; preserve strict Library defaults and actual-marker validation; run focused tests/lint; review. ADR required: yes; ADR path: backlog/decisions/240-workspace-project-context-and-folder-bound-index.md; Reason: implements accepted mixed-source contract without persisted schema or policy changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented mixed Console evidence with shared rendering and construction-time optional markers; dedicated Library and generic captures remain strict. Actual automatic/retry payloads, direct/agent completion, exact snapshots/contract identity and mounted handoff dispatch are covered. Commit278687992c; independent spec/quality review approved. Final23 consumers and2 mounted cases passed;65 dedicated Library service cases passed. Required Chat248passed/2 inherited stop-observer failures reproduced on pre-G2 source; full capture/Library178passed with one inherited UI profile fixture corrected using the official harness and its mounted rerun passed. Zero introduced Ruff findings; diff check clean. ADR: backlog/decisions/240-workspace-project-context-and-folder-bound-index.md. Saved Library policy, scope and file authority unchanged; no indexing. Qualification receipts will be retained in Docs/superpowers/qa/2026-10-20-workspace-project-context/guidance/.
<!-- SECTION:NOTES:END -->
