---
id: TASK-32869
title: Specify Library Artifacts integration
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 21:50'
updated_date: '2026-09-19 22:07'
labels:
  - library
  - artifacts
  - design
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the approved Library-style Artifacts proposal and review into an explicit design, architectural decision, and staged implementation plan. Reports default to all with a Kept filter; the existing ZIP-pack manager stays linked from Library initially.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design records all reports by default with an explicit Kept filter and preserves manual keeping semantics.
- [x] #2 The design preserves kept reports after source deletion and documents catalog, scope, sharing, keyboard, and navigation behavior.
- [x] #3 A canonical ADR and executable implementation plan identify boundaries, alternatives, targeted verification, and atomic delivery stages.
- [x] #4 Documentation references and task metadata are verified without changing application code.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the approved Library Artifacts layout, all-reports default with Kept filter, and linked existing Chatbook manager in a design specification.
2. Create ADR-172 for navigation and cross-owner browsing while retaining source ownership, retention, sharing, and permission boundaries.
3. Write an executable staged implementation plan with concrete file boundaries, capability inventory, source contracts, regression fixtures, and targeted checks.
4. Verify document links, task metadata, and diff scope; record review conclusions and close the design task.
ADR required: yes
ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md
Reason: Library navigation and artifact read contracts span long-lived UX and existing service owners.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recorded the approved Library Artifacts direction in Docs/superpowers/specs/2026-09-19-library-artifacts-design.md and staged execution in Docs/superpowers/plans/2026-09-19-library-artifacts.md. Reports default to All reports with a Kept filter; the existing ZIP-pack manager remains linked. The review now covers source deletion, copy provenance, capability-based actions, sharing lifetime, bounded source reads, keyboard/focus restoration, and compatibility routes.
Created ADR-172 at backlog/decisions/172-library-artifacts-browse-and-navigation.md and indexed it. Imported source IDs are device-local, so the design keeps live and saved copies distinct instead of unsafe numeric-ID deduplication. No storage migration or new permission authority is proposed.
Verification: all 18 document links resolve; all 6 Python examples parse; plan placeholder scan, Backlog metadata/AC/dependency checks, and git diff --check pass. A fresh scan found no task or ADR collisions across all refs and 42 worktrees. Application code is unchanged; runtime tests and native TUI verification belong to the open implementation stages and were not claimed or run for this documentation-only task.
<!-- SECTION:NOTES:END -->
