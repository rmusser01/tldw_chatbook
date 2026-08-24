---
id: TASK-21502
title: Clean generated root artifacts and relocate task reports
status: In Progress
assignee: []
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 05:55'
labels:
  - docs
  - hygiene
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove clearly generated root-level artifacts, prevent them from returning, and relocate useful reports, plans, PRDs, and curated QA evidence into their maintained documentation areas so the repository root contains only active project entry points and intentional top-level documents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generated server logs, the empty suite-result file, and the generated verification screenshot are absent from the tracked root.
- [x] #2 Root ignore rules prevent generated verification captures and the output directory from being recommitted.
- [x] #3 Historical task reports live under `backlog/docs/task-reports/` and all repository references resolve to their new paths.
- [x] #4 Root plan and PRD files live under appropriate `Docs/Development/` locations.
- [x] #5 Curated TASK-1989 QA screenshots live beside their canonical QA record under `Docs/superpowers/qa/`, with documentation references updated.
- [x] #6 README-focused verification and repository hygiene checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Classify tracked root entries conservatively and preserve active scripts, packaging metadata, maintained top-level project documents, and canonical QA records.
2. Delete the clearly generated logs, empty suite result, and root failure screenshot; ignore future generated output and redirect the verification script's failure capture into that ignored directory.
3. Move historical task reports into `backlog/docs/task-reports/`, move the three root plan/PRD documents into `Docs/Development/`, and update references.
4. Move curated TASK-1989 screenshots from the root `output/` tree beside their canonical QA record and update the live-UAT and historical plan paths.
5. Verify root inventory, tracked references, README-focused checks, the verification script syntax, and diff hygiene; then document the ADR outcome and close the task.

ADR required: no
ADR path: N/A
Reason: this is repository/documentation hygiene that preserves content and existing ownership; it does not change architecture, runtime policy, storage contracts, security boundaries, dependencies, or long-lived UX structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed four clearly generated root artifacts and added root-scoped ignore rules for future local verification output. Redirected `verify_ui.py` screenshots into ignored `output/`, relocated four task reports to `backlog/docs/task-reports/`, moved three historical plan/PRD documents under `Docs/Development/`, and moved all 50 curated TASK-1989 screenshots beside their canonical QA record while updating references. The verification script compiles, all QA ledger screenshot basenames resolve, `git diff --check` passes, and the focused README suite passes 5 tests. No ADR was required because the changes preserve existing content and ownership. The repository-wide suite remains non-green only for the four unchanged Agent failures already documented for this branch, so the task remains In Progress under the repository's strict Definition of Done.
<!-- SECTION:NOTES:END -->
