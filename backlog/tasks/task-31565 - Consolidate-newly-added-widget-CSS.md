---
id: TASK-31565
title: Consolidate newly added widget CSS
status: Done
assignee: []
created_date: '2026-09-05 02:39'
updated_date: '2026-09-05 02:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the repository stylesheet registration contract for seven widgets that currently add class-level Textual stylesheet sources and fail the consolidation guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All eight reported widgets expose bundle-only CSS attributes.
- [x] #2 The generated widget-default stylesheets contain the consolidated rules.
- [x] #3 The class-level CSS guard and the complete 71-case vLLM geometry matrix pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rename each plain literal DEFAULT_CSS to the repository bundle attribute appropriate for widgets. 2. Regenerate consolidated CSS artifacts with the canonical builder. 3. Run the consolidation contract and focused affected geometry tests. ADR required: no. ADR path: N/A. Reason: This enforces the existing stylesheet build contract without changing UI ownership or runtime architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Converted the seven failures harvested from the pre-rebase CI run plus the newly merged `VllmSetupView` declaration from `DEFAULT_CSS` to the repository's `BUNDLED_CSS` contract. Regenerated the two widget-default stylesheet artifacts with `build_css.py`. The consolidation guard and all 71 vLLM geometry cases pass (72/72 combined); scoped Ruff and `git diff --check` pass. Modified eight widget/view modules and regenerated `tldw_chatbook/css/widget_defaults_scoped.tcss` plus `tldw_chatbook/css/widget_defaults_self.tcss`. ADR required: no.
<!-- SECTION:NOTES:END -->
