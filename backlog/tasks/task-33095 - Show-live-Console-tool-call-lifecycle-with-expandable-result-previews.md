---
id: TASK-33095
title: Show live Console tool-call lifecycle with expandable result previews
status: Done
assignee:
  - '@codex'
created_date: '2026-09-27 19:52'
updated_date: '2026-09-27 21:09'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/qa/2026-09-27-console-tool-lifecycle.md
  - backlog/decisions/195-console-live-tool-call-presentation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make tool work observable while a Console reply is still running, with compact results and inspectable details.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every complete primary tool call appears in its Assistant turn before execution and updates in place through queued, actual approval wait, running, and terminal outcomes.
- [x] #2 Each result appears immediately with at most three wrapped preview lines and one disclosure for arguments and result, preserving existing full-output and diff access.
- [x] #3 Repeated same-name calls, mixed approvals, cancellation, session switching, and updates preserve correct call ownership, focus, expansion, and scroll.
- [x] #4 Existing permissions, provider history, durable capture policy, and raw-shell lifecycle remain intact; partial output streaming is out of scope.
- [x] #5 Targeted runtime, bridge, mounted UI, token/CSS checks and a disposable live Console verification pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reuse runtime call identities and lifecycle observations through an optional observational display callback; document the session-only boundary in ADR-195.
2. Add failing bridge and mounted transcript regressions for early rows, real approval waits, same-name calls, terminal results, and stable disclosure previews.
3. Project calls into stable display-only TOOL markers, settle interrupted rows, and preserve existing raw-shell and durable capture owners.
4. Render compact wrapped previews and a single arguments/result disclosure using existing Console components and design tokens.
5. Run targeted regression, lint, CSS, and disposable live checks; update the guide and implementation notes.
ADR required: yes
ADR path: backlog/decisions/195-console-live-tool-call-presentation.md
Reason: Record the optional runtime-to-Console display observer and session-only lifecycle projection; existing ADR-078, ADR-080, and ADR-029 remain authoritative for outcomes, capture, and privacy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented stable live Console tool rows through an optional runtime/service observer and a session-scoped call projection. Calls appear before permission review, distinguish actual approval waits from running, settle individually, and retain arguments/result details with a three-row wrapped preview. Completion, diff attachment and model-shell takeover preserve disclosure identity; repeated provider IDs get distinct rows. Existing raw-shell execution authority, permission decisions, provider history and durable capture eligibility remain intact. Observer failures cannot break approvals or run teardown.

Updated the Console guide, the design token catalog and generated CSS. ADR required: yes; decision: backlog/decisions/195-console-live-tool-call-presentation.md. No schema change or dependency added. Ordinary partial-output streaming remains v2.

Verification: main targeted run 630 passed with two independently reproduced baseline failures excluded; subsequent boundary run 30 passed; turn grouping 24 passed; final preview/token/CSS checks 19 passed; teardown-failure and scroll-retention regressions passed individually. Counts overlap. Changed Python lines have no Ruff findings; the new module/test pass lint and formatting, and scoped git diff --check passes. Broader native-transcript checks exposed 32 more existing failures; all 34 reproduce against unmodified HEAD copies. Existing whole-file lint debt remains. The full repository suite was not run.

A disposable real-Console tmux walkthrough with isolated pytest config/data and scripted tool results verified mixed approval, running timing, terminal previews, mouse expansion, Enter collapse and 100-column wrapping; the final assistant response arrived and the run returned done. No external provider was exercised. Evidence and probe limitations: Docs/superpowers/qa/2026-09-27-console-tool-lifecycle.md. Review findings were addressed with regressions; unrelated worktree changes were preserved.

PR integration: transplanted the feature onto current dev a7b6d5864bde6072b01eeb7a54a89e9bc977e1cb in a managed worktree, preserving the original dirty checkout. Renumbered the occupied ADR to backlog/decisions/195-console-live-tool-call-presentation.md. Used current Console panel styles and sizing tokens; retained dev service arguments and tests. Changed config-reading tests use the established bootstrap_profile marker. Final feature run: 185 passed. Production mouse/Enter behavior, CSS reproduction, task-ID guard and changed-line lint are covered in the QA document. Broader existing dev failures were independently reproduced or traced to byte-identical base files; see the QA document for exact limitations. A fresh native probe returned done and exited cleanly after approval, timing, expansion/collapse and narrow preview checks, with unrelated sidebar DB reads isolated. Integration review found no new actionable issue.
<!-- SECTION:NOTES:END -->
