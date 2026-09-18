---
id: TASK-32666
title: Preserve editing context across import option changes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 02:37'
updated_date: '2026-09-16 03:35'
labels:
  - library
  - design-system
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the per-type import controls review: users must be able to change options and reset one group without losing draft editing context or keyboard position. Qualify compact control readability and dependent option state without submitting an import.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Changing a checkbox or selector retains unrelated draft text, cursor/selection and visible keyboard context at compact and wide sizes in both themes.
- [x] #2 Dependent option controls, receipts, validation and Start consent agree with current values; resetting one group preserves other groups and metadata.
- [x] #3 Per-type option controls and disabled explanations remain readable and reachable at 80x24; any discovered layout defect is reproduced and repaired within the existing token language.
- [x] #4 Targeted automated checks, isolated native journeys, static and budget comparisons, focused review and documentation record the result without imports, installs or remote requests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Bounded correction of existing import option updates and responsive controls, preserving capability, submission and persistence contracts.

1. Reproduce checkbox/select/reset editing-context loss using production app CSS, real local preflight and temporary SQLite. Audit all seven existing type groups for clipped control labels at 80×24, including disabled states.
2. Add focused failing journeys for retained metadata and option editor selection, visible keyboard focus, dependency transitions, reset scope and readable controls in both themes and sizes.
3. Repair confirmed defects at their owning update/layout boundary using existing retained-widget and token patterns. Keep the current capability schema, explicit install actions and import submission authority.
4. Run targeted option/caller/consent checks, token/bundle governance if CSS changes, changed-file static/format and relevant size budgets; request focused independent review.
5. Run real TldwCli in an exclusive private profile at 170×48 dark and 80×24 light. Batch native captures, verify exact draft/fixture state, zero jobs and normal exit. Update guide, audit, QA and task; commit locally. No full suite, install, model execution, import submission, remote request, push or dev integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Import option changes now update retained controls, preserving unrelated draft text, selection and keyboard focus. Reset applies only its group; dependencies, receipts, validation and Start/Retry confirmation copy stay current. Queued native/forwarded edits verify current sender/value, ordinary refreshes preserve newer sibling typing, and pending backend layouts consume the current snapshot safely.

Long checkbox explanations wrap, and a scoped token-backed exception keeps the complete install explanation visible inside the compact Library shell. Source CSS was rebuilt. Tests cover all seven type groups, both themes/sizes, event ordering, real-shell paint and directory caller continuity.

Verification: 355 targeted checks pass; zero new Ruff diagnostics and formatted changed ranges. Boot CSS is 615,643/634,050 bytes. The existing LibraryScreen and ingest-controller size ceilings still fail at base/current: screen source shrinks four lines, controller unchanged, no budget increased. Final independent review has no actionable findings.

Actual TldwCli/LinuxDriver run-004 passed at 170x48 dark and 80x24 light. Private persistence confirms ten healthy databases, zero media/messages/jobs and unchanged six synthetic files. Normal terminal Quit returned exit 0 before session cleanup. Optional control availability was explicitly simulated; no installation, extraction, import submission or remote/provider operation.

Updated the import guide, feature audit, QA evidence and event/layout verification lessons. QA records earlier harness/exit-receipt mistakes and the remaining queue Retry/fold overlap for the next review. No full suite, push or dev integration.

ADR required: no. Existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md govern this correction. Evidence: Docs/superpowers/qa/2026-09-16-ingest-controls/README.md.
<!-- SECTION:NOTES:END -->
