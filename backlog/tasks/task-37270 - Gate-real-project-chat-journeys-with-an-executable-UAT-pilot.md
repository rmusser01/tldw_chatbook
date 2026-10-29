---
id: TASK-37270
title: Gate real project-chat journeys with an executable UAT pilot
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-28 23:58'
updated_date: '2026-10-29 02:23'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent project-chat feature failures from passing component-only checks by exercising the real application journey and retaining actionable evidence. Publish a wider workflow catalog and an initial deterministic project-chat pilot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reviewed catalog defines user actions, independent success criteria, and evidence for core journeys while clearly labeling unimplemented coverage.
- [x] #2 An isolated real-app pilot creates a workspace, binds a fixture folder, starts a scoped chat through the UI, and reads actual project evidence through the production provider and tool path.
- [x] #3 Fenced, native, compact schema recovery, and wrong-family argument recovery variants validate the final assistant answer and durable run; negative controls reject false success.
- [x] #4 Only the owned loopback provider can be contacted; private profile isolation, bounded waits, and failure/cleanup artifacts are verified.
- [x] #5 The exact pilot variants are enrolled in an existing required CI lane with complete no-skip result checking and retained diagnostic artifacts after targeted qualification.
- [x] #6 Documentation records exact commands and measured platform coverage, distinguishing scripted Windows/Linux runs from unperformed live Fedora qualification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-069/174/241/253 apply. Review and tighten the design, then write the implementation plan. Build the HTTP script and independent oracle tests first, exercise real startup and UI creation/send in the pilot, qualify negative controls, and enroll exact cases with retained artifacts in the existing CI gate. No full-suite run or live model is part of this first pilot.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the reviewed critical-journey catalog and a real TldwCli project-chat pilot with four scripted-provider variants. UI actions create the workspace/binding/chat, accept project-instruction disclosure and the built-in listing approval when required; actual provider requests, real file tools, private DB call IDs/ownership, and painted/saved answers establish provenance. No product modules or permission gates changed.

The existing admission-sensitive 2 lane now rejects missing/skipped cases and incomplete receipts, retains synthetic diagnostics, and runs a full missing-evidence negative control. Independent review false-green findings (model identity, absent schemas, wrong saved-answer ownership) were fixed with negative tests. Trace bodies remain private; the oracle correlates DB headers with actual HTTP history.

Validation: code head 4c0ce64d6c, Linux run 40805071773: 59 passed in 171.593s; expected missing-code failure after real fs_read verified in 37.802s. Both downloaded report sets independently verified. Local targeted checks: 56 support/report/enrollment passed, 147 affected CI contracts passed; Ruff and formatting clean. No full-suite sweep. Exact commands/timings/coverage are in Docs/superpowers/specs/2026-10-28-critical-journey-uat-design.md.

ADR required: no; existing ADR-069/174/241/253 apply. Consent/trace-privacy lesson added to backlog/docs/lessons-testing-evidence.md. Modified files comprise the catalog/plan, pilot/helpers/report checker, workflow and CI contracts.

Task stays In Progress while the complete existing PR gate finishes. Windows qualification failed with a Textual workspace-section timeout before provider entry; live Fedora/Python 3.14/model/native-terminal qualification is not run. Other catalog rows remain planned. These limitations are explicit, not converted into passing evidence.
<!-- SECTION:NOTES:END -->
