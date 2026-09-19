---
id: TASK-32837
title: Keep Audit navigation bound to its displayed record and current destination
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 05:23'
updated_date: '2026-09-19 05:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Audit Open tool and Adjust permission actions keep the identity the user selected and handle destinations that disappear during navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retired Audit controls cannot navigate to a replacement record; current controls preserve rendered tool and profile identity.
- [x] #2 Missing destination rows do not leave stale tool or permission detail; current and filtered targets still navigate correctly.
- [x] #3 Targeted ownership/routing regressions and private dark/light native journeys verify the bounded repair.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce held Button.Pressed after Audit replacement/clear and loss of a destination row while navigation waits.
2. Bind Audit actions to their mounted control and immutable rendered context; honor destination-selection failure without stale detail.
3. Preserve profile checks and normal/filtered/missing-tool routing; verify targeted tests, independent review and private native dark/light journeys. Save a separate draft PR against dev.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: repair of existing inspector/action ownership and selection contracts; no new runtime authority, persistent state or application structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bound each Audit action control to its rendered tool/profile identity and invalidated retired controls before asynchronous pruning. Both drill destinations now honor failed row selection and leave detail cleared with the existing warning. Added 14 regressions; all 64 targeted cases and seven preflight guards pass. Ruff introduces no diagnostics and changed ranges/new files pass formatting. Independent read-only review found no blocker. Sixteen inspected native captures cover dark/light at 120x40 and 170x48 with real private catalog rows, filtered destinations and missing-tool warnings; clean exit, unchanged defaults, no execution or permission changes. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-navigation/README.md. Existing ADR-150/161 apply; no new architectural decision. Updated both component review ledgers. Same-ID catalog replacement during an in-flight drill remains a separate next review; final visual approval and current-head CI are required before merge.
<!-- SECTION:NOTES:END -->
