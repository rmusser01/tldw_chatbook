---
id: TASK-1504
title: 'Wizard summary: three-state rows (configured / default / needs attention)'
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: '✓ Theme — textual-dark' claims credit for an untouched default. Matrix should distinguish user-configured from left-at-default from missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rows render three visually distinct states
- [ ] #2 Untouched defaults no longer show the same ✓ as user-configured items
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SummaryRow carries state (configured/default/attention) with derived ok + glyph (✓/–/✗); build_summary_rows: untouched defaults render –, provider-without-model and plaintext-keys render ✗ attention, custom theme earns ✓. Renderer uses row.glyph. Four new unit tests.
<!-- SECTION:NOTES:END -->
