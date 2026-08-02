---
id: TASK-2012
title: >-
  Library ingest options panel renders unlabeled checkbox husks and bare values
status: In Progress
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
  - css
priority: high
dependencies: []
---

## Description (the why)

The two highest-impact ingest decisions — "Analyze after ingest" and "Chunk
content" — are invisible in the expanded options panel. The unscoped
`Checkbox { width: 100%; height: 2; }` rule shipped by
`css/features/_conversations.tcss:329` clips the checkbox's content row to a
border-only husk (the bundle's own comments document this same rule breaking
two other screens, which carry per-ID escapes; the ingest panel has none).
Value Inputs use placeholder-as-label, so populated fields show only bare
"1000" / "100" / "auto" with no label anywhere except the collapsed panel
title. An expanded panel also trails ~15 blank rows (unstyled
`.type-group-contents` container height). Found in the 2026-08-02 ingest UAT
(critique snapshot 2026-08-02T21-04-04Z).

## Acceptance Criteria (the what)

- [ ] Both checkboxes render their labels and on/off state in the expanded
      panel.
- [ ] Every non-checkbox, non-select option field has a visible text label
      when populated.
- [ ] An expanded panel has no trailing blank region.
