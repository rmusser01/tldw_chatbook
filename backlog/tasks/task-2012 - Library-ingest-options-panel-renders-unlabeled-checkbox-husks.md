---
id: TASK-2012
title: >-
  Library ingest options panel renders unlabeled checkbox husks and bare values
status: Done
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

- [x] Both checkboxes render their labels and on/off state in the expanded
      panel.
- [x] Every non-checkbox, non-select option field has a visible text label
      when populated.
- [x] An expanded panel has no trailing blank region.

## Implementation Notes

Scoped TCSS escapes in `css/components/_agentic_terminal.tcss` (where the
other `library-ingest` rules live): `LibraryIngestCanvas
.type-group-contents Checkbox { width: auto; height: auto; }` against the
unscoped `_conversations.tcss` rule, `.type-group-contents { height: auto }`
(kills the trailing phantom rows — the container defaulted to fr height),
and a `.type-group-field-label` style for the new `Static` labels
`_compose_type_group` now emits before every value Input. Bundle
regenerated via `build_css.py`; source + bundle committed together.
Live-verified (2026-08-02): `▐X▌ Analyze after ingest` / `▐X▌ Chunk
content` render with labels; `Chunk size`/`Chunk overlap`/`Encoding`
labels sit above `1000`/`100`/`auto`; panel ends at "Reset to defaults"
with no blank region. Residual (stock Textual): the checkbox glyph is
always "X" — on/off is carried by the X's color plus the panel title's
"on/off" text; a glyph-level state needs a ToggleButton subclass
(candidate for TASK-2015's a11y sweep).
