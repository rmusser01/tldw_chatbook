---
id: TASK-24609
title: Unify Inspect rail heading treatments and heading-to-body alignment
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:54'
updated_date: '2026-08-30 02:46'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
  - css
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail uses five different heading treatments, and console-settings-title declares neither text-style nor color with no rule in scope to supply them, so the Session Settings title renders identically to the eight rows it heads. The focus cue adds a sixth bold treatment on top. Separately, section headings render at a two-column indent while their own body rows render at one, so every heading is off-axis from its content by one cell.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every section title in the Inspect rail uses one shared treatment
- [x] #2 The raised-background treatment is reserved for run-inspector sub-groups only
- [x] #3 Headings and their body rows share a left alignment column
- [x] #4 The focus cue remains visually distinct from every heading treatment
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two defects, both measured rather than inferred.

1. .console-settings-title declared neither text-style nor color, and the destination-section class it also carries has no rule in scope inside the rail, so 'Session Settings' rendered identically to the eight .console-settings-row lines it heads. It now takes the same bold + $ds-text-primary treatment as .console-rail-section-title.

2. Headings painted one cell right of their own rows. FIXED BY REMOVING THE HEADING'S INDENT, not by indenting the rows. The first attempt gave rows 'padding: 0 0 0 1' to meet the heading; that costs every inspector row one column of content width, and at the live-work card's 39-column row width it pushed rows over and moved a bounded section's measured demand from 21 to 22, failing the twenty/twenty-one swap-geometry pin. '.console-inspector-group-heading { padding: 0 }' aligns them for free and the raised background still spans the full row. Worth remembering: in a 33-column rail, padding is content.

Testing note: the alignment test first passed vacuously. InspectorHarness is a bare App with no CSS_PATH, so the heading's padding never applied and there was nothing to misalign. It now subclasses with the bundled stylesheet and carries an explicit guard that the padding is actually 1 before comparing.

Deferred, still open in the critique: .console-changed-files-header and .ds-status-badge remain separate heading treatments; consolidating those touches widget-local CSS outside this task's ACs.

Modified: tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_run_inspector.py.
<!-- SECTION:NOTES:END -->
