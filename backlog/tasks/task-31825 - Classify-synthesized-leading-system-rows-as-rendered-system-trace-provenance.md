---
id: TASK-31825
title: Classify synthesized leading system rows as rendered-system trace provenance
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-06 06:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a scratch-only warm Chat handoff repair, the real Capture-On character UAT stops with TraceProvenanceAlignmentError: system category mismatch. Unsaved synthesized leading system rows receive ACTIVE_REQUEST fallback provenance, but the existing semantic system category requires RENDERED_SYSTEM. Real DB persistence is healthy. A scratch descriptor correction makes the full UAT pass; disabling capture would hide the production defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A durable Capture-On send with synthesized leading system content reaches the provider with RENDERED_SYSTEM artifact provenance for that content.
- [ ] #2 Saved revision ownership remains unchanged, and ordinary unsaved active rows retain ACTIVE_REQUEST provenance; nonleading system rows are not reclassified indiscriminately.
- [ ] #3 The typed negative regression fails before correction and passes afterward, and complete character UAT passes with approved warm-resume handling.
<!-- AC:END -->
