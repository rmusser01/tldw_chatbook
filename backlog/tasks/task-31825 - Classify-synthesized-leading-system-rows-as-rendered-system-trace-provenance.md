---
id: TASK-31825
title: Classify synthesized leading system rows as rendered-system trace provenance
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 06:21'
updated_date: '2026-09-06 15:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a scratch-only warm Chat handoff repair, the real Capture-On character UAT stops with TraceProvenanceAlignmentError: system category mismatch. Unsaved synthesized leading system rows receive ACTIVE_REQUEST fallback provenance, but the existing semantic system category requires RENDERED_SYSTEM. Real DB persistence is healthy. A scratch descriptor correction makes the full UAT pass; disabling capture would hide the production defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A durable Capture-On send with synthesized leading system content reaches the provider with RENDERED_SYSTEM artifact provenance for that content.
- [x] #2 Saved revision ownership remains unchanged, and ordinary unsaved active rows retain ACTIVE_REQUEST provenance; nonleading system rows are not reclassified indiscriminately.
- [x] #3 The typed negative regression fails before correction and passes afterward, and complete character UAT passes with approved warm-resume handling.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow approved-console-regressions plan TASK31825: RED durable Capture-On synthesized leading system descriptor, classify only leading unsaved system fallback as RENDERED_SYSTEM while preserving saved and active descriptors, complete provenance/controller/UAT verification and review. ADR required: no; restore existing ADR097 category contract without capture bypass or schema change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved ADR097 repair: preserve saved revision descriptors, classify only unsaved contiguous leading system rows as RENDERED_SYSTEM, leave nonleading and ordinary active artifacts ACTIVE_REQUEST. Durable typed RED and real provider/native SQLite Capture-On tests qualify the change. Current durable/prepared/provenance files pass 91; combined unchanged size-guard selection is 95 passed / 1 existing size failure (/private/tmp/tldw-approved-trace-size-final.xml). Character UAT plus handoff controls pass 8 with no retained SQLite after fixture cleanup (/private/tmp/tldw-31823-resource-retry.xml). Independent code/spec review has no findings; controller retains exactly its 27 baseline lint findings, changed tests/ranges are clean. Two separately discovered route-census failures predate this repair and reflect summary-owner drift, not merely offsets; they are not waived or counted passing. No new ADR or capture bypass. Modified console_chat_controller.py and test_console_durable_turn_acceptance.py.
<!-- SECTION:NOTES:END -->
