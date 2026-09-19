---
id: TASK-32667
title: Keep import queue keyboard focus above the fold hint
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:09'
updated_date: '2026-09-16 05:06'
labels:
  - library
  - design-system
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the queue activity and recovery review. Keyboard users must be able to see the action they reach below a long import form, including Retry this batch and queue recovery controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tab and reverse Tab reveal Retry and queue controls above the docked hint at wide and compact sizes in both themes.
- [x] #2 Queue updates and recovery disclosures preserve readable keyboard context and the staged draft without activating an import.
- [x] #3 Targeted regressions, isolated native verification, static checks and review evidence are recorded with documentation.
- [x] #4 Retry confirmation visibly names replacement of the form; its full label updates without clipping or losing focus.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Restore the existing visible-focus contract within the import canvas; no queue ownership, consent, persistence or navigation boundary changes. TASK-3310 keeps the queue on this canvas.

1. Reproduce the Retry/fold overlap with actual keyboard traversal in the production Library shell. Trace Textual containment and dock-aware scrolling; audit queue controls and updates using synthetic registry jobs only.
2. Add failing wide/compact, dark/light compositor journeys. Repair the canvas focus boundary with the existing dock-aware scrolling pattern and preserve draft/queue semantics.
3. Run targeted queue, Retry, consent and option neighbors, changed-file static/format checks and relevant governance; request focused review.
4. Verify real TldwCli in an exclusive private profile at 170x48 dark and 80x24 light using synthetic queue projection only. Capture a bounded visual batch, check persistence isolation and normal exit, document the audit and commit locally. No full suite, actual ingestion, install, provider/server request, push or dev integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed import queue keyboard continuity: capture focus at the actual queue rebuild, restore it ahead of older action callbacks, honor newer focus, and skip unavailable actions before source-field fallback. The canvas reveals focused controls above both docks, stops stale scroll animation, and rechecks retained focus after content-size layout. Retry confirmation now invalidates measured width so its complete form-replacement label paints.

Verification: 203 targeted checks pass, including 16 new production-CSS queue journeys. Native TldwCli/LinuxDriver run-011 passes 170x48 dark and 80x24 light; all six captures inspected, normal exit 0, ten healthy private databases, zero media/messages/ingest jobs, unchanged source/default-profile hashes, no app log errors. Synthetic registry only; no actual import or external operation. Zero new Ruff diagnostics; new files and changed ranges formatted; diff whitespace clean. The two inherited screen/controller size ceilings remain failing at base/current; no budget raised. Independent final review has no remaining findings.

Updated the import guide, workflow audit, QA evidence and incident-backed testing/live-verification lessons. Existing ADR-014, ADR-150 and ADR-161 apply; no new ADR or CSS/token changes. A general deferred-scroll guard did not address the traced layout clamp and was removed; the existing Library rail pattern supplies the final repair. Broader live-resize focus ownership, grouped outcomes and execution/recovery remain for subsequent review. No full suite, push or dev integration. Evidence: Docs/superpowers/qa/2026-09-16-ingest-queue/README.md.
<!-- SECTION:NOTES:END -->
