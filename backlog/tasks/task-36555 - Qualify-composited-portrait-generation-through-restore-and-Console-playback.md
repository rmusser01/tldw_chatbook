---
id: TASK-36555
title: Qualify composited portrait generation through restore and Console playback
status: To Do
created_date: 2026-10-24 14:24
dependencies:
- TASK-36554
labels:
- console
- characters
- portrait-compositing
references:
- backlog/decisions/242-console-radio-portrait-performance.md
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-24-portrait-compositing-implementation.md
- backlog/decisions/243-source-bound-portrait-compositing.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the new workflow end to end and document which pairing, persistence and real provider outcomes have actually been proved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An installed actual-app private-profile journey marks/saves regions, generates/retries components, opens the actual Review pairs UI to inspect original-frame eye/mouth crop pixels and Console preview, applies and reopens exact source/settings/provenance/frames; an offline QA sheet cannot substitute for the viewer.
- [ ] #2 Starting a fresh batch from an existing complete applied set and failing components leaves only new-batch partial roles, blocks Apply/sample/publication and preserves active DB frames/revision; a failed retry of a valid complete candidate preserves its previous readiness.
- [ ] #3 A native full-profile restore followed by fresh app open and component retry preserves one-time settings and pairs; equivalent prepared PNG encodings preserve canonical identity/revision/retry, actual source/pixel changes reject reuse, and original bytes/portable exports remain unchanged.
- [ ] #4 The existing normal Console playback and offline/guarded speech preview use the corrected six poses without test-only per-frame publication or a replacement animation loop.
- [ ] #5 Pixel invariants, encoding-equivalence controls, hashes, actual bounded pair-viewer captures and real generated-patch provenance are recorded; real configured in-app adapter qualification is distinct from external patch evidence.
- [ ] #6 Targeted new/affected tests, lint/format, migration packaging, SQL allowlist, token/CSS and Backlog checks pass or unrelated failures are reproduced on pre-change source; no full sweep runs without explicit user request.
- [ ] #7 User/architecture guidance and final self-review explain neutral-base/region reuse, fresh versus retry semantics, canonical identity, three-request consent, in-app raw-frame pair review and import/capability limits; unavailable live evidence stays open.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
