---
id: TASK-36555
title: Qualify composited portrait generation through restore and Console playback
status: In Progress
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
assignee:
- '@codex'
updated_date: 2026-10-24 17:53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the new workflow end to end and document which pairing, persistence and real provider outcomes have actually been proved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An installed actual-app private-profile journey marks/saves regions, generates/retries components, opens the actual Review pairs UI to inspect original-frame eye/mouth crop pixels and Console preview, applies and reopens exact source/settings/provenance/frames; an offline QA sheet cannot substitute for the viewer.
- [x] #2 Starting a fresh batch from an existing complete applied set and failing components leaves only new-batch partial roles, blocks Apply/sample/publication and preserves active DB frames/revision; a failed retry of a valid complete candidate preserves its previous readiness.
- [x] #3 A native full-profile restore followed by fresh app open and component retry preserves one-time settings and pairs; equivalent prepared PNG encodings preserve canonical identity/revision/retry, actual source/pixel changes reject reuse, and original bytes/portable exports remain unchanged.
- [x] #4 The existing normal Console playback and offline/guarded speech preview use the corrected six poses without test-only per-frame publication or a replacement animation loop.
- [ ] #5 Pixel invariants, encoding-equivalence controls, hashes, actual bounded pair-viewer captures and real generated-patch provenance are recorded; real configured in-app adapter qualification is distinct from external patch evidence.
- [x] #6 Targeted new/affected tests, lint/format, migration packaging, SQL allowlist, token/CSS and Backlog checks pass or unrelated failures are reproduced on pre-change source; no full sweep runs without explicit user request.
- [x] #7 User/architecture guidance and final self-review explain neutral-base/region reuse, fresh versus retry semantics, canonical identity, three-request consent, in-app raw-frame pair review and import/capability limits; unavailable live evidence stays open.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/243-source-bound-portrait-compositing.md
Reason: Qualify the implemented component workflow across installed UI, native recovery and unchanged playback/export boundaries; no new architecture.

1. Read Unit 5 and installed Console/restore harnesses; identify actual configured reference-capable backends without exposing credentials.
2. Extend the installed production journey through region Save/reopen, exactly three validated component requests, real pair viewer, retry, manual/Apply and Console behavior using fake external transport only.
3. Extend full native F9 backup/restore/fresh-open through recovered settings/provenance and another one-request component retry/Apply, with original source and stale ownership controls.
4. Add an explicit opt-in real-adapter qualification probe and run it only when a supported configured destination is available and authorized; keep missing live coverage open and accurately documented.
5. Run affected installed/restore/export/core/UI checks, verify artifact hashes and final docs, self-review every criterion and commit proved qualification work. Do not mark Done with unfulfilled live acceptance criteria.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented and self-reviewed installed-wheel authoring/pair-viewer/Console qualification, native F9 restore through another confirmed component retry, prepared-PNG encoding equivalence and changed-source/pixel rejection, with original/portable boundaries preserved. Added external-only image/TTS/device fixtures retaining real WAV/PCM analysis, ownership and avatar timers; no fabricated speech snapshots or direct ticks. Added an opt-in five-request real-adapter probe: no fallback/automatic retry, private temporary credentials, retained uncertainty counts and prior-evidence protection.

Evidence: installed journey 1 passed (42.88s); native F9 restore/edit 1 passed (52.81s); probe controls 6 passed (0.32s); affected core/draft/repository/migration/exports/SQL/token/CSS gate 99 passed (40.85s). Both wheels match all 18 changed production files and include exact v83 SQL; 15 visual capture hashes recorded. Ruff/format and git diff checks pass. No full suite, push or merge.

AC5 REMAINS OPEN; status intentionally In Progress. Actual current default/only enabled image backend is SwarmUI, unsupported by this reference-image path; zero live image requests sent and no fallback selected. Historical external ChatGPT patch evidence is distinct from real in-app adapter qualification. No new hardware/live TTS claim. See Docs/superpowers/reviews/2026-10-24-task-36555-review.md and Docs/superpowers/qa/portrait-compositing/2026-10-24/qualification.md. ADR required: yes; existing backlog/decisions/243-source-bound-portrait-compositing.md applies.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
