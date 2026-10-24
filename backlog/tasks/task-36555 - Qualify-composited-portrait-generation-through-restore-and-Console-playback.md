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
updated_date: 2026-10-24 22:58
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
- [x] #8 Interrupting opt-in qualification preserves recorded request/artifact evidence and an honest uncertainty report before private temporary cleanup and re-raising the interruption; incomplete logs are retained with an unknown count, and no retry or private-log/credential disclosure occurs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/243-source-bound-portrait-compositing.md
Reason: Existing qualification/privacy and uncertain-paid-request contracts apply; no new provider or architecture.

The installed authoring/Console/F9 restore qualification and existing evidence remain recorded in the linked plan/review. AC5 remains open for a supported configured live image adapter.

Full-review correction:
1. Add red opt-in probe tests that record a dispatched request and then raise KeyboardInterrupt/SystemExit. Assert safe evidence/report preservation, private temporary cleanup, propagation and no retries; include an incomplete manifest.
2. Preserve safe artifacts/count uncertainty before temporary cleanup, write the availability report, then re-raise interruption. Keep private transport logs/credentials out of evidence and subprocess arguments.
3. Run all probe controls, final affected/installed/restore checks and lint/format; review all three corrections, update plan/task evidence and commit. Keep unavailable live qualification AC5 open and status In Progress.
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

Full-review corrections: all three findings are fixed and regression-tested. Probe
interruptions preserve safe evidence/reporting before private cleanup and re-raise;
incomplete manifests retain unknown counts, with no retry or private-log/credential
exposure. Four new red controls now pass among ten probe cases. Installed/restore
fixtures await actual load workers and rendered confirm focus. Final gates: 108
manual/Console/recovery, 68 region/generation/pair, 10 probe and 13 CSS/token tests;
two fresh installed/native journeys passed in 91.15s with original timeouts. Both
wheels match all 18 production files and v83 SQL. All ten revised Python files pass
Ruff/format; diff is clean. Combined corrections review found and fixed an additional
queued-cancellation race; no further defect found. See the corrections review and
QA corrections/ evidence. ADR-243 applies. AC8 satisfied; AC5 remains open, zero live
requests, current default/only enabled backend SwarmUI unsupported by reference path.
PR #3244 rebased onto dev 2f0b142d42cfff88ad4cd44aa04cd9afe068fdf4. Preserved Canvas v81→v82; portrait storage now v82→v83 and regions/provenance v83→v84, with exact recovery lineage and migration dispatch. Updated current-schema fixtures and theme-aware pair-review harness; rebuilt CSS and retained dev markup protection. Feature/integration gates include 89 migration/recovery, 122 core/export/packaging, 323 speech, 80 authoring, 53 Console/adjacent, 10 probe, 13 CSS/token and 15 prompt passes (3 prompt cases overlap core), plus successful fresh installed authoring/Console and original native F9 restore/edit journeys. Native restore originally timed out in its child; a diagnostic run and original uninstrumented isolated run passed with unchanged timeouts. Both successful wheels match all 50 changed production files and the Canvas/portrait/region SQL chain. All 51 failed nodes in broader selected legacy UI/Actor Pack runs reproduce on unmodified dev; no branch-only failed node. No added diagnostics versus exact legacy lint baselines. See Docs/superpowers/reviews/2026-10-24-portrait-compositing-dev-rebase.md and QA rebase/ evidence. Existing ADR-242/243 apply; no new ADR. AC5 remains unchecked and status In Progress: fresh inventory still SwarmUI only, reference path unsupported, zero image requests and live_qualified=false. No full suite or new live image/TTS/hardware qualification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
