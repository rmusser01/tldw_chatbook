---
id: TASK-21163
title: Order model catalog consent after intro startup
status: Done
assignee: []
created_date: '2026-08-23 15:09'
updated_date: '2026-08-23 17:04'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix startup ordering so an unconfigured model-catalog consent decision appears after the splash/intro, remains topmost and actionable, and reveals a usable initial app screen after Yes or No.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Splash/intro completes before model-catalog consent is shown.
- [x] #2 Unrecorded consent produces one topmost Yes/No modal that cannot be buried by initial-screen startup.
- [x] #3 Either consent choice dismisses to a usable initial app screen while preserving the existing persistence and refresh semantics.
- [x] #4 Focused regression coverage drives the real splash-enabled screen stack and fails under the pre-fix ordering.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a splash-enabled full-app Textual Pilot regression for intro -> Home -> consent, prove the old stack is Screen/Consent/Home, and exercise explicit Deny back to Home.
2. Move startup scheduling from on_mount to the shared post-intro _push_initial_screen path without changing ADR-020 consent persistence or refresh semantics.
3. Add red regressions for the reviewed competing paths: eligible project-skills startup discovery must defer when consent is unrecorded, and completed first-run exit navigation must settle before consent without any implicit None decision.
4. Record the unrecorded-consent startup branch so _push_initial_screen can skip the optional project-skills offer for that launch; sequence completed first-run exit navigation before scheduling consent while preserving no-route and same-Console behavior.
5. Run the complete focused consent, catalog-wiring, first-run startup, and live first-run contract tests; run Ruff and git diff --check; complete independent review.
6. Check acceptance criteria, add implementation notes, and set the task Done only after verification.

ADR required: no
ADR path: backlog/decisions/020-automatic-model-catalog-refresh.md
Reason: This is a lifecycle ordering bug fix that preserves ADR-020 storage, consent, and network boundaries; optional startup prompts are deferred rather than introducing a new modal coordinator.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the startup lifecycle fix so model-catalog consent is scheduled only after the intro and initial routed screen are mounted. Unrecorded consent now owns the launch, deferring the optional project-skills offer; completed first-run exit navigation is awaited before consent, while navigation failures still release consent over the surviving screen and cancellation during shutdown does not. Existing Yes/No persistence, one-shot scheduling, setup ownership, and catalog refresh behavior remain unchanged.

Regression coverage now drives the real splash-enabled Textual screen stack, explicit Deny interaction, project-skills precedence, first-run route sequencing, navigation failure, and cancellation. The original ordering failed with Screen -> ModelCatalogConsentModal -> HomeScreen; the corrected modal suite passes 8/8. Catalog wiring passes 23/23; focused first-run callback/live and focus tests pass. Ruff and git diff --check pass. The complete product-maturity first-run file has one pre-existing navigation-copy timeout at line 552, reproduced identically on base 80e8b50e6; all other tests in that file pass.

ADR required: no. Existing ADR: backlog/decisions/020-automatic-model-catalog-refresh.md. This change preserves that decision and corrects UI lifecycle ordering only. No new dependency, schema, security boundary, or generalized modal coordinator was introduced. No reusable lessons document update was warranted.

PR review follow-up: added the missing async contract-test docstring and replaced the splash regression's short manual Pilot loops/exact default-screen assertion with the repository's bounded `_wait_until` pattern and behavior-level splash/Home ordering assertions. Focused tests and Ruff pass.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

- Renumbered from `TASK-21161` to `TASK-21163` on 2026-08-23 after latest
  `dev` exposed a duplicate task ID.
- The Console private-scratch task's add commit
  `47a417bd13f9fc179e30d3acbec7743edaa86e96` predates this task's add commit
  `7969089c34b163159d21f4a35f7f7a716bc289eb`, so the later claimant moves
  under the repository's older-arrival tie-break rule.
- Every task-file, design-spec, and implementation-plan reference shipped with
  this startup-order slice was updated to `TASK-21163`.
