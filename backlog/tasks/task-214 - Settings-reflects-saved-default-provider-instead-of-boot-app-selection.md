---
id: TASK-214
title: Settings reflects saved default provider instead of boot app selection
status: Done
assignee:
  - '@codex'
created_date: '2026-07-12 12:44'
updated_date: '2026-09-17 17:14'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual from the 2026-07 core-loop waves (same family as the fixed task-177): after the setup card's one-click local connect writes chat_defaults.provider, the Settings Providers & Models category still displays the boot-time app selection (e.g. OpenAI, 'Provider source: Current app selection') until the user reselects, while Console correctly runs the saved provider. Evidence: Docs/superpowers/qa/core-loop-upgrades-2026-07/README.md residuals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening clean Providers and Models after a config-level default change shows the current saved provider, model and endpoint without reselection, including restored and retained screen returns.
- [x] #2 Readiness and provider-test inputs agree with a default-inheriting Console; an explicit Console session selection and unsaved Settings draft keep their existing ownership.
- [x] #3 Production-CSS targeted tests and private native dark/light wide/compact journeys verify the behavior; guide, audit, static checks and review evidence are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: existing backlog/decisions/006-provider-aware-generation-settings.md, backlog/decisions/012-provider-credential-settings-boundary.md and backlog/decisions/033-application-session-state-ownership.md; ADR-031/150/161 govern keys and visual components. Reason: verify the original report against TASK-648 ownership and repair only routine stale projection defects if reproduced.
1. Trace current Settings provider resolution, snapshots/resume and Console defaults/session precedence. Add focused cases for saved-default changes across fresh, restored and retained visits, with draft/session preservation.
2. Repair any reproduced mismatch at its existing owner/refresh boundary; add regression evidence before production edits and run targeted provider/navigation tests.
3. Verify real config persistence and visible provider/readiness/test identity in a private native app across dark/light at 170x48 and 80x24. Review and update guide/audit/task; commit locally. No full suite, push or merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Clean retained Providers & Models now follows saved provider/model/endpoint changes, while sparse unsaved drafts preserve their owning provider/model. API-mode-only edits share the identity pin; later fields use the pinned originals. Unchanged alias/custom-provider returns preserve widgets and test verdicts.

Added 26 production-CSS regressions; all 320 selected tests and 21 additional Console fixture/privacy checks pass. Corrected three baseline harness assumptions (obsolete runtime fake, category-first search priority, off-screen clicks). Four final native dark/light wide/compact journeys verify real config writes, restoration, draft discard and loopback catalog probes; eight captures inspected. Eleven private databases, default-file fingerprints and normal process/server shutdown verified. No generation, full suite, push or merge.

Existing ADR-006/012/033/031/150/161 apply; no new ADR required. Guide, July residual, audit and testing lesson updated. QA: Docs/superpowers/qa/2026-09-17-settings-saved-provider/README.md. Independent review has no remaining findings; changed/new code formatting passes and no lint diagnostics were added. The small baseline harness repairs are the only plan deviation.
<!-- SECTION:NOTES:END -->
