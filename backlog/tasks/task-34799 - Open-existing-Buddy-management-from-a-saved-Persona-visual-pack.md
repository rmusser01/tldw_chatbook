---
id: TASK-34799
title: Open existing Buddy management from a saved Persona visual pack
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-11 23:39'
updated_date: '2026-10-12 18:05'
labels:
  - personas
  - buddy
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The saved Persona visual-pack browser in PR #2885 adds a Buddy workbench button without a production destination. Current dev already has independent Buddy management and presentation. Give the retained shortcut a working, accurately named destination through that existing owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The saved-pack browser has a truthfully named Buddy management action whose actual activation reaches the existing app-owned manager.
- [x] #2 The action is available only for a saved, clean, complete local pack while authoring is idle; unsaved, remote, dirty, incomplete or busy states cannot open it.
- [x] #3 A stale or detached browser request cannot open management after its exact Persona editor authority changes.
- [x] #4 Opening management does not copy artwork, select a Buddy, write preferences or construct a second presentation/store; current token-backed action and focus styles are retained.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the PR audit, existing Persona editor snapshot authority, Buddy management opener, and current UI token contracts.
2. Add regression coverage for the enabled browser action, saved/clean/local eligibility, and stale or detached request rejection.
3. Route the typed browser request through the current Personas editor guard into the one app-owned Buddy manager; preserve artwork and selection until reviewed Apply.
4. Run targeted widget/production-route/management controls, UI governance and changed-file static checks; obtain independent review before publication.

ADR required: no new ADR.
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md; backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md; backlog/decisions/146-independent-buddy-petdex-publication-and-character-creation.md.
Reason: adds a navigation entry into the existing independent owner without changing persistence, ownership, activity scope or error policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the retained saved-pack shortcut through the existing independent Buddy manager. A typed request carries the exact immutable inventory; existing Persona editor/screen authority and current busy/draft fences reject stale requests. No artwork, selection, preferences, parallel store or presentation are changed by opening. Added ten action/eligibility/stale-request regressions and documented the entry. Native Python3.12 targeted verification: 75 editor/publication/lifecycle, manager, fresh-profile manager journey and token cases pass (213.41s); all derived-artifact preflight guards pass. Configured Ruff adds zero findings against exact9f8533686c base; inherited whole-file diagnostics and unchanged authoring-test formatting remain recorded. UI detector is empty and diff whitespace passes. Existing ADR074/139/146 apply; no new ADR. Independent review, latest sequential base and protected merge remain pending. Source prototype and its full comment disposition are preserved in the review evidence.

Qualification update, 2026-10-12: independent specification and quality reviews passed without actionable findings. The combined production opener/stale-inventory/navigation/detachment controls plus the original import guard passed five cases in13.90s on source now preserved unchanged in the private leaf. The private restack15cf61b4 over notification99df40d6 and accepted dev eb0f7a25 retains all six reviewed feature blobs and every one of the25 concurrent PR3082 base files. The one affected original import-budget check passed again (1pass8.48s); its headroom warning remains visible at552/558 modules and417275/425347 LOC. Original ten/75-case receipts retain their original source attribution and are not claimed as new runs. No token values or production runtime/store ownership changed. All eleven source comment dispositions are prepared; actual post-notification publication, posted replies/current-thread resolution and protected CI remain pending. Inherited whole-file static/formatting debt is unchanged and documented, so no blanket clean-DoD or Task Done claim is made.

Actual sequential base binding, 2026-10-12: PR2874 merged as b47eb46041b59b656359785625220f3175e3bb01 after current-head required CI38284069598 and Perf38284069615 passed. The feature-only Buddy rebase7b3d9c08 onto that actual dev has identical production, test, package, script and workflow trees to locally qualified private15cf61b4; only this qualification-note update follows. Independent final publication binding, all eleven posted comment dispositions and current protected CI remain pending. The shortcut and source inventory authority are unchanged. Task remains In Progress because inherited static/format debt and final publication gates are not relabelled as clean DoD.
<!-- SECTION:NOTES:END -->

## Source reconciliation

PR #2885 source `59a1e69e302d11e14987af2446f97d771cf1d1a2` is preserved at `codex/wave2-source-pr-2885`. Its alternate Persona-only controller, host, widget, JSON store and exclusive test/fixture exemptions are superseded by the current independent Buddy owners. Their resize, stale visibility and repeated asset-load review findings are addressed by excluding the parallel implementation, not by claims of successful qualification.

The accepted first-release contract already supplies trusted authored-trigger APIs. Automatic manifest trigger dispatch and a changed terminal-error lease are preserved source prototypes without accepted delivery criteria here. The unrelated source TASK-31757 describes artwork, so it is not used as this runtime/navigation change's completion record.
