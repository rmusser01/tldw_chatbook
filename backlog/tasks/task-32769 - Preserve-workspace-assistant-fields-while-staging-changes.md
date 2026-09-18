---
id: TASK-32769
title: Preserve workspace assistant fields while staging changes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 04:53'
updated_date: '2026-09-18 05:19'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Changing only a workspace persona or tool profile currently drops the other saved field or changes memory mode. Keep staged selections and the Apply action consistent with the persisted defaults, while retaining explicit read-write acknowledgement and imported-profile first-bind review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Changing persona retains the chosen tool profile and starts a new persona read-only; changing only the profile preserves the saved persona and memory mode.
- [x] #2 Selections remain staged until Apply, read-write changes require their existing explicit confirmation, and failures can be retried without losing the intended fields.
- [x] #3 The displayed selections, memory action and resulting registry values agree in compact and wide layouts in both themes, with usable attached focus after refresh.
- [x] #4 Existing clear, unavailable/default-workspace, newer-staging and imported-profile first-bind safeguards pass targeted regressions, with representative native evidence and scoped static checks.
- [x] #5 Staging, confirmation and apply outcomes stay beside assistant controls; deferred reveal must not move another workspace or steal focus after navigation.
- [x] #6 Long persona/profile lists keep the highlighted option and adjacent feedback visible without growing to consume the whole compact viewport.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/079-workspace-assistant-defaults.md, backlog/decisions/107-portable-tool-use-packs.md, backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md, backlog/decisions/150-design-token-system-and-design-language.md, backlog/decisions/161-component-pattern-library.md (existing). Reason: preserve existing saved-field, staging and confirmation contracts without changing authority, schema or commit semantics. 1. Retain three reproduced keyboard/real-registry failures for persona-only and profile-only changes; mounted staging layout checks already pass. 2. Seed the pending selection from the saved default before changing one field, preserving the existing new-persona read-only rule; make profile staging feedback truthful. 3. Add keyboard journeys for each field direction, staged versus persisted state, read-write confirmation, failure/retry and focus, using production CSS at dark/light and 80x24/170x48; execute the original assistant tests under established private-profile ownership. 4. Run targeted service/runtime/UI and static checks, obtain independent review, inspect real native private-profile captures and record lifecycle plus remaining workstream limits. Allocation: fetched origin; all reachable object paths, 316 refs, 30 worktrees; maximum32768 and no content reference to32769 before CLI allocation.
The expanded keyboard journey reproduces compact Apply focus outside the viewport after a pane rebuild. Keep assistant feedback local and workspace-scoped, and reveal the current action/result after the replacement pane has laid out. Preserve this failure and check confirmation text plus attached focus. Rapid repeated Enter waits for the normal Textual button active interval in the test; production debounce is unchanged.
Independent review reproduced receipt reveal hiding the highlighted option in a viewport-height picker with 20 entries. Bound these two Settings pickers with the existing ds-size-5 token, leaving space for the originating action and receipt, and test populated lists in the existing matrix. Rebuild source CSS and recheck governance.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Persona-only staging retains the chosen profile; profile-only staging retains saved persona/memory. Existing new-persona read-only initialization, explicit read-write acknowledgement and imported-profile first-bind token gates remain. Assistant outcomes now use one plain-text receipt anchored to its originating control, with exact workspace/result/focus guards, immediate scrolling and a resize hook for actual text reflow. Two token-backed bounded pickers preserve full selected text and avoid the generic overpainting outline; generated CSS rebuilt. Verified87 distinct targeted cases (four populated keyboard journeys,16 original assistant cases,36 related registry/session cases,31 governance checks), scoped Ruff/format and unchanged116 legacy diagnostics, Backlog and diagnostic guards, independent review, and four real native private-profile cells with12 inspected captures. Native004 exited0 normally with11 healthy databases and unchanged default fingerprints; earlier diagnostic attempts are accounted. Existing ADR-079/107/139/150/161 apply, no new ADR. QA: Docs/superpowers/qa/2026-09-17-settings-workspace-assistant/README.md. A pre-existing confirmation-arm roundtrip and broader Workspace modal/consent flows remain explicit follow-ups; no full suite or provider execution claimed.
<!-- SECTION:NOTES:END -->
