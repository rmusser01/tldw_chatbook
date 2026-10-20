---
id: TASK-35677
title: 'Give Settings composer-paste controls local widget ownership'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-19'
labels:
  - settings
  - textual-patterns
  - maintainability
dependencies:
  - TASK-35676
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Priority 2 of the Textual-book follow-ups: pilot cohesive widget ownership on the small Composer paste handling section of canonical Settings. Keep the paste-collapse toggle, threshold input and their local presentation together so SettingsScreen no longer wires each child control separately. This is one bounded region, not a whole Settings/Library/Console decomposition programme.

Preserve the screen-owned category draft, Save/Revert writer, validation policy, field-search/focus integration and existing configuration keys. The region owns its child widgets and local refresh behavior and communicates through the established widget-message convention; it must not acquire authority to save configuration. Audit callers, tests and field ownership before the move; do not substitute a generic form framework, a proxy to the entire screen, or a second settings surface.

Apply chapter 9's refresh/recompose distinction here: typing and summary updates retain the Input instance, cursor, selection and focus. Recomposition is permitted only when the section's structure genuinely changes. Existing targeted Library ingest updates and Console row reconciliation are precedents, not additional work in this task.

Book basis: chapter 7 custom messages; chapter 9 pp. 167-179; chapter 17 pp. 319-320; chapter 21 pp. 421-422. Existing governing design: Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md. Consult the generalizable ownership/caller census guidance in backlog/docs/library-decomposition-recipe.md; its completed Library programme is not a new dependency.

ADR required: no new ADR expected.
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/161-component-pattern-library.md.
Reason: bounded internal UI extraction under the approved screen-decomposition design, preserving settings and service boundaries. If the ownership census requires a different persistent state owner, service contract or long-lived UX structure, record a new ADR before implementation and rescope rather than expanding silently.

Deduplication: TASK-32809.3 records decomposition candidates and is not an authorization for a broad rewrite. TASK-35017 owns page order, disclosures and field-search visibility; preserve its current/landed layout when integrating, without implementing its pending work here. TASK-33007.7 already extracted shared global-fallback rows. TASK-21116/TASK-31201 own the completed Library recompose work/ratchet; TASK-34939 owns the completed status-widget repaint fix.

### Related work and decisions

- [TASK-35676](./task-35676%20-%20Keep-Console-Behavior-form-refreshes-from-staging-user-edits.md): establish the paste-control event contract before moving it.
- [TASK-32809.3](../tasks/task-32809.3%20-%20Record-the-decomposition-candidates-without-opening-one.md): existing owner or precedent described above.
- [TASK-35017](../tasks/task-35017%20-%20Open-Console-Behavior-on-chat-defaults-and-fold-agent-capture-and-replay-sections-into-Advanced.md): existing owner or precedent described above.
- [TASK-33007.7](../tasks/task-33007.7%20-%20Give-Console-Behavior-s-global-fallbacks-the-shared-rows-and-one-streaming-control.md): existing owner or precedent described above.
- [TASK-32761](../tasks/task-32761%20-%20Qualify-Console-Behavior-and-Storage-Settings-workflows.md): existing owner or precedent described above.
- [TASK-21116](../tasks/task-21116%20-%20Library-screen-still-whole-screen-recomposes-on-per-click-paths-continue-the-canvas-scoped-conversion-with-a-ratchet.md): existing owner or precedent described above.
- [TASK-31201](../tasks/task-31201%20-%20Recompose-census-needs-an-anti-slack-guard-like-the-size-ratchet.md): existing owner or precedent described above.
- [TASK-34939](../tasks/task-34939%20-%20Repaint-EnhancedStatusWidget-instead-of-recomposing-per-message.md): existing owner or precedent described above.
- [033-settings-commit-models-three-honestly-labeled](../decisions/033-settings-commit-models-three-honestly-labeled.md)
- [150-design-token-system-and-design-language](../decisions/150-design-token-system-and-design-language.md)
- [161-component-pattern-library](../decisions/161-component-pattern-library.md)

- [Approved screen decomposition design](../../Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md)
- [Library decomposition recipe](../docs/library-decomposition-recipe.md)
- [Design language](../docs/design-language.md) and [component patterns](../docs/component-patterns.md)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A caller and field-ownership census bounds the move to Composer paste handling and records its compatibility with the approved decomposition design and TASK-35017; no other settings section is extracted.
- [ ] #2 The section owns child composition, local update handling and presentation; Settings retains the existing category draft and save authority, with no whole-screen proxy or duplicate configuration state.
- [ ] #3 Actual typing, toggling and summary refresh preserve the mounted threshold Input, focus, cursor and selection; ordinary value changes do not recompose the enclosing Settings detail pane.
- [ ] #4 Existing staged Save/Revert, validation, field search, category return, tab/focus navigation and programmatic-update suppression behave as before; no new shortcuts or settings destination are introduced.
- [ ] #5 Focused component and mounted Settings tests plus a private-profile keyboard journey cover the extracted region, including compact and wide layouts. Existing tokens/classes and appropriate size/import ratchets are preserved or updated to measured values.
- [ ] #6 The relevant architecture documentation records the region's owner and message flow; any departure from the existing ownership contract has an ADR before code changes.
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/161-component-pattern-library.md
Reason: bounded changes under the existing Settings and approved screen-decomposition contracts; persistence and service ownership stay with their existing owners.
1. Census paste control callers, field-search selectors and existing layout rules.
2. Move composition and local events into one ComposerPasteSettingsPanel; retain screen-owned draft/validation/save.
3. Preserve IDs and token-backed layout, use targeted sync, and prove Input identity/focus/cursor survive updates.
4. Run affected Settings tests, size/import/token checks and private-profile keyboard journeys.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added ComposerPasteSettingsPanel as the sole owner of paste-control composition, displayed state and local Changed messages. The screen receives Edited and retains its existing draft, validation and writer; no config proxy or second save path was introduced. Sync updates existing child instances under prevent(...), preserving equal-value selection and focus. Field IDs, search integration, copy and CSS classes remain; a scoped token-backed selector preserves the original Static inset. The widget imports only when the category renders.

The caller/field census and compact/wide mounted typing, selection, search and category-return evidence are in [qualification](../../Docs/superpowers/qa/2026-10-19-textual-settings-patterns.md). Architecture/app-shell.md records the owner/message flow. SettingsScreen is 35,805 lines and its size ratchet was tightened; pre-import payload and generated CSS checks pass. Existing global governance failures and the unavailable pinned Mermaid interpreter are disclosed rather than changing their limits. No TASK-35017 layout work or other section extraction was included.

ADR required: no new ADR; existing [ADR-033](../decisions/033-settings-commit-models-three-honestly-labeled.md), [ADR-161](../decisions/161-component-pattern-library.md) and the approved screen-decomposition design apply. Modified files: the new widget/component tests, screen, source/generated Settings CSS, size ratchet, PR UI census and architecture documentation.
<!-- SECTION:NOTES:END -->
