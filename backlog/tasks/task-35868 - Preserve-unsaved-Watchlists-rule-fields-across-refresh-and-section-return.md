---
id: TASK-35868
title: Preserve unsaved Watchlists rule fields across refresh and section return
status: Done
created_date: 2026-10-20 19:25
labels:
- watchlists
- textual-patterns
- correctness
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md
assignee:
- '@codex'
updated_date: 2026-10-20 21:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Editing a new or existing Alert Rule must survive a changed Rules list and a temporary switch to another Watchlists section. Refreshing presentation must not silently erase the user draft or retarget it.

### Evidence and bounded scope

RulesPane.rules uses recompose=True at rules_pane.py:73; compose():131-196 recreates fields from selected_rule/defaults. The live _load_rules assignment is watchlists_collections_screen.py:12890. A minimal mounted real-pane probe set Name to Unsaved audit rule, then changed rules: Name became empty and the old Input detached/replaced. The probe did not assert settled pre-refresh focus. Section reconstruction retains only form visibility/original editing row (:1092-1093, :2807-2820, :12904-12915), so section-return draft loss is statically established and still needs the full keyboard journey.

Book basis: Chapter 9 printed pp.170-171 / PDF pp.179-180 and printed pp.179-180 / PDF pp.188-189; custom-message ownership printed pp.138-142 / PDF pp.147-151.

### Existing ownership and decisions

TASK-2200 preserved form visibility/original seeding during unrelated workbench refresh, TASK-15461 scoped rebuilds, TASK-15778 fixed pre-mount seeding, and TASK-2310 updates condition guidance in place. None retains the edited field draft across a changed list or Rules-section reconstruction.

Related tasks: [TASK-2200](./task-2200%20-%20Watchlists%20screen%20stops%20full-screen%20recomposing%20from%20background%20loaders.md); [TASK-15461](./task-15461%20-%20Watchlists---one-scoped-rebuild-per-section-or-tree-interaction.md); [TASK-15778](./task-15778%20-%20Watchlists-batch-the-cold-Read-tab-region-swap-and-drop-the-pre-mount-seeding-recompose.md); [TASK-2310](./task-2310%20-%20Label%20the%20Watchlists%20toolbar%20Selects%20and%20stage%20the%20Artifacts%20toolbar.md).

ADR required: no new ADR expected.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [042-watchlists-reader-first-ia](../decisions/042-watchlists-reader-first-ia.md); [161-component-pattern-library](../decisions/161-component-pattern-library.md).
Reason: No new ADR is expected for a Watchlists-local view-state correction within the existing pane/screen owner. Do not introduce a root application state, another editor surface, or new storage/sync conflict policy.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A changed Rules-list refresh preserves unsaved Name, Condition, Threshold, Severity and Enabled values in both New and Edit forms, with the same target rule identity.
- [x] #2 A settled focused editable control retains its instance, focus, cursor and selection during list-only refresh; tests establish these preconditions before refreshing.
- [x] #3 Switching Rules to another Watchlists section and back restores the unsaved new/edit draft and its original target without an implicit save.
- [x] #4 Explicit Save and Cancel retire only the appropriate draft; validation failure retains it, and existing condition-guidance, backend eligibility and mutation behavior remain intact.
- [x] #5 Mounted tests type real changes before both triggers, cover New/Edit and Save/Cancel, and retain the completed visibility/seeding/guidance regressions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/042-watchlists-reader-first-ia.md; backlog/decisions/161-component-pattern-library.md
Reason: Retain raw Rule draft as ephemeral Watchlists-local view state under existing pane/screen ownership; no storage or sync-policy change.

- Add New/Edit tests that actually type changed fields before a changed Rules snapshot; establish settled focus/cursor/selection first. Include raw invalid threshold text and original target identity. Add a mounted screen Rules -> another section -> Rules return plus Save/Cancel/validation cases.
- Record expected value/identity/section-return failures on original code.
- Update only list presentation on list changes; retain raw form draft at the existing narrow owner and carry it across section reconstruction. Preserve condition guidance and selection/initial-seeding contracts.
- Run new tests plus existing Rules pane/scoped-rebuild/seeding tests and measured size/import/static guards; record exact evidence.
- Update docs, self-review and commit owned files; no storage/sync policy or broad workbench rewrite.

Detailed plan: Docs/superpowers/plans/2026-10-20-textual-book-followup-fixes.md (Task 4).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rules-list refresh now updates table presentation without reconstructing form controls. Raw Name, Condition, Threshold, Severity and Enabled values travel through the existing pane/screen visibility-state message and remain ephemeral at the existing owner. Section return restores New/Edit drafts; original editing identity stays independent of browsed selection. Save/Cancel retire the appropriate state and field-validation failure keeps it. Updated RulesPane, four narrow Watchlists screen methods, mounted regressions, architecture docs, exact size ratchet and the native Input replacement-key lesson.

Evidence: commit 3e2327eb3c. RED demonstrated New/Edit value loss and original-target drift after real typed preconditions. Final GREEN: 8 new keyboard/private-profile cases, 12 existing Rules cases, 2 exact size/slack guards, 1 import guard and 6 design checks pass. Pane/test Ruff and format, scoped screen comparison, UTF-8 compile/396 unchanged method ASTs and whitespace pass. Watchlists budget tightened from 14,324 (14,301 actual before fix) to exact 14,283; Settings row remains 35,732.

Superpowers independent review: spec compliant, quality Approved, no introduced Critical/Important issue. Limits retained: 5 legacy scoped/destination tests stop before pane behavior at unchanged profile bootstrap; 1 Windows-only unchanged design regex mismatches path separators after correctly detecting its injected hex. Existing screen 3 E721 and 2 formatter hunks reproduce at BASE outside changed methods. Inherited warnings remain visible, no filters or broad migration. Mounted production-screen keyboard journeys cover both section paths; no native terminal UAT or full suite.

ADR required: no; existing backlog/decisions/033-application-session-state-ownership.md, 042-watchlists-reader-first-ia.md and 161-component-pattern-library.md apply. Backend writer/eligibility is unchanged. Plan: Docs/superpowers/plans/2026-10-20-textual-book-followup-fixes.md. Added an incident lesson: Edit precondition exposed Ctrl+A moving Home; explicit replacement keys establish the typed value before testing preservation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
