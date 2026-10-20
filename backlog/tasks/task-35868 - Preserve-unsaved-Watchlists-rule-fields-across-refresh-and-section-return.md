---
id: TASK-35868
title: Preserve unsaved Watchlists rule fields across refresh and section return
status: To Do
created_date: 2026-10-20 19:25
labels:
- watchlists
- textual-patterns
- correctness
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md
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
- [ ] #1 A changed Rules-list refresh preserves unsaved Name, Condition, Threshold, Severity and Enabled values in both New and Edit forms, with the same target rule identity.
- [ ] #2 A settled focused editable control retains its instance, focus, cursor and selection during list-only refresh; tests establish these preconditions before refreshing.
- [ ] #3 Switching Rules to another Watchlists section and back restores the unsaved new/edit draft and its original target without an implicit save.
- [ ] #4 Explicit Save and Cancel retire only the appropriate draft; validation failure retains it, and existing condition-guidance, backend eligibility and mutation behavior remain intact.
- [ ] #5 Mounted tests type real changes before both triggers, cover New/Edit and Save/Cancel, and retain the completed visibility/seeding/guidance regressions.
<!-- AC:END -->
