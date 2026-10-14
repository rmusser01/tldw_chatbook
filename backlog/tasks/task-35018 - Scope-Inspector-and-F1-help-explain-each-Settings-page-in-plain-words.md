---
id: TASK-35018
title: Scope Inspector and F1 help explain each Settings page in plain words
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- ux
- copy
- docs
dependencies:
- TASK-33007
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Scope Inspector takes about a fifth of the Settings screen. Today it repeats the category name and the rail text. It lists raw config paths under 'Owns', and it shows developer contract rows that name Python classes and database files ('Runtime owner', 'Writes allowed', 'RawCliRuntime launch-local arm state'). On read-only pages it shows a roadmap sentence as 'Read-only'. Users cannot act on any of that, and it pushes the focused-field guide down the pane. F1 help on a page only repeats the save contract and explains none of the page's settings. This task makes the Inspector show what the focused setting does, with its config key only in one closed 'config key' disclosure. It makes F1 explain the page: what it is for, which settings it holds, how saving works, who uses it, and how to recover. It extends to every category the rule that raw config keys never appear in UI copy, which Phase 7 applied to Providers & Models. It requires PR #3057.

Plan: `backlog/docs/plan-2026-10-14-settings-plain-language-copy.md`. Findings closed: D5, A-pm-design:heuristic-2-owns-list, D-minor-7-F1, D4 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On every Settings category the Scope Inspector shows no 'Selected category', 'Runtime owner', 'Writes allowed', 'Owns' or 'Read-only' row
- [ ] #2 On Appearance and Storage, as on Providers & Models and Console Behavior, the focused setting's config key appears only inside the Inspector's closed 'config key' disclosure, and expanding it shows the key of the field that has focus
- [ ] #3 No other Inspector text on any category shows a config key, TOML section name, database file name, Python class name, raw enum value, ADR or task id, or roadmap note
- [ ] #4 On Providers & Models, Console Behavior, Appearance, Storage and RAG the Inspector body holds the focused-field guide plus that category's existing action and apply blocks, with no Affects/Recovery/Boundary block; every other category shows a plain-language Affects/Recovery/Boundary block, and the Overview boundary note stays fully visible at rest
- [ ] #5 F1 on every category says what the page is for, names the settings on it by their visible labels (per OD-1), states how saving works exactly once, and gives who uses it, its boundary, its recovery and its shortcut keys
- [ ] #6 Workspaces' recovery copy points to the page's own Create workspace action instead of Library
- [ ] #7 At 211x44 and 235x52 the whole focused-field guide for a focused Appearance field and a focused Storage field is visible in the Inspector without scrolling
- [ ] #8 '/' search returns the same top match as before for queries that match a category's owned config keys, such as 'ssl'
- [ ] #9 A plain-copy guard runs on every pull request and fails if a config key, database file name, class name, raw enum value or roadmap note is put back into Inspector, field-guide or F1 copy; it flags each verbatim string from the review
<!-- AC:END -->
