---
id: TASK-33002
title: 'Model config P2: one field table and honest copy'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-2
  - console
  - settings
  - ux
dependencies:
  - TASK-33001
references:
  - 'backlog/docs/spec-2026-09-26-model-config-redesign.md'
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/report.md'
  - 'backlog/decisions/033-settings-commit-models-three-honestly-labeled.md'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'Tests/Architecture/test_module_size_ratchet.py'
  - 'Tests/Architecture/test_screen_size_ratchet.py'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 of the model-configuration redesign (backlog/docs/spec-2026-09-26-model-config-redesign.md §8; qa/model-config-ux-review-2026-09-26/judge-synthesis.md §4). Ships as one PR. It changes copy and labels only; density and layout come later. It lands after phase 1, for two reasons:
- The field table carries phase 1's field-to-request-key definition.
- The scope copy describes the D1 convergence that phase 1 ships. Before that, the copy would be untrue.

What it closes:
- C1(a): Providers & Models save copy has no scope ('Provider settings saved.' at UI/Screens/settings_screen.py:30215, a toast at :30242-30244, and 'Shared with Console' at :9455).
- C3: the Provider Test result is one ' | '-joined dump of verdict prose and config-key spellings (settings_screen.py:15213-15335, joined at :15325/:15332).
- C7(d): chips print raw provider keys (UI/Screens/chat_screen.py:9826-9831, Chat/console_display_state.py:776), and several shipped keys have no display name (config.py:4120-4151).
- Label drift across four editors, for example 'Think budget' (settings_screen.py:17378, :18695) against 'Budget' (Widgets/Console/console_settings_modal.py:2125).

It keeps ADR-033's State badge and adds the unsaved count. Absorbs TASK-486 (custom-named credential query parameters in Test evidence). TASK-194 (popover display names) stays open: per the spec it closes when the popover rows are rebuilt.

Constraints:
- console_settings_modal.py has zero headroom (module-size ratchet 7,807).
- chat_screen.py must stay within its 25,363-line budget.
- ADR-097 ratchets never rise.
- ADR-066 legacy aliases stay selectable.
- Test copy for cloud providers stays a local readiness check (TASK-30011 AC#2). Changing what Test checks is out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every model-configuration field has one label and one help line wherever it is edited
- [ ] #2 The Provider Test result reads as labelled rows that lead with the outcome, with no config-key spellings and no leaked secrets
- [ ] #3 The Providers & Models save and the State line say what a save applies to
- [ ] #4 The Settings State line keeps naming its save model and counts unsaved edits
- [ ] #5 Console chips and notices name providers by display name, and one display-name map serves every surface
- [ ] #6 No Settings copy refers to 'Console Defaults' or 'Override current Console model'
- [ ] #7 Rendered captures at 211x44, plus one at 235x52, of every changed surface are attached to the PR
- [ ] #8 The phase changes no layout, CSS, design token or geometry
- [ ] #9 console_settings_modal.py does not grow, chat_screen.py stays within its budget, and no ADR-097 ratchet value rises
- [ ] #10 TASK-486 is closed as Done
- [ ] #11 Docs/User_Guide pages updated (settings.md, console.md), including their Verified-against stamps
- [ ] #12 ./scripts/preflight.sh passes
<!-- AC:END -->
