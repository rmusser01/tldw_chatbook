---
id: TASK-32500
title: 'Unshadow shell nav hotkeys on Roleplay: mode chips ctrl+1-4 to c/p/d/l'
status: Done
assignee: []
created_date: '2026-09-12 06:55'
updated_date: '2026-09-12 08:01'
labels:
  - roleplay
  - nav
  - keybindings
dependencies: []
---

## Renumbering provenance

Renumbered from TASK-32480 on 2026-09-12: the id collided with "Tokenize
features/_chat.tcss end-to-end as the migration reference", which arrived on
dev while this branch was in review, so the older arrival keeps it (owner
rule TASK-19601 — the fast lane's backlog-id uniqueness check caught it).

Renumbered again from TASK-32494 to TASK-32500 on 2026-09-12: a second
collision — "Cross-platform Console reply speech playback" arrived on dev
(commit 97041ab370, renumbering the voice tasks to 32494-32496) while this
branch was in review, so the older arrival keeps it (owner rule TASK-19601).
Jumped to 32500 with a small buffer because dev's task-id stream is moving
faster than this PR's review cycle.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shell destination hotkey layer (ADR-031 task-32458 refinement: ctrl+1..0, then f2/f3/f4/f5/f7) is app-global, and every nav label promises its key. PersonasScreen (Roleplay) binds ctrl+1..ctrl+4 for its mode strip, shadowing ctrl+1 Home / ctrl+2 Console / ctrl+3 Library on that screen — the nav bar's labels lie there, and the user guide has to carry a standing caveat ('Ctrl+2 didn't take me to Console' is a published troubleshooting entry). The task-32458 refinement recorded this as a deliberate destination-local carve-out ('harmless since the user is already there') — but that only covers Roleplay's own ctrl+4 slot; ctrl+1/2/3 still navigate away in every other context and mislead here. Move mode switching to ADR-031-rule-3 single letters (c/p/d/l, mirroring the mode strip order), keeping [ ] cycling, so no screen binds a key the nav layer owns.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Roleplay mode keys are c/p/d/l zipped against MODE_CHIP_ORDER; no ctrl+digit binding remains on PersonasScreen,Footer hint and chip tooltips teach the new letters; ctrl+1-4 mode hint removed,Guard test asserts no BaseAppScreen subclass binds any shell destination hotkey,Tests and User Guide pages updated (index exception block, roleplay pages' caveats, troubleshooting entry),Targeted personas + nav suites green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ADR-152 amending ADR-031 (narrow the task-32458 Roleplay carve-out); ADR-031 refinement sentence updated to point at it
2. personas_screen.py: MODE_HOTKEYS tuple + BINDINGS zip + chip tooltip + footer hint + action docstring
3. Guard test in test_master_shell_navigation.py (no BaseAppScreen binds a shell destination hotkey)
4. test_personas_workbench.py: footer-hint assertion + test_mode_keys_switch_modes ctrl+2 -> p
5. User Guide: index.md exception block + table row; roleplay-chat-dictionaries.md note + troubleshooting; characters-and-personas.md, chat-dictionaries.md, lore-books.md
6. Targeted suites + headless smoke; mark Done
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported the Roleplay mode-key unshadowing onto dev (reversing ADR-031's task-32458 destination-local carve-out via new ADR-152).

Implementation: personas_screen.py gains MODE_HOTKEYS = (c, p, d, l) zipped positionally against MODE_CHIP_ORDER in BINDINGS (replacing the ctrl+{index+1} loop), chip tooltips now read '… (c)' etc., the footer hint is 'c/p/d/l mode', and action_personas_mode's docstring updated. Guard test test_no_screen_shadows_shell_destination_hotkeys (test_master_shell_navigation.py) walk_packages-imports every UI.Screens module and asserts no BaseAppScreen subclass binds any SHELL_DESTINATION_SHORTCUTS key — passes clean, confirming PersonasScreen was the only offender. Tests updated: footer-hint assertion (ctrl+1-4 -> c/p/d/l), chip tooltip expectations, test_mode_keys_switch_modes now presses 'p' with a search-field focus guard.

Verification: guard test green; test_master_shell_navigation.py 48 passed; personas TestWorkbenchShell+TestKeyboardInteraction+TestPersonasMode 48 passed with 1 failure (test_resize_sync_skips_work_when_compact_state_is_unchanged) confirmed pre-existing via stash-and-rerun on pristine dev; full-app headless smoke on dev: ctrl+4 -> PersonasScreen, p -> personas mode, c -> characters, ctrl+2 -> ChatScreen (the unshadowed navigation), footer renders c/p/d/l; ruff finding count identical to pristine dev on all touched files; screen_navigation nav/persona-filtered run has 38-40 flaky failures that are ALL pre-existing on pristine dev (diffed failure lists: zero new failures introduced).

Docs: User Guide index exception block and hotkey-table exception removed; roleplay-chat-dictionaries.md mode note + keyboard table + two troubleshooting entries rewritten; characters-and-personas.md, chat-dictionaries.md, lore-books.md mode-key teachings updated to c/p/d/l.

Files: tldw_chatbook/UI/Screens/personas_screen.py; Tests/UI/{test_master_shell_navigation,test_personas_workbench}.py; backlog/decisions/{031 (refinement narrowed), 152 (new)}; Docs/User_Guide/{index,roleplay-chat-dictionaries}.md + roleplay-chat-dictionaries/{characters-and-personas,chat-dictionaries,lore-books}.md. ADR: ADR-152 created (amends ADR-031).
<!-- SECTION:NOTES:END -->
