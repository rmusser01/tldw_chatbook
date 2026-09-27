---
id: TASK-33011
title: Break app.py down so it shrinks far below its size ratchet
status: To Do
created_date: 2026-09-27 10:40
dependencies:
- TASK-32954
labels:
- architecture
- size-governance
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/app.py` is about 21,200 lines and has no room left under its size ratchet: on dev at 74965f694f it measured exactly its 21,176-line budget. Because of that, every feature that needs a few lines of app-level wiring turns the ratchet red. TASK-32954 (PR #2842, merged by owner decision with the ratchet red) needed a Console to Personas event forwarder and a built-in-skills config loader, which took the file to 21,215 lines, 39 over budget. Holding the line one PR at a time does not work: the file needs to shrink a lot, so that app-level wiring has somewhere to live and `TldwCli` becomes a small composition root rather than a 21k-line class.

`backlog/docs/size-decomposition-candidates-2026-09-18.md` already maps the first extractions:
- `LibraryIngestQueueMixin`: about 4,500 lines, already a mixin.
- The 10 command-palette providers: about 1,050 lines.
- The `_wire_*` service composition: about 1,950 lines.

That map is a starting point, not the whole target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `tldw_chatbook/app.py` is under 8,000 lines, and its row in `Tests/Architecture/test_module_size_ratchet.py` is lowered to the measured size, so the reduction cannot silently grow back.
- [ ] #2 `LibraryIngestQueueMixin`, the command-palette providers and the service-composition (`_wire_*` / lazy service builders) code live in their own modules. Each old import path that tests or other modules use still resolves, or is updated.
- [ ] #3 App-level message forwarders (for example `on_model_catalog_refreshed` and `on_character_card_changed`) and similar per-feature glue live outside `app.py`, so a new feature's wiring no longer has to grow `app.py`.
- [ ] #4 The UI-ready module census (`Tests/Performance/test_ui_ready_module_census.py`) and the app import-weight tests stay within their ratchets: extraction does not load more modules at boot.
- [ ] #5 No behaviour change. The full test suite's failure set matches dev's by name, compared with the ADR-126 recovery-gate method, and the app boots and passes a live smoke check.
- [ ] #6 The work lands as a series of independently mergeable PRs, one extraction cluster each, each with the field-ownership check from the decomposition doc's §2 recipe.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Context from TASK-32954 (PR #2842): the same PR also added lines to three files that dev already has over their ratchets. That debt belongs to the equivalent Personas, Console and Library decompositions, not to this task:
- `personas_screen.py`: +84
- `console_chat_controller.py`: +63
- `library_screen.py`: +13
<!-- SECTION:NOTES:END -->
