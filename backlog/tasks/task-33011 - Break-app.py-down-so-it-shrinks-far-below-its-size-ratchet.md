---
id: TASK-33011
title: Break app.py down so it shrinks far below its size ratchet
status: Done
created_date: 2026-09-27 10:40
updated_date: 2026-09-29 13:30
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
- [x] #1 `tldw_chatbook/app.py` is under 8,000 lines, and its row in `Tests/Architecture/test_module_size_ratchet.py` is lowered to the measured size, so the reduction cannot silently grow back.
- [x] #2 `LibraryIngestQueueMixin`, the command-palette providers and the service-composition (`_wire_*` / lazy service builders) code live in their own modules. Each old import path that tests or other modules use still resolves, or is updated.
- [x] #3 App-level message forwarders (for example `on_model_catalog_refreshed` and `on_character_card_changed`) and similar per-feature glue live outside `app.py`, so a new feature's wiring no longer has to grow `app.py`.
- [x] #4 The UI-ready module census (`Tests/Performance/test_ui_ready_module_census.py`) and the app import-weight tests stay within their ratchets: extraction does not load more modules at boot.
- [x] #5 No behaviour change. The full test suite's failure set matches dev's by name, compared with the ADR-126 recovery-gate method, and the app boots and passes a live smoke check.
- [x] #6 The work lands as a series of independently mergeable PRs, one extraction cluster each, each with the field-ownership check from the decomposition doc's §2 recipe.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Context from TASK-32954 (PR #2842): the same PR also added lines to three files that dev already has over their ratchets. That debt belongs to the equivalent Personas, Console and Library decompositions, not to this task:
- `personas_screen.py`: +84
- `console_chat_controller.py`: +63
- `library_screen.py`: +13

**Outcome (2026-09-29).**

Shrank `tldw_chatbook/app.py` from 21,234 to 5,712 lines in nine PRs, one extraction cluster each, each moving code verbatim:

| PR | Extraction | app.py after |
|---|---|---|
| #2859 (A) | entry tail -> `app_entry.py` (lazy `__getattr__` re-export) | 20,484 |
| #2865 (B) | destination handlers -> `app_destinations.py` (lazy function module, `TldwCli` stubs) | 19,682 |
| #2881 (C) | `LibraryIngestQueueMixin` -> `app_ingest_queue.py` | 14,930 |
| #2891 (D) | `_wire_*` / lazy service builders -> `ServiceWiringMixin` (`app_service_wiring.py`) | 11,506 |
| #2893 (E) | speech handlers, owners and admission -> `app_speech.py` (lazy, stubs) | 10,601 |
| #2895 (F) | lifecycle, shutdown and quit -> `LifecycleMixin` (`app_lifecycle.py`) | 8,524 |
| #2898 (G) | screen navigation -> `NavigationMixin` (`app_navigation.py`) | 7,467 |
| #2900 (H) | the 10 command-palette providers and key helpers -> `app_command_providers.py` (re-exported) | 6,403 |
| #2901 (I) | per-feature glue (reminders, FTS backfills, server parity, persona-buddy forwarders, `on_character_card_changed`, model catalog) -> `FeatureGlueMixin` (`app_feature_glue.py`) | 5,712 |

Approach and decisions:
- PR-A to PR-E used lazy function modules with `TldwCli` stubs to keep the UI-ready census down (1033 -> 1026). From PR-F on, census headroom allowed eager mixins: no stubs, and `inspect.getsource` pins and unbound `TldwCli.<m>(fake)` calls resolve through the MRO. The census ends at 1030, below the 1033 it started at; ADR-097 needed no exception.
- `@on`-decorated handlers stay on `TldwCli`, because Textual dispatches decorated handlers only from Textual classes; name-based `on_*` handlers moved. The mixins precede `App` in the bases.
- Moved code reads its new module's globals, so a test that patches `tldw_chatbook.app.X` silently misses it. `Tests/Architecture/test_app_extracted_patch_targets.py` fails on such a patch, including inside f-string subprocess scripts. `Tests/app_module_patches.py` (`patch_app_global` / `set_app_global`) patches every app module that binds the name, reading bindings from source so lazy modules stay unloaded.
- Field ownership is unchanged (decomposition doc §2): mixin methods run on the same `TldwCli` instance, so every attribute keeps its single owner.
- Each PR lowered app.py's `test_module_size_ratchet.py` row to the measured size, gave the new module its own row, and re-pinned the diagnostic inventory (statements moved, none rewritten).

Verification (AC#5):
- Full suite on the stack tip vs its dev merge-base, compared by failure name (ADR-126 method): 27 branch-only names. Re-run in isolation, 3 were real, all source-reading tests outside the PR gate, and PR-F commit 220b75569c fixed them: terminal privacy `TERMINAL_RUNTIME_OWNERS`, TTS `on_unmount` location, and the server-client migration audit (red on dev since #2891). The rest failed on dev too or were flaky.
- Caveat: locally about 20k tests fail in both trees with `RecoveryRequired: raw_source_selection_changed` (an ADR-126 local-only failure that persists with a fresh scratch profile; CI passes them), which could mask a regression. As a narrower check, the 196 test files that read app.py source were run on dev tip and on the stack tip: identical failure sets (3,034 each, same names).
- Live smoke on the full stack under a scratch profile: boot, palette navigation, Ctrl+Q quit.

Lesson: an extraction PR must grep `Tests/` for the moved file's path string and run those files. The per-PR gate does not run them, which is how #2891 left the migration-audit test red on dev.
<!-- SECTION:NOTES:END -->
