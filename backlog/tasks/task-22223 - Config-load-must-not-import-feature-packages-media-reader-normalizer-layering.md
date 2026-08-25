---
id: TASK-22223
title: >-
  Config load must not import feature packages (media-reader normalizer
  layering)
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 21:54'
labels:
  - architecture
  - startup
  - library
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22223).

`config.py:1802` (added 2026-08-24, `455d8d061`) imports
`Library.library_media_reader_state` inside `_load_settings_uncached`, whose comment
claims the import is lazy — but `load_settings()` runs at config-module import
(`config.py:7416`), so it fires on every boot inside the app import closure. Honest
scoping from this review: the incremental cost TODAY is small (the Library package was
already in the pin's import closure via `app.py:238`), so this is a layering finding, not
a milliseconds finding — config load acquiring feature-package imports is the mechanism by
which the next 100 ms lands silently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preference normalization happens without config.py importing the Library package (move the normalizer to a config-safe leaf module, or normalize at first read in the Library layer)
- [ ] #2 A comment or guard at the site prevents the next feature from repeating the pattern
- [ ] #3 No behavior change to media-reader preference handling
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the escalated circular import at base: solo-run Tests/Character_Chat/test_character_persona_scope_service.py (collection ImportError through config.py:1802 -> Library.__init__ -> Sync_Interop -> Chat -> runtime_policy.bootstrap partially initialized).\n2. Red-first guard: new Tests/Packaging/test_config_import_closure.py (subprocess-isolated, family pattern) asserting import tldw_chatbook.config leaves tldw_chatbook.Library out of sys.modules -> RED at base.\n3. Fix shape (a): git mv Library/library_adaptive_reader_state.py (pure stdlib leaf: dataclasses+typing only) -> Utils/adaptive_reader_state.py (Utils/__init__.py is empty, config-safe). Update all importers (config.py, library_media_reader_state.py, library_screen.py, settings_appearance_defaults.py, library_adaptive_reader_shell.py + 2 test files). Single source of truth, no duplication.\n4. config.py: import the normalizer at module top from the Utils leaf; drop the pretend-lazy site import; leave a layering comment naming the guard test.\n5. Verify: solo repro collects and passes; guard green; targeted suites (Library reader state, media reader, adaptive shell, settings appearance, config, Packaging closures, import-weight) + full --collect-only sweep; preflight.\n6. Mutation test: restore the config->Library import; guard AND solo repro must red; revert.\n7. Update task description honestly (layering finding escalated to live circular import), tick ACs, notes, Done.
<!-- SECTION:PLAN:END -->
