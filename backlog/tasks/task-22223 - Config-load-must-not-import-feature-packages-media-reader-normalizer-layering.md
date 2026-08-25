---
id: TASK-22223
title: >-
  Config load must not import feature packages (media-reader normalizer
  layering)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 22:08'
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

**Escalation (2026-08-25, found by TASK-22217's implementer, confirmed at
implementation base `76f130138`):** this is no longer only a layering note — the import
participates in a LIVE circular import. By implementation time the site had become
`Library.library_adaptive_reader_state` (the media-reader state was generalized to the
adaptive reader), and the cycle is: any module that imports `runtime_policy.bootstrap`
before config (e.g. `Character_Chat/server_character_persona_service.py`) →
`bootstrap:12 import config` → config's module-scope `load_settings()` →
`config.py:1802 import Library.…` → `Library/__init__` → `library_collections_service`
→ `library_collections_state` → `Sync_Interop/__init__` → `Chat/__init__` →
`server_chat_conversation_service` → `from runtime_policy.bootstrap import …` →
`ImportError: cannot import name … from partially initialized module`.
`Tests/Character_Chat/test_character_persona_scope_service.py` run SOLO could not even
be collected at base (it passed only in orderings where something imported config
first). Measured cost: the one import put 66 feature modules (Library 7, Chat 10,
Sync_Interop 29, Skills_Interop 10, runtime_policy 6, Notes 2, DB +2) into EVERY
`import tldw_chatbook.config` — 106 tldw modules before, 40 after the fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preference normalization happens without config.py importing the Library package (move the normalizer to a config-safe leaf module, or normalize at first read in the Library layer)
- [x] #2 A comment or guard at the site prevents the next feature from repeating the pattern
- [x] #3 No behavior change to media-reader preference handling
- [x] #4 The escalated circular import is fixed: `Tests/Character_Chat/test_character_persona_scope_service.py` collects and runs when invoked solo (its one remaining failure, `test_app_wires_character_persona_services`, is baselined as a pre-existing dev red with an unrelated cause — a stale test double passing `object()` where `ActorPackActivationService` requires a `CharactersRAGDB`)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the escalated circular import at base: solo-run Tests/Character_Chat/test_character_persona_scope_service.py (collection ImportError through config.py:1802 -> Library.__init__ -> Sync_Interop -> Chat -> runtime_policy.bootstrap partially initialized).\n2. Red-first guard: new Tests/Packaging/test_config_import_closure.py (subprocess-isolated, family pattern) asserting import tldw_chatbook.config leaves tldw_chatbook.Library out of sys.modules -> RED at base.\n3. Fix shape (a): git mv Library/library_adaptive_reader_state.py (pure stdlib leaf: dataclasses+typing only) -> Utils/adaptive_reader_state.py (Utils/__init__.py is empty, config-safe). Update all importers (config.py, library_media_reader_state.py, library_screen.py, settings_appearance_defaults.py, library_adaptive_reader_shell.py + 2 test files). Single source of truth, no duplication.\n4. config.py: import the normalizer at module top from the Utils leaf; drop the pretend-lazy site import; leave a layering comment naming the guard test.\n5. Verify: solo repro collects and passes; guard green; targeted suites (Library reader state, media reader, adaptive shell, settings appearance, config, Packaging closures, import-weight) + full --collect-only sweep; preflight.\n6. Mutation test: restore the config->Library import; guard AND solo repro must red; revert.\n7. Update task description honestly (layering finding escalated to live circular import), tick ACs, notes, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shape (a) from the finding, made trivial by a census: the normalizer module
`Library/library_adaptive_reader_state.py` was ALREADY a pure leaf (its only imports are
`dataclasses` and `typing`); the entire cost was the `Library` package `__init__`
executing on the way in. So the whole module moved via `git mv` to
`tldw_chatbook/Utils/adaptive_reader_state.py` (`Utils/__init__.py` is empty — genuinely
config-safe), and every consumer now imports the one leaf: config.py (module-top import,
replacing the pretend-lazy function-scope import), `Library/library_media_reader_state.py`
(keeps its compat re-exports, so `normalize_media_reader_preferences` is still the same
object), `UI/Screens/library_screen.py`, `UI/Screens/settings_appearance_defaults.py`,
`Widgets/Library/library_adaptive_reader_shell.py`, and the two test files. No function
body changed — AC #3 is a move-only diff plus import paths (behavior identity also covered
by the untouched normalization assertions in `Tests/Library/test_library_adaptive_reader_state.py`
and `Tests/test_config_library_defaults.py`, all green).

Guard (AC #2): new subprocess-isolated `Tests/Packaging/test_config_import_closure.py`
(the `*_import_closure.py` family pattern) — bare `import tldw_chatbook.config` must leave
`Library`, `Chat`, `Sync_Interop`, `Skills_Interop`, and `runtime_policy` out of
`sys.modules`, with an anti-vacuity assertion that the Utils leaf IS resident (the
normalizer still runs at config load). Plus a layering comment at the config.py site and
in the leaf's docstring naming the guard.

Evidence:
- Red-first: the guard FAILED at base; the solo persona-scope run could not collect at
  base (circular ImportError through `runtime_policy.bootstrap`).
- After: guard passes; the solo run collects, 53/54 pass; the 1 failure
  (`test_app_wires_character_persona_services`) reproduced IDENTICALLY at base
  `76f130138` in a `Tests/Character_Chat/` directory run (988 passed / 1 failed) — a
  pre-existing stale test double (`fake_app.chachanotes_db = object()` rejected by
  `ActorPackActivationService`'s isinstance gate, `Actor_Packs/activation.py:116`),
  unrelated to this change and not fixed here.
- Targeted: 17 config suites + Tests/Character_Chat + import-weight guard: 1198 passed /
  1 failed (that same baselined red). Reader-state/shell/flow/settings/packaging batch:
  214 passed; the 2 fails + 62 errors are the venv's missing build backend
  (`setuptools.build_meta`, `python -m build`) and reproduced identically at base; one
  pointer-click flake (`test_grips_emit_correct_toggle…`) passed solo and in-file rerun.
- `--collect-only` sweep: 59,445 collected / 28 errors (all missing optional deps:
  numpy ×21, playwright ×3, + cascades); base: 59,444 / the SAME 28.
- Mutation: re-adding a config→Library import (via `library_media_reader_state`) reds the
  guard AND kills solo collection of the persona-scope module; reverted.
- Closure measurement: `import tldw_chatbook.config` = 106 tldw modules before → 40
  after (−66), ~0.12 s.
- `./scripts/preflight.sh`: all checks green.
- Lesson added to `backlog/docs/lessons-testing-evidence.md`: a function-scope import is
  not lazy if the function runs at module import; deferral claims need a fresh-interpreter
  `sys.modules` probe, and a solo-only collection failure means an import cycle whose
  direction depends on who imports first.

Out of scope, worth follow-ups: (1) `Library/__init__.py` eagerly imports the
collections/tool service stack, so ANY `tldw_chatbook.Library.*` import pays ~66 modules
and re-arms the cycle risk for the next config-adjacent consumer — a PEP 562 lazy
`__init__` (the `TTS/__init__` precedent) would kill the class; (2) the stale
`test_app_wires_character_persona_services` double is red on dev today (also in-suite) —
tests-that-cannot-collect-solo masked nothing here, but the double predates the
ActorPackActivationService wiring and needs a real `CharactersRAGDB` fake.
<!-- SECTION:NOTES:END -->
