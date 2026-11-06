---
id: TASK-39057
title: "The Persona_Visual pixel-migu installer has had no production caller since 11150b849d, yet its tests and the packaging probe still exercise it instead of the Buddy library's seeding"
status: To Do
created_date: 2026-11-06 10:09
labels:
- follow-up
- persona-visual
- cleanup
- needs-owner-decision
priority: low
references:
- tldw_chatbook/Persona_Visual/builtin_pixel_migu.py
- tldw_chatbook/Persona_Buddy/library.py
- Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py
- Tests/Packaging/test_pixel_migu_distribution.py
- Tests/Architecture/test_persona_visual_runtime_boundary.py
dependencies:
- TASK-38990
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the Architecture branch that lands TASK-38990 (the state after it merges). TASK-38990's notes list it as a follow-up.

**No production caller.**
- An untruncated `git grep` on dev finds `ensure_builtin_pixel_migu_buddy` in `tldw_chatbook/` only at its own definition (`Persona_Visual/builtin_pixel_migu.py:27`).
- 11150b849d (2026-09-08, "Add independent Buddy management…") removed the `app.py` import and call.
- Production now seeds the Buddy through `Persona_Buddy/library.py:400` (`ensure_builtin`, key `builtin:pixel-migu`, `:412`).
- `Character_Chat/builtin_pixel_migu.py`, a different module, is still live (`config.py:8468-8472`).

**Its only callers are tests.**
- `Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py` imports it (`:12`, `:139`, `:291`, `:326`, `:360`, `:386`); it passes on dev, 21 passed at `-n 4`.
- `Tests/Packaging/test_pixel_migu_distribution.py:44` calls it inside the installed-wheel probe (`:51`, `:73`). So that packaging test proves a path production does not run.
- TASK-38990 pins the import as a reviewed exception, with a liveness pin (`Tests/Architecture/test_persona_visual_runtime_boundary.py:36-41`, branch).
- Neither test file runs on a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides whether the module is deleted or relocated (for example as a test helper); the decision is recorded
- [ ] #2 The packaging probe proves the production seeding path (`Persona_Buddy.library` `ensure_builtin`) from both built artifacts, and is shown red against a wheel that omits the Buddy's packaged assets (or another recorded negative control)
- [ ] #3 The boundary exception row is removed, or kept with its reason updated, and the liveness pin is green
- [ ] #4 An untruncated grep shows no remaining caller of any removed symbol
<!-- AC:END -->
