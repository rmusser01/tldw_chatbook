---
id: TASK-39057
title: "The Persona_Visual pixel-migu installer has had no production caller since 11150b849d, yet its tests and the packaging probe still exercise it instead of the Buddy library's seeding"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 17:59
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
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on Architecture PR #3436 (head b7c8e809b3), which lands TASK-38990. TASK-38990's notes list it as a follow-up. Line numbers are dev's; ones marked #3436 are from that head.

**No production caller.**
- An untruncated `git grep` on dev finds `ensure_builtin_pixel_migu_buddy` in `tldw_chatbook/` only at its own definition (`Persona_Visual/builtin_pixel_migu.py:27`).
- 11150b849d (2026-09-08, "Add independent Buddy management…") removed the `app.py` import and call.
- Production now seeds the Buddy through `Persona_Buddy/library.py:400` (`ensure_builtin`, key `builtin:pixel-migu`, `:412`).
- `Character_Chat/builtin_pixel_migu.py`, a different module, is still live (`config.py:8468-8472`).

**Its only callers are tests.**
- `Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py` imports it (`:12`, `:139`, `:291`, `:326`, `:360`, `:386`); it passes on dev, 21 passed at `-n 4`.
- `Tests/Packaging/test_pixel_migu_distribution.py:44` calls it inside the installed-wheel probe (`:51`, `:73`). So that packaging test proves a path production does not run.
- TASK-38990 pins the import as a reviewed exception, with a liveness pin (#3436 `Tests/Architecture/test_persona_visual_runtime_boundary.py:36-41`).
- Neither test file runs on a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner decides whether the module is deleted or relocated (for example as a test helper); the decision is recorded
- [x] #2 The packaging probe proves the production seeding path (`Persona_Buddy.library` `ensure_builtin`) from both built artifacts, and is shown red against a wheel that omits the Buddy's packaged assets (or another recorded negative control)
- [x] #3 The boundary exception row is removed, or kept with its reason updated, and the liveness pin is green
- [x] #4 An untruncated grep shows no remaining caller of any removed symbol
<!-- AC:END -->

## Implementation Plan

Owner decision (2026-11-06): delete the module and its reviewed boundary exception; retarget the packaging probe and the Buddy tests to the live `Persona_Buddy.library` seeding path.

1. Delete `tldw_chatbook/Persona_Visual/builtin_pixel_migu.py`. `Character_Chat/builtin_pixel_migu.py` is a different, live module and stays.
2. Drop both rows from `REVIEWED_BOUNDARY_EXCEPTIONS` in `Tests/Architecture/test_persona_visual_runtime_boundary.py`; keep the empty table and its liveness pin.
3. Rewrite the installed-wheel probe in `Tests/Packaging/test_pixel_migu_distribution.py` to seed through `BuddyLibrary.ensure_builtin` (the call `app_service_wiring` makes) and resolve every Buddy state from both artifacts. Negative control: the probe fails against a wheel without one Buddy frame.
4. Retarget `Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py` to `BuddyLibrary.ensure_builtin` and the app readiness path. Installer-only cases with no library equivalent go, and are listed in the notes.
5. Untruncated `git grep` for every removed symbol.

## Implementation Notes

**Status: Done.**

Owner decision (2026-11-06): "Delete it: remove the module and its reviewed boundary exception; retarget the packaging probe and the Buddy tests to the live Persona_Buddy.library seeding path so they prove what production runs." (AC#1)

**Changes.**
- Deleted `tldw_chatbook/Persona_Visual/builtin_pixel_migu.py`. `Character_Chat/builtin_pixel_migu.py`, the live character seeder, is untouched.
- AC#3: `REVIEWED_BOUNDARY_EXCEPTIONS` in `Tests/Architecture/test_persona_visual_runtime_boundary.py` is now an empty, typed table, and its comment says why. The liveness pin stays and is green. That file, the profile-path inventory and the Persona Buddy boundary ran 24 passed, serial.
- AC#2: `INSTALLED_PROBE` in `Tests/Packaging/test_pixel_migu_distribution.py` now seeds through `BuddyLibrary(db, root).ensure_builtin()`, the call `app_service_wiring`'s readiness pass makes. It still checks the 18 character expressions, which it now finds through `find_builtin_pixel_migu_character` because no installer Persona exists any more. It then asserts:
  - the Buddy's name, `source_key="builtin:pixel-migu"`, and that it has no Persona;
  - 64 assets;
  - all 9 Buddy states, with and without reduced motion, through `resolve_preview`;
  - after a restart, the same record, one listed Buddy, and the same graph identity.
- `Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py` now drives `BuddyLibrary.ensure_builtin` and `TldwCli.ensure_actor_pack_recovery`. It has 14 nodes:
  - a fresh install with every state;
  - a restart that does no resource reads;
  - a failed publication that leaves no owner row and then retries;
  - a post-commit interruption (KeyboardInterrupt, SystemExit, RuntimeError) that keeps the committed owner's id;
  - app readiness installing the Buddy;
  - a retired legacy Persona (inactive, deleted, archived) blocking the Buddy. The legacy state is now built with the Persona service and `publish_persona_visual`, as an older release left it;
  - reading resources from a ZIP traversable;
  - canonical manifest validation.
- Dropped installer-only cases: Persona JSON edits and tombstones, coordinator rollback, and the racing-service loser. These tested the deleted module's own mechanics. Also dropped: non-flat asset filenames and a symlinked resource leaf. `BuddyLibrary.ensure_builtin` has no equivalent check (see Proposed follow-ups in the report).

**Evidence.**
- Packaging: `Tests/Packaging/test_pixel_migu_distribution.py` 7 passed in 119.9 s, serial, with both `test_installed_pixel_migu_fresh_profile[wheel]` and `[sdist-wheel]` green.
- Packaging negative control (temporary test edit, reverted): the wheel case installed a wheel built without `assets/persona_visual/pixel_migu/speaking-4.png`. The probe failed inside the installed `tldw_chatbook/Persona_Buddy/library.py`, line 437, in `ensure_builtin`, reading the missing frame (`assert 1 == 0` on the probe's exit code).
- The 14 retargeted nodes pass (`-n 4`). Each was shown red under a temporary edit to the seeding path, each reverted:
  - library ignores the existing record, skips owner cleanup, reads the bundle as a plain path → the failed-publication, restart, ZIP and three manifest nodes are red;
  - library ignores `legacy_retired` → the three retired-legacy nodes are red;
  - readiness skips `ensure_builtin` → the readiness node is red;
  - library deletes the owner on any failure → the three interruption nodes are red;
  - the source key changes → the fresh-install node is red.
- `Tests/Persona_Visual`, `Tests/Persona_Buddy` and `Tests/Character_Chat/test_builtin_pixel_migu.py` at `-n 4`, load 42: 825 passed, 1 skipped, 48 failed. The base run had the identical 48 failures (the two lists `diff` empty), for example `find_spec('tldw_chatbook.Persona_Visual.buddy') is None` and `persona_visual_authority_changed`. None comes from this change.
- AC#4, untruncated `git grep -F` over the whole repository: `ensure_builtin_pixel_migu_buddy`, `PIXEL_MIGU_PERSONA_ID`, `Persona_Visual.builtin_pixel_migu` and `Persona_Visual import builtin_pixel_migu` appear only in frozen QA evidence (`Docs/QA`, `Docs/superpowers/qa`), one 2026-09-10 review note and backlog task records. No code or test references them.

Files: `tldw_chatbook/Persona_Visual/builtin_pixel_migu.py` (deleted), `Tests/Architecture/test_persona_visual_runtime_boundary.py`, `Tests/Packaging/test_pixel_migu_distribution.py`, `Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py`.
