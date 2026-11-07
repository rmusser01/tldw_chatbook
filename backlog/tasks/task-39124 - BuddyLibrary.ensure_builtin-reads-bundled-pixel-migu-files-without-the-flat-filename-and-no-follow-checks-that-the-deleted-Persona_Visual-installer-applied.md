---
id: TASK-39124
title: "BuddyLibrary.ensure_builtin reads bundled pixel-migu files without the flat-filename and no-follow checks that the deleted Persona_Visual installer applied"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 07:07
labels:
- follow-up
- security
- persona-buddy
priority: low
references:
- tldw_chatbook/Persona_Buddy/library.py
- tldw_chatbook/Persona_Visual/publication.py
- Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py
- Tests/Packaging/test_pixel_migu_distribution.py
dependencies:
- TASK-39057
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-arch, which lands TASK-39057. TASK-39057's notes list it as a follow-up. Line numbers are dev's.

**Why the check moved.** Owner decision (2026-11-06), quoted from TASK-39057's notes: "Delete it: remove the module and its reviewed boundary exception; retarget the packaging probe and the Buddy tests to the live Persona_Buddy.library seeding path so they prove what production runs."

**What the live path does not check.**
- `BuddyLibrary.ensure_builtin` reads each bundled asset with `source.joinpath(filename).read_bytes()`, taking `filename` straight from the bundled `assets.json` (`Persona_Buddy/library.py:421-438`; unchanged on fu/r11-arch).
- The deleted installer (dev `Persona_Visual/builtin_pixel_migu.py:135-148`) did check:
  - a flat filename (`validate_path`, and `publication_path.name != filename`);
  - a read through `load_persona_visual_asset`, which does not follow symlinks.
- TASK-39057 dropped that installer's tests for these checks (`test_buddy_seed_rejects_non_flat_asset_paths`, `test_buddy_seed_rejects_symlinked_resource_leaf`).

**Probe.** On dev, with a copied bundle and publication stubbed, `ensure_builtin` read and passed on `../escape.png`, an absolute path, `nested/escape.png` and a symlinked leaf.

**Why it is defence in depth.** Publication still pins each asset's byte count and sha256 against the bundle (`Persona_Visual/publication.py:888`, `:938`), so only matching bytes can publish. What is left is a read that can leave the bundle directory and follow symlinks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `ensure_builtin` refuses a non-flat `assets.json` filename (`../escape.png`, an absolute path, `nested/escape.png`) before reading it, and publishes nothing; tests are red before the change
- [x] #2 `ensure_builtin` refuses a symlinked resource leaf (skipped only where symlinks cannot be created); red before the change
- [x] #3 The ZIP-traversable seeding case and the installed-wheel packaging probe stay green
- [ ] #4 The new tests run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

1. Restore the deleted installer's two tests against `BuddyLibrary.ensure_builtin`: a non-flat `assets.json` filename (`../escape.png`, an absolute path, `nested/escape.png`) and a symlinked resource leaf are refused before any read or publication, and leave no Buddy. Show them red on the current seeding path.
2. In `ensure_builtin`, refuse any filename that is not a single flat name, and read each asset through `importlib.resources.as_file` + `load_persona_visual_asset` (no-follow leaf, byte count and sha256 checked), as the deleted installer did (`git show 0909623ad9^:tldw_chatbook/Persona_Visual/builtin_pixel_migu.py`).
3. Keep the ZIP-traversable seeding case and the installed-wheel packaging probe green.
4. No laned file covers Buddy seeding; list the file for the lanes stream.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** `BuddyLibrary.ensure_builtin` (`Persona_Buddy/library.py`) now applies the deleted Persona_Visual installer's checks (`git show 0909623ad9^:tldw_chatbook/Persona_Visual/builtin_pixel_migu.py`, TASK-39057):
- each `assets.json` filename must be one flat name (a `str`, not `""`/`.`/`..`, and `Path(filename).name == filename`), else `ValueError("pixel_migu_asset_path_invalid")` before anything is read;
- each asset is read through `importlib.resources.as_file` and `load_persona_visual_asset` on the materialized leaf, which opens it without following a link and checks its byte count and sha256 (`PersonaVisualAssetError` is a `ValueError`). `as_file` also materializes a ZIP-backed Traversable.
The installer's `validate_path(filename, target)` check had a publication directory to resolve against; the seed has none, so the flat-name test stands in for it.

**Red first.** `test_buddy_seed_rejects_non_flat_asset_paths[../escape.png|/escape.png|nested/escape.png]` and `test_buddy_seed_rejects_symlinked_resource_leaf` (`Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py`, restored from the deleted installer's tests and retargeted at `ensure_builtin`). The escape file holds the real asset's bytes, so before the change all four reached publication: `Failed: invalid asset reached publication` (x3), `Failed: linked asset reached publication`. Green after; each also asserts no Buddy row.

**AC#3.** `test_buddy_seed_reads_resources_from_zip_traversable` and `Tests/Packaging/test_pixel_migu_distribution.py` (installed-wheel probe) pass: the file plus the probe, 25 passed at `-n 4`. Wider covering run (`Tests/Persona_Buddy/test_buddy_startup.py`, `test_buddy_library.py`, `Tests/Packaging/test_installed_distribution.py`): 253 passed, 3 failed; the same 3 fail on base `fu-base-r12` (`test_buddy_archive_publication_cannot_spoof_import_provenance`, `test_native_archive_review_is_read_only_and_preserves_notices`, `test_installed_wheel_loaders_entry_points_and_assets_are_immutable`).

**AC#4 not met.** No PR lane runs `Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py` (or any Buddy-seeding file), and this round's streams other than lanes do not edit workflows. Listed for the lanes stream. Timings of the four new nodes, serially (`-p no:xdist`, run.sh): 19.4 s, 14.7 s, 15.1 s at load 67-79 (pytest start-up included); the whole file 18 passed in 79.9 s at load ~70.

**Files.** `tldw_chatbook/Persona_Buddy/library.py`, `Tests/Persona_Visual/test_builtin_pixel_migu_buddy.py`.
<!-- SECTION:NOTES:END -->
