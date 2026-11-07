---
id: TASK-39124
title: "BuddyLibrary.ensure_builtin reads bundled pixel-migu files without the flat-filename and no-follow checks that the deleted Persona_Visual installer applied"
status: To Do
created_date: 2026-11-07 02:03
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
- [ ] #1 `ensure_builtin` refuses a non-flat `assets.json` filename (`../escape.png`, an absolute path, `nested/escape.png`) before reading it, and publishes nothing; tests are red before the change
- [ ] #2 `ensure_builtin` refuses a symlinked resource leaf (skipped only where symlinks cannot be created); red before the change
- [ ] #3 The ZIP-traversable seeding case and the installed-wheel packaging probe stay green
- [ ] #4 The new tests run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
