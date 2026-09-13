---
id: TASK-22217
title: >-
  Keep PIL off the warm boot path: lazy import in visual_identity
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
labels:
  - performance
  - startup
  - personas
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22217).

Traced live this review: every boot, `app.py:8784 _init_notes_service` ->
`get_chachanotes_db_lazy()` -> `seed_builtin_content()` (`config.py:7231`) ->
`Character_Chat/visual_identity.py:24` module-level `from PIL import Image, ...`.
`ensure_builtin_samira` preflights and exits early on warm boots — but the PIL import
(~80 modules) is paid before the preflight can run, on the init thread pool, every boot,
in every profile. This undermines TASK-21103/21200 (which keep PIL out of the import
closure) via the construct-time gap the guards cannot see; PIL was confirmed present at
`_ui_ready` on tip.

## Acceptance Criteria

- [x] A warm boot with seeding already terminal loads no PIL **through the seeding
  chain**: the real `seed_builtin_content` -> `ensure_builtin_samira` chain imports
  `visual_identity` with zero `PIL*` modules in `sys.modules` (subprocess census,
  `Tests/Character_Chat/test_visual_identity_pil_lazy.py`). **Corrected from the
  original whole-process wording** ("census at `_ui_ready` under a default
  profile"): measured on a real headless warm boot, PIL is STILL present at
  `_ui_ready` (16 modules) via `UI/Console_Modules/character_avatar_layout.py:7`
  reached from `Chat/console_image_view.py:21` -- chat/Console pre-first-paint
  chains, which are finding 22213's scope, not this task's. A whole-process
  census cannot pass until 22213 lands, and would not isolate a regression in
  THIS chain even if it could; the committed guard therefore scopes to the
  seeding chain. (Warm-boot PIL footprint still dropped 72 -> 16 modules,
  because the seeding chain no longer triggers `Image.open`'s full plugin
  registry.)
- [x] Fresh-profile seeding still works end to end (Samira card + pack created)
- [x] PIL imports move inside the code paths that actually do image work in `visual_identity.py`

## Implementation Plan

1. Census every PIL-name use in `visual_identity.py` (Image, UnidentifiedImageError,
   annotations, except clauses) and every module importing PIL names FROM
   visual_identity. Confirm the file has `from __future__ import annotations`.
2. Red-first census test (`Tests/Character_Chat/test_visual_identity_pil_lazy.py`):
   two clean subprocesses against one scratch profile DB. Phase 1 runs the REAL
   `seed_builtin_content` chain on a fresh DB (seeds Samira; PIL allowed) and
   asserts the pack exists. Phase 2, a clean interpreter against the now-terminal
   DB, runs the REAL `seed_builtin_content` -> `ensure_builtin_samira` chain and
   asserts `tldw_chatbook.Character_Chat.visual_identity` is in `sys.modules`
   while no `PIL*` module is. Red today (module-level import at line 24).
3. Fix: drop the module-level `from PIL import Image, UnidentifiedImageError`;
   import it function-locally at the single function that does image work
   (`_inspect_image_bytes`, which contains ALL runtime uses incl. the except
   tuple). The `Image.Image` annotation on `_image_duration_ms` stays a string
   under `from __future__ import annotations`. Missing-PIL still surfaces as the
   same ImportError, now at first image work instead of module import; both are
   contained by `seed_builtin_content`'s `except Exception` on the boot path.
4. Verify the whole-boot claim with a real headless boot (subprocess, isolated
   scratch profile, boot twice, census PIL at `_ui_ready` on the warm boot).
   Finding 22213 says chat_screen's own chains load PIL pre-first-paint; if that
   holds, scope the committed guard to the seeding chain and correct AC 1
   honestly, citing 22213.
5. Fresh-profile e2e: run the existing lifecycle tests that exercise the real
   image path (`Tests/Character_Chat/test_visual_identity_lifecycle.py`, assets,
   contract, resolution, preflight suites).
6. Targeted tests + `--collect-only` sweep (tee everything); `./scripts/preflight.sh`.
7. Mutation test: Edit-restore the module-level PIL import -> census test must
   go red; Edit-revert.
8. Tick ACs (with honest corrections), Implementation Notes, status Done, commit,
   push.

## Implementation Notes

The warm-boot seeding chain no longer imports PIL. `visual_identity.py`'s
module-level `from PIL import Image, UnidentifiedImageError` moved inside
`_inspect_image_bytes` -- the census of the whole 3,489-line file found that
EVERY runtime PIL use (the `Image.DecompressionBombWarning` filter, `Image.open`,
and the except tuple catching `UnidentifiedImageError`/`DecompressionBombError`/
`DecompressionBombWarning`) already lives inside that one function, so one
function-local import covers them all. The only other reference, the
`Image.Image` annotation on `_image_duration_ms`, stays a string under the
file's `from __future__ import annotations`; a `TYPE_CHECKING`-guarded
`from PIL import Image` keeps it resolvable for type checkers and ruff. No
module imports PIL names FROM `visual_identity` (grepped package + Tests), so
nothing else changes. Missing Pillow (a hard dep, so hypothetical) still
surfaces as the same ImportError, now at first image work instead of module
import; on the boot path both are contained by `seed_builtin_content`'s
`except Exception`.

- New guard: `Tests/Character_Chat/test_visual_identity_pil_lazy.py` -- two
  clean subprocesses on one scratch profile: phase 1 runs the REAL
  `seed_builtin_content` fresh seed (asserts card + pack, so phase 2 cannot
  measure a half-seeded profile); phase 2 reruns the real chain warm and
  asserts `visual_identity` imported with zero `PIL*` modules. Red before the
  fix (caught `['PIL', 'PIL.ExifTags', 'PIL.Image', ...]`), green after;
  mutation-verified (re-adding a module-level PIL import turns it red).
- AC 1 corrected to the seeding-chain scope, with first-hand measurement:
  real headless warm boot to `_ui_ready` still holds 16 PIL modules via
  `character_avatar_layout.py:7` <- `console_image_view.py:21` (finding 22213,
  out of scope here); fresh boot holds 72. See the review doc, finding 22217.
- Verification: red-first census; 1,198 passed across Tests/Character_Chat +
  Tests/Actor_Packs + vendor-pin (1 pre-existing unrelated red,
  `test_character_persona_scope_service.py::test_app_wires_character_persona_services`,
  fails identically with the module-level import restored -- an Actor_Packs
  Mock-wiring failure, and its solo-run collection hits the 22223 circular
  import); 311 passed across the seven Tests/UI visual-identity suites;
  full `--collect-only` 59,176 collected (28 errors, all missing optional
  deps: numpy/audio/TTS/transcription/Confluence); `./scripts/preflight.sh`
  all green; ruff clean on both changed files.
- Files: `tldw_chatbook/Character_Chat/visual_identity.py`,
  `Tests/Character_Chat/test_visual_identity_pil_lazy.py`, this task file.
