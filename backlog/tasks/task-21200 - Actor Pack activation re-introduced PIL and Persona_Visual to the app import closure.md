---
id: TASK-21200
title: >-
  Actor Pack activation re-introduced PIL and Persona_Visual to the app import
  closure
status: Done
assignee:
  - '@claude'
created_date: '2026-08-23'
updated_date: '2026-08-23'
labels:
  - bug
  - regression
  - performance
  - imports
  - actor-packs
  - startup
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Packaging/test_persona_buddy_import_closure.py::test_app_import_does_not_execute_persona_visual_or_pil`
is red on dev `7969089c3`. That guard was shipped by TASK-21103, which removed PIL,
`textual_image`, `rich_pixels`, `Persona_Buddy` and `Persona_Visual` from the
`import tldw_chatbook.app` closure. Evidence, measurements and file:line cites for the
original defect: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21103).

Origin trace. The Actor Packs import/activation feature re-introduced the whole chain
across three commits on one feature branch:

- `a160bbe83` "feat: export Actor Pack visual sections" — put the heavy imports at module
  scope in `Actor_Packs/export.py`;
- `ac1037732` "feat: import and activate Actor Packs" — did the same in
  `Actor_Packs/activation.py` and `Actor_Packs/importer.py`;
- `a98f3c14d` "fix: harden Actor Pack import activation" — carried them forward.

Those commits were authored 2026-08-22, before the guard existed. The guard merged as
`6c0abdba7` (#2002) at 2026-08-23 01:56, and the Actor Pack branch merged as `ae817fefe`
(#1998) at 2026-08-23 09:20 — after the guard, without rebasing onto it, and while CI was
not yet enforcing checks. So a branch that predated the invariant landed on a trunk that
had just established it, and nothing failed at the merge. CI now enforces checks, so the
regression is visible.

The cost is the one TASK-21103 removed: booting the app executes PIL and most of the
`Persona_Visual` tree to construct services the user may never invoke.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `import tldw_chatbook.app` executes no `PIL`, `textual_image`, `rich_pixels` or `tldw_chatbook.Persona_Visual` module.
- [x] #2 Actor Pack import, activation and export behaviour is unchanged; the Actor_Packs suites pass.
- [x] #3 Every public name of `tldw_chatbook.Actor_Packs` still resolves to the identical object it resolved to before the change.
- [x] #4 No import ordering (submodule-first, package-first, heavy-dependency-first) raises `ImportError`.
- [x] #5 A regression guard names `Actor_Packs.activation`'s heavy dependencies and fails with the offending import chain, not merely a module count.
- [x] #6 The guard is proven to fail when the regression is reintroduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the red guard; re-derive the chain with a `sys.meta_path` import hook rather
   than reading `-X importtime` indentation.
2. Decide the fix shape against the two entry paths (`Actor_Packs/__init__` and app.py's
   direct submodule imports).
3. Defer the heavy imports; prove every use site is still bound in its scope.
4. Prove name parity for the package's public surface against the pre-change surface.
5. Probe for circular imports unmasked by the deferral, in several orders.
6. Strengthen the guard, then mutation-test it.
7. Run the Actor_Packs, Packaging and App suites; A/B every red against `7969089c3`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the three Actor_Packs modules that `app.py` imports at module scope
(`activation.py`, `export.py`, `importer.py`) import their heavy dependencies
function-locally instead. `import tldw_chatbook.app` now executes 0 forbidden modules,
down from 15.

**Chain, re-derived with a `sys.meta_path` finder** (records the module executing at
first-import time, so it cannot be misread the way importtime's completion-ordered
indentation can):

```
tldw_chatbook.app -> Actor_Packs.persona_coordinator -> Actor_Packs (package init)
  -> Actor_Packs.activation -> Persona_Visual.repository
  -> Actor_Packs.activation -> Character_Chat.visual_identity -> PIL
```

**Why this shape.** Two alternatives were rejected on evidence:

- *A PEP-562 lazy facade on `Actor_Packs/__init__.py`* (the TASK-21103 shape) does not fix
  this. `app.py:392-396` imports `Actor_Packs.activation`, `.export` and `.importer`
  **directly**; importing a submodule runs the package init either way, and a lazy init
  still leaves the submodule itself executing PIL. The facade would have made the guard no
  greener. It was also unnecessary: nothing changed about the package surface, so the
  eager init stays and there is no facade to unmask a circular import.
- *Moving app.py's eight `Actor_Packs` imports into `_wire_character_persona_services`*
  would have turned the guard green while moving the cost, not removing it: that method is
  called from `TldwCli.__init__` (app.py:6076), so every real boot would still pay PIL.
  Fixing the three modules at the source removes PIL from module import **and** from app
  construction (`visual_identity` is only needed when a pack is actually
  imported/exported/activated), and gives the property to any future module-scope consumer
  rather than to app.py alone.

Deferral points, one import statement per consuming function (6 total), plus a
`TYPE_CHECKING` import for `export.py`'s single annotation — all three modules already
have `from __future__ import annotations`, so annotations stay strings at runtime:
`activation.__init__`, `activation._prepare_character_sections`,
`export._capture_persona_visual`, `export._capture_shared_visual`,
`importer._visual_authorities`, `importer._validate_sections`.

**Verification.**

- *Scope binding*: an AST pass confirmed all 14 load-references of the 9 deferred names
  resolve to a binding in their own function (or the `TYPE_CHECKING` block) — 14/14.
- *Name parity*: the pre-change public surface was read from `git show HEAD:` and every
  name checked with `getattr(pkg, name) is <direct submodule import>` — **33/33 identical
  objects**. Reading the baseline from git rather than from the edited file means a
  surface that silently shrank could not pass.
- *Circular imports*: 16 fresh-subprocess orderings — each of the 10 submodules first and
  alone, the package first, reverse-init order, heavy-dependency-first in both directions,
  and every deferred target resolved after the package — **16/16 pass**, no `ImportError`.
  (TASK-21160 is the precedent for taking this seriously: a lazy facade there unmasked a
  latent cycle that the eager init had been front-loading in a safe order.)
- *Guard mutation-tested*: re-adding one module-level `visual_identity` import to
  `activation.py` turned both guards red, and the new tracer printed the exact chain
  `__main__ -> tldw_chatbook.app -> tldw_chatbook.Actor_Packs -> ...activation ->
  ...Character_Chat.visual_identity -> PIL`. Reverted with `Edit`, never `git checkout`.

**Before/after**, 5 runs each, `import tldw_chatbook.app` in an isolated
HOME/XDG/`TLDW_CONFIG_PATH` env, fix worktree vs a clean worktree at `7969089c3`:

| | `sys.modules` | forbidden resident | min wall |
|---|---|---|---|
| `7969089c3` | 1700 | 15 (10 PIL + 5 Persona_Visual) | 0.741 s |
| this change | 1684 | **0** | 0.707 s |

The wall-clock delta here is ~34 ms, not the 1.28 s the holistic review measured — that
figure came from a slower floor machine with a cold PIL. The module-count and residency
deltas are the robust signal; the honest claim is that the closure is clean again, not
that this machine saves a second.

**Guard strengthening.** `Tests/Packaging/test_persona_buddy_import_closure.py` gained a
`_ChainTracer` (a `sys.meta_path` finder using only `sys._getframe`, installed before the
import under test) so both closure guards now report *which chain* pulled each heavy
module in, plus the advice to defer at the last `tldw_chatbook` module in the chain. New
`test_actor_pack_modules_do_not_execute_persona_visual_or_pil` pins the property at the
source: it imports the three modules directly, names each one's heavy dependencies in a
comment, and carries anti-vacuity assertions so it cannot pass on a failed or no-op
import.

**Test results** (venv pytest; every red A/B'd against a clean worktree at `7969089c3`):

- `Tests/Actor_Packs/` + `Tests/Architecture/test_actor_pack_boundary.py` — 211 passed.
- `Tests/Packaging/` — 1 failed, 47 passed, 58 errors; baseline 2 failed, 45 passed, 58
  errors. The delta is exactly the guard flipping red→green plus the new guard passing.
  `test_openai_tts_mapping_resource` and the 58 `test_mcp_unified_distribution` errors are
  pre-existing and identical on both sides.
- `Tests/App/` — 178 passed.
- Actor Pack UI workflows (5 files) + `Tests/UI/test_console_runtime_ownership.py` — 1
  failed, 59 passed. The failure,
  `test_app_fences_console_then_drains_buddy_before_profile_teardown`, is **pre-existing**:
  the file alone gives an identical 1 failed / 12 passed on both `7969089c3` and this
  branch. It is an asyncio drain timeout in Persona_Buddy teardown, untouched here.
- Full `--collect-only` sweep — 55,747 collected, 29 collection errors, byte-identical to
  the baseline's 29. No import-time breakage introduced.
- `scripts/check_persistent_diagnostic_inventory.py` — no drift (532 owners, 1222 TASK-492
  calls, 7261 TASK-494 calls, 8 sink files).

**Files modified**: `tldw_chatbook/Actor_Packs/activation.py`,
`tldw_chatbook/Actor_Packs/export.py`, `tldw_chatbook/Actor_Packs/importer.py`,
`Tests/Packaging/test_persona_buddy_import_closure.py`.
Not modified: `tldw_chatbook/Actor_Packs/__init__.py` (surface unchanged, see above),
`tldw_chatbook/app.py`.
<!-- SECTION:NOTES:END -->
