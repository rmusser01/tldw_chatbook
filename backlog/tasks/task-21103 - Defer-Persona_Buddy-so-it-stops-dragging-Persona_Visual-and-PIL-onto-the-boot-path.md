---
id: TASK-21103
title: >-
  Defer Persona_Buddy so it stops dragging Persona_Visual and PIL onto the boot
  path
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 08:39'
labels:
  - performance
  - startup
  - imports
  - persona-buddy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21103).

`Persona_Buddy` (eager at app.py:393) drags 93% of Persona_Visual (6,633 LOC) via
`controller.py:18,23` and `rendering.py:11-13` - the latter imports the tree for a single int
constant. This chain puts `PIL.Image`/`PIL._imaging` on the boot path: measured 1.276 s of the
3.10 s cold app import (41%). Both consumers already tolerate absence (`app.py:8582`,
`console_runtime.py:468,519`); the lazy-service house pattern is `_build_rag_admin_services`
(app.py:6054-6124).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `import tldw_chatbook.app` no longer imports Persona_Visual or PIL - pinned by a sys.modules assertion test
- [x] #2 The buddy controller is constructed lazily at first feature use; enabling/using Persona Buddy still works end to end
- [x] #3 Cold and warm importtime before/after numbers recorded in the task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: cold+warm python -X importtime on isolated config env (test-logs/t21103/) + empirical PIL census of the app import closure.
2. Census result: PIL enters via (a) app.py:188 console_runtime -> Persona_Buddy.console_adapter -> Persona_Buddy/__init__ -> controller -> Persona_Visual.repository/runtime -> Persona_Visual/__init__ (imports authoring/assets/importer -> 4x 'from PIL import Image'); (b) app.py:433 chat_message_enhanced (direct PIL + textual_image); (c) app.py:495 image_gen_command_provider -> image_gen_demo_screen -> Image_Generation.worker -> request_validation (PIL). All three must break for the PIL assertion to hold.
3. Convert Persona_Buddy/__init__.py and Persona_Visual/__init__.py to PEP-562 lazy facades (house precedent: tldw_api facade) so importing console_adapter or any PV submodule no longer executes the tree; rendering.py keeps importing MAX_ASSET_DIMENSION from PV.contracts, now tree-free (constant NOT moved - lazy package init achieves the same with zero import-site churn; deviation documented).
4. app.py: drop the eager Persona_Buddy import; replace eager PersonaBuddyController construction with slot+lock + lazy read-only property modeled on _build_rag_admin_services (task-254): property returns cached controller, else None when [persona_buddy].enabled is false (keeps the disabled-case reconcile early-out construction-free), else builds on first access; ensure_persona_buddy_controller() builds regardless (explicit feature use); _shutdown_persona_buddy peeks the slot.
5. personas_screen._handle_persona_buddy_action: resolve controller via ensure accessor so enabling Buddy from a disabled state still works end to end.
6. Defer app.py:433 ChatMessageEnhanced to function-local imports (4 query sites) and image_gen_command_provider's ImageGenDemoScreen into search().
7. Red-first guard: Tests/Packaging/test_persona_buddy_import_closure.py modeled on test_chunking_import_closure.py - subprocess assert no Persona_Visual*, no Persona_Buddy.controller/rendering, no PIL*, no rich_pixels/textual_image after import tldw_chatbook.app + anti-vacuity closure members.
8. Verify: buddy tests (Tests/Persona_Buddy/, UI mount/widget, workbench -k buddy, console runtime/bridge), architecture boundary tests, packaging closure family, base_app_screen reconcile tests; A/B known reds against base 41a240ccd; full --collect-only sweep.
9. After importtime: record cold+warm before/after table in task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`import tldw_chatbook.app` no longer executes Persona_Buddy's controller chain, Persona_Visual, PIL, rich_pixels, or textual_image — pinned by the subprocess guard `Tests/Packaging/test_persona_buddy_import_closure.py` (red on the unfixed base with 42 heavy modules resident, green after).

### PIL census (empirical, importtime tree on base `41a240ccd`)

The buddy chain was NOT the only PIL path. Four independent chains put PIL in the app import closure; all four were broken (each cheap and behavior-preserving), so the full "no PIL after app import" AC holds unscoped:

1. **Buddy chain** (the review's finding): app.py:188 `Chat/console_runtime` → `Persona_Buddy.console_adapter` → eager `Persona_Buddy/__init__` → `controller` → `Persona_Visual.repository/runtime` → eager `Persona_Visual/__init__` → `assets`/`importer`/`authoring_workspace`-adjacent modules with module-level `from PIL import Image`; plus the eager `from .Persona_Buddy import ...` at app.py:390.
2. app.py:433 → `Widgets/Chat_Widgets/chat_message_enhanced` (module-level PIL AND the whole `textual_image` package, 17 modules).
3. app.py:495 → `UI/image_gen_command_provider` → module-level `ImageGenDemoScreen` import → `Image_Generation.worker` → `request_validation` (module-level PIL).
4. `Actor_Packs/contracts.py` module-level `try: from PIL.Image import DecompressionBombError` (in closure via `Actor_Packs.persona_coordinator`, app.py:387) — masked in the review's census because the buddy chain had already imported PIL by then.

### Approach

- `Persona_Buddy/__init__.py` and `Persona_Visual/__init__.py` are now PEP-562 lazy facades (TYPE_CHECKING imports + `_EXPORTS` name→submodule map + module `__getattr__`/`__dir__`). Importing the stdlib-only seams (`console_adapter`, `preferences`, `Persona_Visual.contracts`) no longer executes the tree. **Deviation from the task sketch:** the `rendering.py` int constant (`MAX_ASSET_DIMENSION`) was NOT moved — the tree-pull was the package-init side effect, not the constant's home, and the lazy `Persona_Visual/__init__` makes `from ...contracts import MAX_ASSET_DIMENSION` tree-free with zero import-site churn (pinned by the second guard test).
- app.py: eager construction replaced by slot + `threading.Lock` (initialized BEFORE `ConsoleRuntime(self)`, whose constructor reads the property) + lazy `persona_buddy_controller` property mirroring the `_build_rag_admin_services` house pattern. Property semantics: cached controller if built; `None` without constructing when `[persona_buddy] enabled` is false (parsed via the stdlib-only preferences seam — the every-screen-mount reconcile early-out stays PIL-free); first access on an enabled profile constructs. A setter allows test-double injection. `ensure_persona_buddy_controller()` constructs regardless of the enabled flag; `PersonasScreen._handle_persona_buddy_action` falls back to it when the passive read is None, so "Use for Buddy" from a disabled profile still enables end to end. `_shutdown_persona_buddy` peeks the slot (a never-built controller must not be constructed just to be drained). Construction gates only on `local_character_persona_service` (the old wiring ran right after `_wire_character_persona_services()`; `chachanotes_db` passes through as-is — legitimately None on the test-app factory).
- Chains 2–4: `ChatMessageEnhanced` became function-local in the two TTS event handlers that query it; `ImageGenDemoScreen` import moved inside `ImageGenCommandProvider.search()`; `DecompressionBombError` resolution moved into `_validate_portrait` (its only consumer).
- Source-pin updates in `Tests/UI/test_console_runtime_ownership.py`: the two pins that asserted the old eager `__init__` construction now pin the new contract (init construction-free, slot precedes ConsoleRuntime, builder keeps the portrait-loader semantics behind the service guard, disposer peeks the slot), plus a new behavior test for the property's three states (disabled→None without construction; ensure→constructs and caches; enabled→first passive read constructs; setter respected).

### importtime (this machine: M-series, Python 3.12.11, isolated HOME/XDG/TLDW_CONFIG_PATH, `[first_run] setup_completed=true`, `[splash_screen] enabled=false`; cold = bytecode purged, warm = 3 consecutive runs; logs in `test-logs/t21103/`)

| metric | base `41a240ccd` | fixed | delta |
|---|---|---|---|
| cold (run A / run B) | 2.092 s / 2.476 s | 1.956 s / 1.991 s | ≈ −0.14 to −0.49 s (cold is noisy) |
| warm median of 3 (run A / run B) | 0.778 s / 0.772 s | 0.716 s / 0.710 s | ≈ −0.063 s (−8.2%) |
| modules imported | 1,711 | 1,629 | −82 |

Note the review's 1.276 s was PIL's cold cost on the floor machine; on this dev machine PIL is page-cache-hot (~8 ms warm), so the wall-clock delta here is modest while the structural change (PIL/_imaging + 93% of Persona_Visual + textual_image off the boot path, −82 modules) is exactly what the guard pins for the machines where it hurts.

### Verification (all vs base `41a240ccd`, worktree venv)

- New guard: 2/2 green (red-first proven: the closure test failed on the unfixed base listing 42 resident heavy modules).
- Buddy/visual/architecture/packaging batch: 674 passed; the only failures were 58 errors + 1 fail in `Tests/Packaging` build-artifact tests, all "Backend 'setuptools.build_meta' is not available" (environmental, this uv venv can't build wheels; unrelated).
- UI buddy trio + workbench state (`test_persona_buddy_app_mount`, `test_persona_buddy_widget`, `test_console_persona_visual_identity`, `test_personas_workbench_state`): 62 passed, 0 failed.
- `test_personas_persona_visual_authoring` + `test_personas_persona_visual_pack` + `test_console_agent_bridge` + `test_persona_buddy_console_adapters` + `test_app_import_weight`: 291 passed, 3 skipped; 2 errors = tiktoken cache-miss network-guard teardowns, reproduced 3× on base (pre-existing/environmental).
- `test_console_realtime_loop` + `test_console_fleet_wake` + `test_screen_navigation`: 193 passed. (`test_fleet_teardown_notice.py` excluded — known >420 s hang.)
- `test_personas_workbench.py -k buddy`: 19 passed.
- `Tests/UI/test_console_runtime_ownership.py`: file is UN-COLLECTABLE standalone on base AND fixed (pre-existing circular import: test cross-imports reach `settings_screen` → `RAG_Search.config_profiles` ↔ `simplified`). Run with a cycle-breaking pre-import: fixed 14 passed / 1 failed vs base 11 passed / 1 failed — the one failure is the KNOWN pre-existing `test_app_fences_console_then_drains_buddy_before_profile_teardown` red, failing with the IDENTICAL TimeoutError signature on both sides.
- Live construction probes on the real `TldwCli` (test-app factory): init completes with the lazy slots; disabled profile passive read → None with PIL absent; `ensure` → real `PersonaBuddyController`, PIL loads lazily, snapshot works; enabled profile passive read constructs and caches.
- Full `--collect-only` sweep: 55,379 collected (base 55,376; +3 = the new tests), 29 collection errors byte-identical to base (all pre-existing).

### Files

- `tldw_chatbook/Persona_Buddy/__init__.py`, `tldw_chatbook/Persona_Visual/__init__.py` — PEP-562 lazy facades
- `tldw_chatbook/app.py` — lazy property/builder/ensure/setter, slot before ConsoleRuntime, slot-peeking shutdown, function-local `ChatMessageEnhanced`
- `tldw_chatbook/UI/Screens/personas_screen.py` — ensure-fallback in the Buddy action handler
- `tldw_chatbook/UI/image_gen_command_provider.py` — screen import deferred into `search()`
- `tldw_chatbook/Actor_Packs/contracts.py` — `DecompressionBombError` function-local
- `Tests/Packaging/test_persona_buddy_import_closure.py` (new), `Tests/UI/test_console_runtime_ownership.py` (pins rewritten + behavior test)
- `backlog/docs/lessons-testing-evidence.md` — MagicMock-auto-attr getattr trap + source-pin-substring-in-comment trap (both bitten in this task)
<!-- SECTION:NOTES:END -->
