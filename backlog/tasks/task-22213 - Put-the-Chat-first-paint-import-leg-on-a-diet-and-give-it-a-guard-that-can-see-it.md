---
id: TASK-22213
title: >-
  Put the Chat first-paint import leg on a diet and give it a guard that can see
  it
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 16:06'
labels:
  - performance
  - startup
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22213).

Measured this review: warm boot-to-`_ui_ready` regressed ~140 ms (~11%) vs pin
`35d4bf3a1` (1323-1368 -> 1413-1509 ms, five interleaved runs) while the app IMPORT
closure got smaller and every import guard stayed green — the growth is on the legs the
guards cannot see. The Chat first-paint import leg grew +11,638 LOC / +10 modules since
the pin. Named edges (lane 5's AST closure, not diff-grep):
`UI/Screens/chat_screen.py:51` module-level-imports the entire TrajectoryScreen (~4,600
LOC of trajectory work landed since the pin rides the Chat leg);
`Chat/console_voice_input` (2,260 LOC) newly on the leg via `chat_screen.py:241`;
`Widgets/Console/__init__.py` eagerly re-exports the new tree/speech/authority widgets;
`Internal_Prompts` (10 modules) is still on the mount leg via
`Chat/console_chat_controller.py:266` although TASK-21731's title claims otherwise — its
guard (`Tests/Packaging/test_rag_boot_import_closure.py`) imports one module, never
`chat_screen`. PIL and keyring also load pre-first-paint via chat_screen chains
(pre-existing; `session.py:189 -> visual_identity.py:24`; `image.py:38 ->
Image_Generation/config.py:15`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `TrajectoryScreen` is not imported at chat_screen module level (screen-registry route or local import at the navigation seam)
- [x] #2 `Internal_Prompts` is off the Chat mount leg, or kept with a measured, stated cost
- [x] #3 The closure guard is extended to assert DEFERRED_PREFIXES absent after importing `UI.Screens.chat_screen` (closing the one-module blind spot)
- [x] #4 chat_screen module import time and boot-to-`_ui_ready` measured before/after with the review's interleaved A/B method; the regression is at least halved or the residual is attributed
- [x] #5 A `sys.modules` census at `_ui_ready` is pinned as a guard so mount-leg growth is visible in review (the import-weight guard's documented blind spot)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red-first: extend Tests/Packaging/test_rag_boot_import_closure.py with a chat_screen-leg closure test (import UI.Screens.chat_screen in a fresh interpreter; assert DEFERRED_PREFIXES absent AND the trajectory family absent; anti-vacuity: on-demand resolution still works). Confirm RED today for Internal_Prompts (10 modules) + trajectory family.\n2. Defer TrajectoryScreen: remove chat_screen.py:51; local import at the only use site (action_open_trajectory_view push seam). Chat.trajectory (line 144) stays: needed by _build_trajectory_snapshot and on the leg via agent_service anyway.\n3. Take Internal_Prompts off the leg at BOTH edges found by the first-import trace: console_chat_controller.py:266 (lazy get_internal_prompt wrapper, same name so call sites and module-namespace patches keep working) and Agents/agent_service.py:37-38 (lazy wrapper + PEP 562 module __getattr__ for the module-level SUBAGENT_SYSTEM_PROMPT catalog constant, cached on first access; consumers console_agent_bridge/tests are off the leg). Re-probe and iterate until the leg is clean.\n4. console_voice_input is NOT deferred: chat_screen.py:241 is not the load-bearing edge -- composer_bar.py:39, dictation.py:120, hands_free.py:124 all import it at module level and are legitimately on the leg; deferring means reworking the dictation/hands-free seams (separate task, documented in notes).\n5. New census guard Tests/Performance/test_ui_ready_module_census.py: subprocess headless-Pilot boot to _ui_ready against a scratch profile (first_run completed, splash off, valid-shaped key); pin tldw-module count at ready with a stated raise procedure; assert deferred families absent at ready (catches deferrals that merely move cost to mount); document blind spots honestly.\n6. Interleaved A/B (5 pairs) vs base fce939e00 (git-archive extracted tree, same venv, cwd pins the package, tldw_chatbook.__file__ asserted per probe): boot-to-_ui_ready + chat_screen module import time. Report spread honestly.\n7. Targeted tests + --collect-only sweep + preflight; mutation-test both guards (re-add module-level TrajectoryScreen import; eager Internal_Prompts import).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both named legs cut, both guard blind spots closed; the wall-clock A/B is an honest wash on a loaded machine while the deterministic censuses prove the diet.

**Changes.**
- `UI/Screens/chat_screen.py`: the module-level `from .trajectory_screen import TrajectoryScreen` (line 51) is gone; the class is imported locally at its only use site, the `y` action (`action_open_trajectory_view`). That takes the whole trajectory family off the Chat first-paint leg -- `trajectory_screen` + `trajectory_import` + `trajectory_export` + `trajectory_timeline` + `trace_filter_bar` (~4,400 LOC). `Chat.trajectory` deliberately stays (the action's off-thread snapshot build needs it, and `agent_service` imports its redaction helpers on the same leg anyway).
- `Chat/console_chat_controller.py`: the module-scope `from tldw_chatbook.Internal_Prompts import get_internal_prompt` is replaced by a same-name, same-signature lazy wrapper -- call sites and module-namespace patches unchanged.
- `Agents/agent_service.py` (the FIRST edge by import-stack trace, via `console_fleet_wake`): same lazy `get_internal_prompt` wrapper, plus a PEP 562 module `__getattr__` for `SUBAGENT_SYSTEM_PROMPT` (cached into `globals()` on first access; `console_agent_bridge`'s from-import triggers it off the boot leg; the dual-prefix `_is_subagent` contract is untouched).
- `Tests/Packaging/test_rag_boot_import_closure.py`: new `test_chat_screen_import_does_not_execute_the_deferred_packages` -- imports `UI.Screens.chat_screen` in a fresh interpreter, asserts all three DEFERRED_PREFIXES absent AND the new `CHAT_LEG_DEFERRED_MODULES` (trajectory family) absent, then proves on-demand resolution still works (controller prompt resolve, `agent_service.SUBAGENT_SYSTEM_PROMPT` == catalog default, `TrajectoryScreen` importable). Born RED (named exactly the 10 Internal_Prompts modules), green after the diet.
- `Tests/Performance/test_ui_ready_module_census.py` (new): subprocess headless-Pilot boot to `_ui_ready` against a scratch profile (first-run completed, splash off, valid-shaped key, pre-importer pinned off), censuses `tldw_chatbook.*` residency at the ready flag. Budget 970 vs measured 938-939 WARM (the test warms the profile with a throwaway first boot); asserts Chunking / RAG_Search.simplified / trajectory family absent at ready; raise procedure and blind spots documented in the module docstring (residency-not-time, pre-importer excluded, post-ready timers invisible, tldw-only).

**AC2 taken on its second arm.** Internal_Prompts is OFF the Chat IMPORT leg (both edges). It is still resident at `_ui_ready` because the MOUNT path resolves it: `chat_screen._ensure_console_agent_bridge` imports `console_agent_bridge`, whose module-scope catalog constants (`CONSOLE_AGENT_OPERATING_PROMPT`, `_KNOWN_SUBAGENT_PREFIXES` seed) need the catalog. Measured marginal cost with the app already imported: **1.0-2.4 ms warm** (the 65-92 ms standalone number is the parent-package import, not the catalog). Making the bridge lazy would mean touching the security-relevant `_is_subagent` prefix-seeding mechanism for ~2 ms -- rejected under the stability-over-quick-wins ruling; documented in the census guard.

**Measurements (A/B vs own base fce939e00; git-archive base tree, same venv, cwd-pinned package, `__file__` asserted per probe; scratch profiles per the review recipe).**
- Deterministic axes: bare `chat_screen` leg 600 -> 568 tldw modules (-32); after-app chat_screen closure 905 -> 890 (-15); warm `_ui_ready` census 943-946 -> 938-941 (-5, the trajectory family; Internal_Prompts stays via the bridge). First-import stack traces on file in the closure-guard docstrings.
- chat_screen module import time: equal within noise (base 221-357 ms, after 233-618 ms across 4 interleaved pairs) -- matches the review's own finding that module count and import time decoupled here.
- Boot-to-`_ui_ready`: 10 forward interleaved pairs showed after faster 8/10 (median ~-250 ms) -- but an A/A control (base vs base) measured a +/-400 ms noise floor AND a systematic second-position advantage, and 5 order-REVERSED pairs collapsed the delta to a wash (median -19 ms, mean +85 ms). Honest verdict: no reproducible wall-clock claim on this loaded machine (ready times ran 2.5-4.8 s vs the review's 1.3-1.5 s). The residual is attributed: this task's leg is verifiably gone on the deterministic axes; the review's other filed contributors (22214 pre-importer payload, 22215 worker fleet, 22217 PIL seeding, 22222 CSS bytes + catalog refresh on the push line) own the rest of the ~140 ms. Lesson recorded in lessons-testing-evidence.md (positional bias in interleaved pairs).

**Mutation tests.** Re-adding the module-level TrajectoryScreen import reds the new guard naming the whole family; re-adding an eager Internal_Prompts import at the controller reds it naming all 10 modules. Both restored by Edit.

**Census guard's first catch (documented, not fixed here):** a FRESH profile's first boot has the entire Chunking engine (34 modules) resident at ready via `_init_media_db -> _apply_migration_v6_to_v7 -> Chunking._template_conversion` -- legitimate one-time migration work; the guard censuses the warm boot and names this as a first-boot blind spot (22200 post-upgrade-window family).

**Not done, deliberately.** `console_voice_input` (2,260 LOC) stays on the leg: chat_screen.py:241 is not the load-bearing edge -- `console_composer_bar.py:39`, `Console_Modules/dictation.py:120`, and `Console_Modules/hands_free.py:124` all module-import it and are legitimately on the leg; deferring it means reworking the dictation/hands-free seams (its own task). `Widgets/Console/__init__.py` eager re-exports untouched for the same reason.

**Pre-existing dev reds hit during verification (all reproduced byte-identical on pristine base fce939e00, not this task's):** `Tests/UI/test_console_modal_dismissal.py` x2 (AttributeError `Image_Generation.worker` in the AST walk; 4 undeclared modals incl. `ConsolePromptComparisonModal` -- the 22212 family -- `TraceExportDialog`, and the two ProjectInstruction modals), `Tests/Internal_Prompts/test_websearch_prompt_parity.py::test_result_relevance_eval_parity` (catalog copy drifted from the test's expected text), `Tests/UI/test_console_hands_free_wiring.py` x4. Collect-only sweep: 59,368 collected / 28 errors, all missing optional extras, identical set at base (59,366 / 28).

**Test counts (from tees).** Closure+import+census guards 13 passed / 3 env-skips; trajectory+modal batch 334 passed / 2 pre-existing reds; agents+internal-prompts 191 passed / 1 pre-existing red; controller+voice 406 passed; dictation-UI+boot 170 passed / 4 pre-existing reds; census guard 3x consecutive green; preflight all green.
<!-- SECTION:NOTES:END -->
