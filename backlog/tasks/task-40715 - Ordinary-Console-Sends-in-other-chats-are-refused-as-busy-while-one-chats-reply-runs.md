---
id: TASK-40715
title: "Ordinary Console Sends in other chats are refused as 'Workspace activity is busy' while one chat's reply runs"
status: Done
created_date: 2026-11-13 06:28
labels:
- follow-up
- console
- bug
- workspaces
priority: high
references:
- tldw_chatbook/Chat/console_ordinary_tree.py
- tldw_chatbook/Workspaces/worktree_activity.py
- tldw_chatbook/Agents/ordinary_tree.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** On macOS and Linux with default settings, while one chat's reply is running, Send in any other chat is refused with "Workspace activity is busy; retry after it finishes." (`Chat/console_chat_controller.py:10165`). A probe on 2026-11-12 (dev e0186fa10a) reproduced it with a second chat.

**Cause.**
- `Chat/console_ordinary_tree.py` `_capture_maximum` marks a turn's worktree footprint opaque when the run budget allows sub-agents and `spawn_subagent` is advertised and allowed (:540-545).
- `max_subagents` defaults to 2 (`Agents/agent_models.py:892`), and the Console does not make it configurable (`Chat/console_agent_bridge.py:578-583`).
- `install_skill` with a skills service (:551) and MCP-eligible non-ephemeral sessions (:457-481) also make the footprint opaque.
- `Workspaces/worktree_activity.py:145-146` makes an opaque footprint conflict with every other footprint. So `_prelaunch` (:410-417) returns no claim, and `Agents/ordinary_tree.py:626` raises `CapacityRefused("worktree_activity_busy")`.

Every turn goes through `admit_ordinary_turn` (`Chat/console_chat_controller.py:10149-10159`), which receives agent eligibility only as a flag. Only Windows skips the tree (`Chat/console_ordinary_tree.py:67`). Because an opaque footprint conflicts in both directions, a character chat or a send with a prefill is refused too while an agent turn runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A turn's footprint becomes opaque only once the turn does something whose footprint cannot be declared: spawning a sub-agent, installing a skill or running a skill script. Merely having those tools available does not make it opaque.
- [x] #2 With default configuration on macOS or Linux, a Send in a second chat, made while the first chat's provider call is still open, reaches the provider when neither turn has done such a thing.
- [x] #3 An opaque claim, including one widened mid-turn, still conflicts with every other turn in the runtime (opaque is runtime-wide, across workspaces).
- [x] #4 A POSIX regression test holds the first chat's provider call open, then sends from a second chat, both an agent chat and a character chat. Each gets its own provider call.
- [x] #5 When widening would conflict with another running turn, that one tool call is refused to the model with a clear message and the turn carries on; widening never waits.
<!-- AC:END -->

## Implementation Plan

ADR required: yes. AC #1 contradicted ADR-285 Decision 5, its Task8D2 ordinary-tree contract and Task 8D1 ("bounded claims cannot be upgraded"). Owner decision 2026-11-13: gate fix, then widen at the call; ADR-285 amended in this branch.

1. Reproduce on POSIX (agent and character second chat while the first agent reply is held).
2. Add `WorktreeActivity.widen_to_opaque`: upgrade a live claim in place under the activity lock, refuse (never wait) when any other claim, recovery protection or workflow window exists; keep the pre-widening footprint valid for `validate_claim`.
3. `OrdinaryTurn.widen`, called at child attach (every spawn path) and, after confirmation, before `install_skill` and `run_skill_script` effects; a busy refusal becomes a clear tool error and the turn continues.
4. Drop the three availability terms from `_capture_maximum`; MCP and the other up-front opaque terms stay.
5. Tests in a new file (the old one is red on dev), unit tests on the activity, mutation checks, covering files base vs head, preflight, Architecture.

## Implementation Notes

Owner decisions: 2026-11-13, widen at the call; 2026-11-14, the default-config terms. Both are recorded in ADR-285 ("Late opaque widening", plus pointers in Decision 5 and the Task8D2 section).

- `WorktreeActivity.widen_to_opaque` (`Workspaces/worktree_activity.py`) upgrades a live claim to OPAQUE in place under the activity lock. It returns `None` at once when any other claim, recovery protection or workflow window exists, so two widening turns refuse each other instead of deadlocking. The custody check moved into `_held_state`, so widening validates and upgrades under one lock.
- `validate_claim` audit (7 callers): it now also accepts the claim's recorded pre-widening footprint (`widened_from`). That covers `OrdinaryTurn.borrow`, which may read the bounded footprint while another thread widens. The five workflow callers validate claims that never widen, and `validate_opaque_claim` reads the current footprint, so neither changes.
- `OrdinaryTurn.widen` (`Agents/ordinary_tree.py`). `attach` widens first, so every child launch through `_reserve_child_execution` (spawn, skill forks, Code Review children, continuations) widens before the child's row. A busy refusal becomes a `SpawnAdmissionRefusal` with `widen_refusal` copy (`Agents/agent_service.py`); the turn carries on.
- `install_skill` and `run_skill_script` (`Chat/console_agent_preparation.py`) widen after the user's confirmation and before the effect; a busy refusal is a tool error and the turn carries on.
- `_capture_maximum` (`Chat/console_ordinary_tree.py`) no longer counts spawn/install/script availability. MCP-eligible sessions and the other up-front opaque terms are unchanged (remaining scope, per the decision).
- Gate fix (decision item 1): not shipped as separate code. Once availability of these three tools no longer feeds the capture, gating that availability on the request flags and `sandbox_supported()` would be deleted by the widening commit. Its outcome (no opacity for tools that are not used) is what the new tests check.
- Default-config terms (2026-11-14). A real-app probe found a default first Send still OPAQUE through personal context, Canvas and chatbook's own MCP tools (`builtin:tldw_chatbook`); the light tool surface offers none of them.
  - Admission fixes the light/full choice before capture (`tree.light_surface`, from the bridge's own `light_surface_for`). A `$`-mention draft always counts as full, because it may bind a skill.
  - `_prepare_native_reply` keeps a light admission light even if the live choice turned full, so the reply cannot widen past the claim.
  - On the light surface, Canvas no longer marks the maximum OPAQUE, and `builtin:tldw_chatbook` MCP rows no longer count. Full agent tools still count both up front. Composition and catalog binding are unchanged; the light allow-list keeps those tools from dispatch.
  - Personal context: the runtime's own app with nothing cached yet is bounded, because the stock bootstrap touches only private profile storage and `publish_personal_service` still refuses a non-stock result under a bounded claim. A cached service must be stock (the existing check), so a custom service stays OPAQUE. No code or function identity is read (ADR-356 rule 4).
- AC #2 evidence: `Tests/Chat/test_console_default_send_capture.py[default]`. A private child mounts the real `TldwCli` over the stock config template and holds the first chat's provider call; a second chat, sent through the Console's own pre-send capture, reaches the provider. Its `[full_tools]` case shows Full agent tools still count Canvas and chatbook's MCP tools up front (the second chat is refused busy).
- Tests: `Tests/Chat/test_console_ordinary_tree_widening.py` (new; the old file is broadly red on dev): the reproduction (agent and character), spawn widens and other chats are busy and released afterwards, a conflicting spawn is refused and not awaited, install/script widen or are refused, personal context is bounded only when stock (none/stock/custom), and the light surface fixed at capture cannot widen at reply. Two unit tests in `Tests/Workspaces/test_worktree_activity.py`, and the real-app probe above. Mutations: restoring the spawn term, dropping the widen in `attach`, skipping the skill-tool widen, ignoring other claims, and reverting each 2026-11-14 change (personal, Canvas, MCP filter, reply honour, capture pin) each fail the matching tests.
- Found, not fixed: `console_project_dispatch.py:115` cancels agent runs before the confirm callback; `Tests/Chat/test_console_ordinary_tree.py` is broadly red on dev (two child nodes checked: no provider call is made). A child run in these hosts also needs a resolution with `continuation_protocol`, which `_RecordingGateway`'s double lacks.

Review fixes, 2026-11-14 (Opus review: widening, lock and pin design sound; no Critical):

- Retained claim after a real child (was "found, not fixed", and it blocked every chat once the claim turned OPAQUE). Root cause: not an ordering race. `host_returned` reads the child's row inside a second `source.activate()` nested in `_finish_execution`'s own. On a fleet child's thread no SQLite handle is open at that point, so the readback opens one; the inner `operation_owned_connection` defers it to the outer owner (by design), but the inner `_retained_source_stack` then sees a handle that "did not return", raises, and never decrements `source_scopes`. The child stays not durable and the turn never retires (same on base 1455ceeb96). Fix in `_retained_source_stack` (`Agents/native_source_storage.py`): a stack nested in this thread's own retention of the same tree checks no handles; the outer stack's owner closes them and the outer stack checks them. Not the TASK-40716 change (that one only orders `host_returned`/`durable`); it does not touch `native_source_storage.py`.
- Fleet children kept `run_skill_script` after their parent's turn: `OrdinaryTurn.widen` runs `check()` and the disposed refusal first, then returns for an already-OPAQUE claim, then refuses a sealed turn. So a fleet child (or an up-front-OPAQUE turn) keeps its skill tools after the parent sealed, but not after disposal or a session/binding change. `attach` refuses a sealed turn first, so no new child joins it.
- Decision (b) for local tools: on the light surface the capture's permits drop local tools the surface does not offer (`ask_user` and `LIGHT_SURFACE_LOCAL_OPT_INS` stay), so `local_maximum` has no write tools and writable bindings get no write scopes from it. Those tools get no receiving record (`bind_catalog`) and are refused at dispatch. `bind_local_provider` requires write coverage of a writable root only when a reachable local tool writes; without that, two writable bindings failed the light send ("ordinary local root outside original maximum").
- Minor: `validate_claim` no longer accepts `footprint=None` for a never-widened claim; the widening refusal says "while other workspace activity is running"; the real-app probe waits for the second chat's provider call before releasing the first.
- Run-log root (owner decision 2026-11-14, ADR-285): the per-run `.<dir>/<run_id>` run-log directory is private storage, so `_capture_maximum` no longer declares a write over `tree.source.root` (the first writable binding). Only tools declare project writes; two agent chats on the same writable project now run at once. Checked that nothing else read that declaration (local root coverage, spill and sandbox under scratch, the run-log writer's own root checks, Change Review OPAQUE up front).
- `_retained_source_stack` keys nesting by (tree, asyncio task), so another task's retention held across an await on the loop thread is not an outer stack.
- Also, a Send made before startup's skill sources are ready captures OPAQUE (pending skill capture, by design); the probe now waits for `standard_console_configuration_sources` first, which made its `default` case flaky before.
- Tests (widening file unless noted):
  - `test_fleet_child_script_after_its_parent_runs_only_while_the_turn_is_current[none|rebound]`. It fails without the widen fix (script refused "this turn has already closed"), without the storage-scope fix (now in 15 s: a turn that never retires is released so teardown cannot hang), without `attach`'s sealed check, and, for `rebound`, with the early OPAQUE return before `check()`.
  - `test_widened_turn_refuses_a_skill_install_after_runtime_disposal` fails with the early return before the disposed refusal. A fleet child's script after disposal is stopped by other guards first, so disposal is tested on the parent's `install_skill`.
  - `test_retained_source_stack_defers_handles_only_to_its_own_task` fails when nesting is keyed per thread.
  - `Tests/Workspaces/test_worktree_activity.py::test_never_widened_claim_refuses_a_missing_footprint`.
  - Real-app probe `writable` (two writable bindings, light surface): the second chat's Send reaches the provider while the first is held. It fails with `local_writes` = fs_edit/fs_mkdir/fs_patch/fs_write without the light filter, makes no provider call without the `bind_local_provider` change, and is refused busy with the run-log root write restored.
