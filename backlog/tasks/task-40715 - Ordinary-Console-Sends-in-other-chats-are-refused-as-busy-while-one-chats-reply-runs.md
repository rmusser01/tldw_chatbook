---
id: TASK-40715
title: "Ordinary Console Sends in other chats are refused as 'Workspace activity is busy' while one chat's reply runs"
status: In Progress
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
- [ ] #2 With default configuration on macOS or Linux, a Send in a second chat, made while the first chat's provider call is still open, reaches the provider when neither turn has done such a thing.
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

Owner decision 2026-11-13: widen at the call. ADR-285 is amended in this branch ("Late opaque widening", plus pointers in Decision 5 and the Task8D2 section).

- `WorktreeActivity.widen_to_opaque` (`Workspaces/worktree_activity.py`) upgrades a live claim to OPAQUE in place under the activity lock. It returns `None` at once when any other claim, recovery protection or workflow window exists, so two widening turns refuse each other instead of deadlocking. The custody check moved into `_held_state`, so widening validates and upgrades under one lock.
- `validate_claim` audit (7 callers): it now also accepts the claim's recorded pre-widening footprint (`widened_from`). That covers `OrdinaryTurn.borrow`, which may read the bounded footprint while another thread widens. The five workflow callers validate claims that never widen, and `validate_opaque_claim` reads the current footprint, so neither changes.
- `OrdinaryTurn.widen` (`Agents/ordinary_tree.py`). `attach` widens first, so every child launch through `_reserve_child_execution` (spawn, skill forks, Code Review children, continuations) widens before the child's row. A busy refusal becomes a `SpawnAdmissionRefusal` with `widen_refusal` copy (`Agents/agent_service.py`); the turn carries on.
- `install_skill` and `run_skill_script` (`Chat/console_agent_preparation.py`) widen after the user's confirmation and before the effect; a busy refusal is a tool error and the turn carries on.
- `_capture_maximum` (`Chat/console_ordinary_tree.py`) no longer counts spawn/install/script availability. MCP-eligible sessions and the other up-front opaque terms are unchanged (remaining scope, per the decision).
- Gate fix (decision item 1): not shipped as separate code. Once availability of these three tools no longer feeds the capture, gating that availability on the request flags and `sandbox_supported()` would be deleted by the widening commit. Its outcome (no opacity for tools that are not used) is what the new tests check.
- AC #2 is shown on the `ordinary_host` test host (real ConsoleRuntime, controller, bridge and store, scratch-profile defaults), not in the full app. Not yet verified live: by default the light tool surface (Full agent tools off) offers none of the Library, personal-context, local, canvas or chatbook MCP tools, but `_capture_maximum` still counts them, so a default-config agent turn may stay opaque in the full app (for example the personal-context term until the service is cached as native).
- Tests: `Tests/Chat/test_console_ordinary_tree_widening.py` (new; the old file is broadly red on dev): the reproduction (agent and character), spawn widens and other chats are busy and released afterwards, a conflicting spawn is refused and not awaited, and install/script widen or are refused. Two unit tests in `Tests/Workspaces/test_worktree_activity.py`. Mutations: restoring the spawn term, dropping the widen in `attach`, skipping the skill-tool widen, and ignoring other claims each fail the matching tests.
- Found, not fixed: on dev a successful child in the `ordinary_host` host never proves durable (its `host_returned` readback fails, although the same row reads back correctly afterwards, so likely an ordering race), so the turn's claim is retained until restart; the same happens on base; `console_project_dispatch.py:115` cancels agent runs before the confirm callback; `Tests/Chat/test_console_ordinary_tree.py` is broadly red and hangs at session end.
