---
id: TASK-40715
title: "Ordinary Console Sends in other chats are refused as 'Workspace activity is busy' while one chat's reply runs"
status: To Do
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
- [ ] #1 A turn's footprint becomes opaque only once the turn does something whose footprint cannot be declared, such as spawning a sub-agent or installing a skill. Merely having those tools available does not make it opaque.
- [ ] #2 With default configuration on macOS or Linux, a Send in a second chat, made while the first chat's provider call is still open, reaches the provider when neither turn has done such a thing.
- [ ] #3 An opaque claim still conflicts with every other turn in its worktree.
- [ ] #4 A POSIX regression test holds the first chat's provider call open, then sends from a second chat, both an agent chat and a character chat. Each gets its own provider call.
<!-- AC:END -->
