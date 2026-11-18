---
id: TASK-41648
title: "The Console workspace tree raises KeyError for conversation-attention-running when mounted during a running run"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- ui
priority: medium
references:
- tldw_chatbook/Widgets/Console/console_workspace_tree.py
dependencies:
- TASK-41313
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** In the burn-down's UI runs, mounting the Console while a chat held a running or retained run logged `KeyError: "No 'conversation-attention-running' key in COMPONENT_CLASSES"`. It is raised from `get_component_styles` in the workspace tree's projection sync. The key is declared (`Widgets/Console/console_workspace_tree.py:280`), so the likely cause is a style lookup that runs before styles are applied. A mid-run rebuild in the real app could hit it too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mounting the Console with a running chat does not raise or log this KeyError
- [x] #2 A test mounts with a running chat and asserts that the running attention style applies once styles are ready
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (a one-call defect fix inside one widget; no contract changes)

1. Reproduce: sync a running chat into an unmounted `ConsoleWorkspaceTree`, as the left rail does in `compose`.
2. Write a test that syncs before mount, mounts, and checks the running row's painted icon colour; show it red.
3. Fix the one component-style lookup in `render_label` so an unstyled widget falls back without caching, and mutation-check the colour assertion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** The left rail calls `sync_projection` in `compose` (`UI/Console_Modules/left_rail.py`), before Textual applies the tree's styles. `sync_projection` ends in `_update_tooltip`, which calls `get_node_at_line(-1)` when no row has the cursor; the negative index builds the tree lines. `Tree._build` measures every visible row through `render_label`, and a row with an attention class asked for `get_component_rich_style(data.action_class)` while `_component_styles` was still empty. So a running (or any attention-marked) chat's first sync raised `KeyError: "No 'conversation-attention-running' key in COMPONENT_CLASSES"`, and the Console failed to compose while a chat held a run. Later syncs on a mounted tree were fine, which is why the existing unmounted running-chat tests passed: their first sync had no attention.

**Fix.** `render_label` passes `default=base_style` to `get_component_rich_style`. The measuring pass gets that style. Painted rows get the real class colour because applying styles calls `notify_style_update`, which clears the widget's rich-style cache, and the Tree override clears its line cache too. So the fix would hold even if the fallback were cached; Textual also happens not to cache it (review note, 2026-11-18). One call site covers every caller that builds lines before styling, or after unmount, which also clears component styles.

**Evidence** (worktree fu2-redui at d0486978dd, scratch-profile wrapper):
- New `Tests/UI/test_console_workspace_tree.py::test_running_chat_synced_before_mount_gets_running_style_once_styled` syncs a running chat before mount, mounts with a CSS rule for `.conversation-attention-running`, and asserts the painted icon segment has that colour. Red on the unfixed code with the KeyError from `sync_projection`; green with the fix.
- The test reproduces the left rail's compose-time call on the tree in a small harness; it does not mount the whole Console. The left rail's `compose` is the only pre-styling caller (`sync_projection` has two callers, both in `left_rail.py`; the other runs on a mounted tree).
- Mutation: looking up `tree--guides` instead of the row's class keeps the call from raising but paints grey (`(62, 62, 62) != (0, 255, 0)`), so the test checks the style that applies, not only the absence of the error.
- `test_console_workspace_tree.py`, `test_console_workspace_tree_performance.py` and `test_console_workspace_context_rail.py` at `-n 4`, load ~35: 115 passed, 1 failed. The failure, `test_console_workspace_tree_performance.py::test_old_projection_baseline_is_reproducible` (fixture digest mismatch), fails the same way on dev d0486978dd and is reported as a follow-up.

Files: `tldw_chatbook/Widgets/Console/console_workspace_tree.py`, `Tests/UI/test_console_workspace_tree.py`.
<!-- SECTION:NOTES:END -->
