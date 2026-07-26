---
id: TASK-654
title: Surface sandboxed file tools to the agent loop
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 18:05'
labels:
  - agents
  - tools
  - security
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tools/file_operation_tools.py` provides `ReadFileTool`, `ListDirectoryTool` and `WriteFileTool`, all confined to a configurable `[tools] file_sandbox_root`. They are registered on the global `ToolExecutor`, but the Agents runtime's `BuiltinToolProvider` hardcodes only `CalculatorTool` and `DateTimeTool` — so the agent loop never sees them.

That gap has a concrete cost: skill-script output is deliberately retained under the file-tool sandbox root (task-584) precisely so the app's existing contained tooling can reach it, and today nothing in the agent loop can.

This is a posture-relevant change even though the tools stay behind their existing gates, so it is split from task-584 to be reviewed on its own merits rather than riding along with output retention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `read_file` and `list_directory` are reachable from the agent loop when their existing `[tools]` gates are enabled
- [x] #2 The default posture is unchanged — with no config, the agent sees exactly the tools it saw before
- [x] #3 Each gate is independent; enabling one does not surface the other
- [x] #4 A tool that cannot be constructed (missing module, bad config) is simply absent rather than breaking provider construction
- [x] #5 The names are covered by the shadowed-builtin guard, so a skill cannot silently shadow them once a gate is enabled
- [x] #6 `list_directory` cannot enumerate outside the sandbox root via a planted symlink
<!-- AC:END -->

## Implementation Notes

Extended `BuiltinToolProvider.__init__` to add `ReadFileTool`/`ListDirectoryTool` when `read_file_enabled` / `list_directory_enabled` are set, reusing the **same** `[tools]` gates that already govern their registration on the global `ToolExecutor`. Both default to disabled, so this changes *reachability*, not the default posture. Construction failures are swallowed per-tool, so an unavailable tool is absent rather than fatal.

**AC#5 is the non-obvious one.** The shadowed-builtin drift guard (task-580) builds a `BuiltinToolProvider` with *default* config, so a config-gated tool is invisible to it — the guard would never have flagged these names. They are therefore pinned explicitly in `_SHADOWED_BUILTIN_NAMES`, with a test that enables both gates and asserts coverage. Without that, a skill named `read_file` would silently shadow a real builtin the moment a user turned the gate on.

`WriteFileTool` is deliberately **not** surfaced: granting the agent loop filesystem writes is a larger decision than making retained output readable, and nothing here needs it.

**AC#6 was found by review and is the reason this split mattered.** A symlink planted inside `file_sandbox_root` let `ListDirectoryTool`'s recursive walk enumerate files anywhere on disk — reproduced before fixing (an out-of-sandbox file appeared in the listing). Latent while nothing in the agent loop could call the tool; reachable the moment this PR surfaces it, so it is fixed here rather than deferred. The walk now refuses to descend into symlinked directories and re-checks that each child resolves under the root.

Files: `tldw_chatbook/Tools/file_operation_tools.py`, `tldw_chatbook/Agents/tool_catalog.py`, `tldw_chatbook/Library/library_skills_state.py`, `Tests/Agents/test_builtin_file_tools.py`
