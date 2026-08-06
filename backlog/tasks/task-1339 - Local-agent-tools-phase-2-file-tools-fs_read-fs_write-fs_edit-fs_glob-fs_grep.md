---
id: TASK-1339
title: >-
  Local agent tools phase 2: file tools
  (fs_read/fs_write/fs_edit/fs_glob/fs_grep)
status: Done
assignee: []
created_date: '2026-08-05 05:09'
updated_date: '2026-08-05 05:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md (phase 2). Plan: Docs/superpowers/plans/2026-08-04-local-agent-tools-phase2.md. ADR: backlog/decisions/032. Builds on task-1338 (phase 1, fs_list pilot). NOTE: fs_list already landed in phase 1 and is out of scope here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 fs_read pages line-numbered output with offset/limit and refuses binary files
- [x] #2 fs_write creates/overwrites files confined to workspace root with mutates risk tag
- [x] #3 fs_edit performs unique-match replacement with ambiguity errors and replace_all
- [x] #4 fs_glob and fs_grep search the workspace with result caps
- [x] #5 Legacy ReadFileTool/WriteFileTool delegate to the shared cores with unchanged legacy behavior
- [x] #6 Local deny/timeout outcomes are audit-recorded
- [x] #7 Tools remain reachable via find_tools/load_tools past the direct-disclosure threshold
- [x] #8 All new tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-04-local-agent-tools-phase2.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented on branch `feat/local-agent-tools-p2` (stacked on phase 1, PR #1352) via subagent-driven development with per-task spec + quality review.

- `Tools/local_tool_impls.py`: five new sync cores — `read_file` (1-based line numbers, offset/limit paging, NUL-sniff binary refusal, empty-file notice), `write_file` (parent-must-exist, encode-before-write), `edit_file` (unique-match or replace_all, count-bearing ambiguity errors, identical-strings refusal, CRLF-preserving `newline=""`, encode-before-write, non-UTF-8 read wrapped in LocalToolError), `glob_files` (mtime-desc, `..`-escape lexical guard, workspace-relative rendering), `grep_files` (content/files/count modes, escaping-symlink skip via resolve-then-compare).
- `Agents/local_tool_provider.py`: specs for all five; `fs_write`/`fs_edit` carry `tags=("mutates",)` (risk-floored asks); `record_decision` audit seam (refusals only, MCP parity: kill-switch/deny → "denied", timeout/no_callback → "denied-timeout", initiator="agent"), never-raise guarded, wired in `_compose_local_provider` to `service.record_tool_decision` under `local:__local__`.
- `Tools/file_operation_tools.py`: legacy ReadFileTool/WriteFileTool/ListDirectoryTool refactored onto a `_WorkspaceFileTool` base delegating confinement to `resolve_workspace_path` (default root cwd at execute time); rendering/I-O deliberately unchanged (legacy structured dicts don't map to the string cores — plan's minimal-diff fallback). Their previously broken `validate_path(path, "file")` confinement was NOT preserved — the tools were effectively unusable before.
- `AGENTS.md`: Tool Calling section rewritten for the provider seam; stale `Coding_Window.py` reference fixed.
- Tests: 100+ new — per-core units, Hypothesis properties (sound `str.replace` oracles, atomicity-on-refusal, confinement via `is_relative_to`), 18-test legacy characterization suite, e2e integration (find/load path over a padded 9-entry registry, 8-entry direct-disclosure boundary via `initial_disclosure`, allow-state zero-approval, deny-payload symmetry).

Review-driven hardening beyond the plan: CRLF fidelity (Hypothesis falsifier), sound property oracles (self-overlapping needles), encode-before-write (file-corruption path on unencodable payloads), grep symlink confinement, glob `..` escape guard.

Deviations: legacy delegation is confinement-only (not full I/O) per the plan's minimal-diff rule; local `no_callback` audits as "denied-timeout" (matches the pinned local refusal copy) vs MCP's "denied" — documented in the provider.

Final whole-implementation review: Ready to merge; all 8 ACs verified. Worktree runs: 302 passed (Agents+Tools), 1064 passed (Chat), 313 passed (Utils+config) — only the two known pre-existing base failures.
