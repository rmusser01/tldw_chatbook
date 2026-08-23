---
id: TASK-21161
title: Console per-chat private scratch space and workspace file authority
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-23'
labels:
  - console
  - security
  - workspaces
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Ordinary Console Chats must work without choosing or binding a folder. Each live
chat needs its own temporary file scratch space so local file operations cannot
leak files across chats. Only named Workspaces may add access to user folders,
and only through their explicit folder bindings.

Latest `dev` currently mixes three incompatible behaviors: the built-in file
tools always include one shared durable sandbox, the local `fs_*` provider falls
back to `[console] workspace_root` or the process working directory, and the
Console labels Default Chats as `File tools: Off in Default`. This task replaces
those fallbacks with one explicit per-chat authority policy and verifies the
result through targeted automation plus live DeepSeek UAT.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] A user can create and send an ordinary Console Chat without a folder
  prompt; the Console describes its local file capability as private scratch
  space rather than as disabled or misconfigured.
- [ ] Every live Console session receives an independent owner-only temporary
  scratch directory, and local file operations in one chat cannot list, read,
  overwrite, or infer files from another chat's scratch directory.
- [ ] Default Chats expose only their private scratch directory to
  Chatbook-managed built-in and local filesystem operations; neither
  `[console] workspace_root` nor the process working directory grants them
  filesystem authority.
- [ ] A named Workspace retains private per-chat scratch space and may add only
  its explicit folder bindings to the existing multi-root built-in tools;
  selected project roots, binding fingerprints, and read-only/read-write rules
  remain enforced for local `fs_*` and Git tools.
- [ ] Retained skill-script output and sandbox-fallback agent run logs use the
  owning chat's scratch space instead of the shared global sandbox.
- [ ] Closing a chat immediately makes its scratch space unavailable to new
  work, deletes it after outstanding file-operation leases drain, and never
  reattaches it when a saved conversation is reopened; application disposal
  performs best-effort cleanup of all remaining scratch spaces.
- [ ] Folder-recovery guidance explains optional Workspace folder access and
  never presents folder binding as a prerequisite for ordinary chatting.
- [ ] Existing permission prompts, sensitive-path checks, Workspace binding
  validation, and non-Console tool behavior remain intact; external MCP
  servers, attachments, Library/RAG data, and generated media keep their
  separate authority models.
- [ ] Targeted authority, lifecycle, UI, and provider tests pass, including the
  selected-root swap fail-closed regression, and live DeepSeek UAT demonstrates
  ordinary chat, scratch write/read, cross-chat isolation, Workspace-bound
  access, and no user-config mutation.
<!-- AC:END -->

## Design Records

- ADR required: yes
- ADR path: `backlog/decisions/081-console-per-chat-private-scratch-space.md`
- Reason: this task changes filesystem authority, temporary-data ownership,
  cross-thread teardown behavior, and long-lived Console/Workspace semantics.
- Design spec:
  `Docs/superpowers/specs/2026-08-23-console-per-chat-private-scratch-space-design.md`

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a thread-safe, generation-fenced `ConsoleScratchSpaceManager` with
   owner-only random directories, leases, tombstones, a single off-loop cleanup
   worker, and bounded best-effort disposal.
2. Make `ConsoleRuntime` own the manager, capture one immutable scratch snapshot
   in each `ConsoleTurnExecutionContext`, tombstone it before session removal,
   preserve it across ordinary navigation, and dispose it at application exit.
3. Thread the captured root and lease through built-in and local providers so a
   Default Chat never falls back to `[console] workspace_root` or cwd, while
   named Workspace bindings and ADR-069 selected-root guards remain intact.
4. Propagate the same authority through agent review and dispatch, retained
   skill-script output, and run-log fallback/readback without persisting paths.
5. Replace folder-required/disabled copy with private-scratch and optional
   Workspace-folder guidance, then update the Console, Settings, tool, and
   developer documentation.
6. Run the targeted authority/lifecycle/artifact/UI suites and perform live
   DeepSeek UAT from an isolated mode-0600 config copy, verifying the original
   configuration hash is unchanged.
7. Complete task acceptance criteria, implementation notes, ADR links, and Done
   status only after all targeted and live gates pass.

Detailed red-green steps, exact files, interfaces, commands, and commit points:
`Docs/superpowers/plans/2026-08-23-console-per-chat-private-scratch-space.md`.

ADR required: yes

ADR path: `backlog/decisions/081-console-per-chat-private-scratch-space.md`

Reason: the plan implements the accepted filesystem-authority, temporary-data
ownership, provider-boundary, and cross-thread teardown decision.
<!-- SECTION:PLAN:END -->

## Baseline Evidence

- Branch base after refreshing all remote heads, then refreshed again before
  UAT: `origin/dev` at `ae817fefed519921d7da5047e22634756337fc34`.
- Related targeted baseline: 55 passed, 1 failed. The single failure,
  `test_selected_root_swap_fails_closed_before_local_invoke`, is a stale test
  invocation that omits the review hook's required `run_id`; blame shows the
  signature predates the test. Production root-swap enforcement did not run.
- Conflicting in-flight work: open PR #1657 / TASK-16316 requires a folder in
  the Console New Workspace modal. This task does not reuse that policy and
  must not be merged as though the two designs were compatible.
- The refreshed clean-dev adjacent rail sweep initially had 151 passes and
  five failures because those tests still drove the retired outer/body scroll
  owners after the bounded-section migration. Their rendered visibility and
  coordinate-click contracts now drive the current local section viewports;
  the same sweep passes all 156 tests.
