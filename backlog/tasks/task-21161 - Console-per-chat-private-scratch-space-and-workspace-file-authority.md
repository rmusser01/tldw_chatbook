---
id: TASK-21161
title: Console per-chat private scratch space and workspace file authority
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23'
updated_date: '2026-08-23 18:36'
labels:
  - console
  - security
  - workspaces
  - uat
dependencies: []
priority: high
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
- [x] #1 A user can create and send an ordinary Console Chat without a folder
  prompt; the Console describes its local file capability as private scratch
  space rather than as disabled or misconfigured.
- [x] #2 Every live Console session receives an independent owner-only temporary
  scratch directory, and local file operations in one chat cannot list, read,
  overwrite, or infer files from another chat's scratch directory.
- [x] #3 Default Chats expose only their private scratch directory to
  Chatbook-managed built-in and local filesystem operations; neither
  `[console] workspace_root` nor the process working directory grants them
  filesystem authority.
- [x] #4 A named Workspace retains private per-chat scratch space and may add only
  its explicit folder bindings to the existing multi-root built-in tools;
  selected project roots, binding fingerprints, and read-only/read-write rules
  remain enforced for local `fs_*` and Git tools.
- [x] #5 Retained skill-script output and sandbox-fallback agent run logs use the
  owning chat's scratch space instead of the shared global sandbox.
- [x] #6 Closing a chat immediately makes its scratch space unavailable to new
  work, deletes it after outstanding file-operation leases drain, and never
  reattaches it when a saved conversation is reopened; application disposal
  performs best-effort cleanup of all remaining scratch spaces.
- [x] #7 Folder-recovery guidance explains optional Workspace folder access and
  never presents folder binding as a prerequisite for ordinary chatting.
- [x] #8 Existing permission prompts, sensitive-path checks, Workspace binding
  validation, and non-Console tool behavior remain intact; external MCP
  servers, attachments, Library/RAG data, and generated media keep their
  separate authority models.
- [x] #9 Targeted authority, lifecycle, UI, and provider tests pass, including the
  selected-root swap fail-closed regression, and live DeepSeek UAT demonstrates
  ordinary chat, scratch write/read, cross-chat isolation, Workspace-bound
  access, and no user-config mutation.
<!-- AC:END -->

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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a runtime-owned `ConsoleScratchSpaceManager` that lazily creates random
  owner-only roots per live session, generation-fences immutable snapshots,
  leases active file operations, tombstones on close, cleans after lease drain,
  and performs bounded best-effort disposal at app shutdown. Reopening a saved
  conversation receives a fresh root.
- Captured scratch authority once per turn and propagated it through built-in
  file tools, local `fs_*`/Git composition, tool review/dispatch, retained skill
  output, and fallback run logs. Default Chats no longer inherit
  `[console] workspace_root` or cwd. Named Workspaces preserve scratch and add
  only validated explicit bindings; selected-root identity and read/write
  guards remain fail-closed.
- Made folderless project-instruction setup a valid scratch-only send path while
  preserving recovery for a selected binding that disappears or drifts.
  Updated Console, Settings, Workspace-create, change-review, and user-guide
  copy to describe private scratch and optional explicit folders.
- Live DeepSeek UAT at 180×55 verified folderless send, Chat A write/read, Chat
  B isolation, fresh scratch after close/reopen, named Workspace read/write,
  and Default Chat denial of the Workspace path. The real user config SHA-256
  remained exactly
  `4b4a5c250ad439952eea04e41041b2ca576ceb18505ce72ea228bd967ec8315b`.
  Full evidence is in
  `Docs/UAT/2026-08-23-console-deepseek-private-scratch.md`.
- Independent review found that built-in path metadata and local errors could
  expose the opaque scratch locator to model history and persisted run logs.
  Provider-boundary redaction now emits relative scratch-owned paths without
  changing explicit Workspace-path behavior; success, error, and real run-log
  regressions cover the fix.
- A broad UI diagnostic found ten Workspace-create Pilot failures. Untouched
  latest `dev` reproduced all ten because the bare harness omitted the modal's
  production CSS. Loading `WorkspaceCreateModal.BUNDLED_CSS` restored real
  geometry and made the complete modal module pass 23 / 23; the incident is
  recorded in `backlog/docs/lessons-testing-evidence.md`.
- Final post-rebase verification: 388 targeted authority/lifecycle/provider/
  artifact tests passed; 56 focused mounted Console/Workspace UI tests passed;
  Ruff and Python compilation passed for every changed Python file;
  `git diff --check` and the diff secret-pattern scan were clean. A full
  repository sweep was not run, per repository policy.
- ADR: implemented and linked
  `backlog/decisions/081-console-per-chat-private-scratch-space.md`; no further
  ADR was needed for the review fixes because they directly enforce ADR-081's
  existing non-persistence boundary. No dependencies, schema, license, or
  external service contracts changed.
- Latest `dev` introduced a duplicate `TASK-21161` after this task's earlier
  add commit. Following the repository's older-arrival rule, the later
  model-catalog startup-order task and its complete reference slice were
  renumbered to `TASK-21163`; the duplicate-ID guard passes.
<!-- SECTION:NOTES:END -->

## Design Records

- ADR required: yes
- ADR path: `backlog/decisions/081-console-per-chat-private-scratch-space.md`
- Reason: this task changes filesystem authority, temporary-data ownership,
  cross-thread teardown behavior, and long-lived Console/Workspace semantics.
- Design spec:
  `Docs/superpowers/specs/2026-08-23-console-per-chat-private-scratch-space-design.md`

## Baseline Evidence

- Live UAT began on the then-current `origin/dev` at
  `ae817fefed519921d7da5047e22634756337fc34`. Before closeout the branch was
  rebased cleanly onto latest `origin/dev` at
  `be0a1694696b7b2296dcb79017696dd79c56f677`; the intervening upstream change
  was isolated to the Library UI, and fresh post-rebase targeted gates passed.
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
