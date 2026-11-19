---
id: TASK-40281
title: Protect workspace folders from model shell wipes
status: Done
assignee:
  - '@codex'
created_date: '2026-11-11'
updated_date: '2026-11-12T05:27:00Z'
labels:
  - security
  - tools
dependencies: []
references:
  - backlog/tasks/task-25905 - Raw-shell-unbypassable-hardline-command-floor.md
documentation:
  - Docs/superpowers/specs/2026-11-11-workspace-shell-deletion-guard-design.md
  - Docs/superpowers/plans/2026-11-11-workspace-shell-deletion-guard.md
  - backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace chats need to use the raw shell without accidentally deleting or emptying the folders bound to their workspace. Keep ordinary edits and targeted deletions available. The guard covers recognizable local shell commands and applies even when the chat already has a raw shell session grant.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recognizable Bash, PowerShell and cmd commands that delete a local binding, delete a parent containing it, or wipe the binding are refused before a shell launches.
- [x] #2 Every local binding of the chat's owning workspace is protected, including read-only and missing folders. A session grant or earlier one-use approval cannot override the guard.
- [x] #3 Ordinary edits, targeted deletions and user-authored shell commands retain their existing behavior.
- [x] #4 ADR-094 and the architecture documentation explain what the guard recognizes and where static detection stops.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-094 and test the bounded shell command detector.
2. Enforce the detector before approval and launch using live bindings from the owning workspace.
3. Run the targeted shell suites and Ruff checks, then review the diff.

ADR required: yes
ADR path: backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
Reason: This extends the existing model raw shell refusal policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- The Console loads current local folder bindings from the owning workspace during shell review and dispatch. The raw shell executor checks the command against the selected dialect and launches the exact argv it checked.
- Recognition covers bounded Bash, PowerShell, and cmd deletion forms. ADR-094 and `Docs/Architecture/console-file-authority.md` describe the supported forms and limits of static inspection.
- The six targeted Windows suites passed 292 tests and skipped seven POSIX-only cases. An earlier run hit eight subprocess startup deadlines. All eight cases passed on the serial rerun. The Ruff lint gate, undefined-name check, and changed-line formatting passed.
- After rebasing onto dev, the guard uses the shared Console provider composition. The five focused Windows launch and admission tests passed on the rebased branch.
- Independent review caught a native call that could remain active after a final safety refusal. The runtime now settles that refusal and retires pending verifier evidence. The detector also handles literal directory fallbacks, drive casing, all-star Bash globs, PowerShell literal invocation, plain status output between deletes, and literal parent pruning. The focused Windows run passed 182 tests and Ruff.
- Follow-up review covered line continuations in conditional commands and pipelines, possible failure of redirected directory changes, and literal status output. The final parser and provider run passed 119 tests and Ruff.
- CI exposed two startup regressions. The Console now reads the workspace registry directly, and the raw CLI loads the detector on the first guarded request. The branch was rebased onto dev with the upstream Console configuration fixture correction.
<!-- SECTION:NOTES:END -->
