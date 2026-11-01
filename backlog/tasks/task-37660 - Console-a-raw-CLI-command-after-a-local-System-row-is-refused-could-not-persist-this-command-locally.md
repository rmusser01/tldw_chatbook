---
id: TASK-37660
title: >-
  Console: a raw-CLI command after a local System row is refused 'could not
  persist this command locally'
status: To Do
assignee: []
created_date: '2026-10-29 16:05'
labels:
  - console
  - raw-cli
dependencies: []
references:
  - >-
    backlog/tasks/task-33622.9 -
    Console-a-refused-raw-CLI-command-doubles-itself-in-the-composer-and-Run-is-offered-while-raw-CLI-is-unarmed.md
  - backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Impact.** With raw CLI armed, a `! ` command is refused with "Raw CLI could not
persist this command locally. The exact draft was restored." whenever the chat's
active leaf is a local System row (for example the refusal row an empty `! ` leaves).
The command never runs, and the only way forward is to send something else first.
TASK-33622.9's review found this on dev 64579cce2c and asked for it to be filed
separately; it was out of that task's scope.

**Reproduced live 2026-10-29** on the TASK-33622.9 branch (built on dev 38c9d2edcf,
160x45, Anthropic haiku through the review proxy, raw CLI unlocked and armed in
place): `! pwd` ran (Tool row, exit 0) but the transcript also gained a System row
"Raw CLI persistence was incomplete."; then `! ` + Enter left the System row "Raw CLI
command is empty. The exact draft was restored."; then `! pwd` + Enter was refused
with "Raw CLI could not persist this command locally. The exact draft was
restored." The draft came back exactly once ("! pwd"). Nothing about it reached the
app log. Evidence: `tools/evidence/bd5/U11/34-fix-rawcli-enter-run.*`,
`35-fix-rawcli-empty-refusal.*` and `36-fix-rawcli-run-after-system-row.*` in the
Console UX review session's tools directory.

The refusal comes from `refuse_for_persistence_error` in
`UI/Console_Modules/raw_cli.py` (`ConsoleRawCliController._execute`); the first
run's "persistence was incomplete" row from `report_persistence_error` in the same
method. Neither path logs which persistence step failed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With raw CLI armed, a `! ` command whose chat's active leaf is a local System row or a raw-CLI Tool marker runs and is persisted like any other raw command
- [ ] #2 A raw command that does run never leaves a "Raw CLI persistence was incomplete." row unless a persistence step really failed, and that failure is logged with the failing step (type and message only, no frame locals)
- [ ] #3 A test on the real ChatScreen and controller runs a raw command after a System row and after a raw-CLI marker; it fails on the pre-fix code
<!-- AC:END -->
