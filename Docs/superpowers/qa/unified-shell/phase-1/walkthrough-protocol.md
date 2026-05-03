# Phase 1 Shell QA Walkthrough Protocol

Status: active for `TASK-2.1`

Use this protocol before marking shell destinations, primary actions, or recovery states usable. The protocol is intended to be run against the actual Textual app, not only route tables, static docs, or unit-level click handlers.

## Entry Commands

Use one of these entry paths and record which one was used:

```bash
python3 -m tldw_chatbook.app
```

Focused mounted checks that support, but do not replace, manual QA:

```bash
python3 -m pytest Tests/UI/test_master_shell_navigation.py -q
python3 -m pytest Tests/UI/test_shell_destinations.py -q
python3 -m pytest Tests/UI/test_unified_shell_qa_protocol.py -q
```

## Required Walkthrough Scope

For each shell destination or shell-contract workflow, record:

- Whether navigation reaches the expected destination.
- Whether the destination header states the purpose, source authority, and current status.
- Whether primary actions are reachable by keyboard and mouse.
- Whether disabled, blocked, unavailable, pending approval, and recovery states explain the owner, reason, and next action.
- Whether the workflow completes, is honestly blocked with recovery, or fails.
- Whether repeated use remains fast enough for power users.

## Evidence Rules

- Do not count render-only checks as workflow completion.
- Do not count click-event-only checks as workflow completion.
- Do not mark a task verified unless the functional result is recorded.
- Do not hide optional dependency, server, auth, runtime, or policy limits.
- If a workflow is blocked, record the recovery path and severity instead of treating the blocked state as a pass.

## Severity Labels

Use exactly one severity when a defect is found:

- `blocker` - prevents basic use or traps the user.
- `workflow-degradation` - seriously slows or breaks a core workflow but leaves a workaround.
- `recoverability` - blocked/error state exists but recovery copy, ownership, or next action is unclear.
- `polish` - visual or wording issue that does not block completion.

## Output

Create one markdown summary per walkthrough using `walkthrough-template.md`. Store summaries in this directory and link them from `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md` or the relevant Backlog task.
