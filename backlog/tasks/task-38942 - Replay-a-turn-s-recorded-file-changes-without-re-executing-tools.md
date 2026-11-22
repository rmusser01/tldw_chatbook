---
id: TASK-38942
title: Replay a turn's recorded file changes without re-executing tools
status: To Do
created_date: 2026-11-05 16:09
updated_date: 2026-11-22 16:10
labels:
- console
- review
- tools
priority: low
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- https://code.claude.com/docs/en/plugins/mods/overview
documentation:
- Docs/Architecture/console.md
- Docs/Architecture/agent-runtime.md
- Docs/Architecture/console-file-authority.md
- Docs/superpowers/specs/2026-08-02-agent-change-review-design.md
- tldw_chatbook/Workspaces/change_turn_tracker.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users inspect how an agent changed files during a turn by stepping through the changes that were actually recorded. Make incomplete evidence visible instead of requiring users to reconstruct a sequence from the entire transcript.

What is recorded: Change Review takes one Begin (B) and one End (E) snapshot per root per turn (`Workspaces/change_turn_tracker.py`, "Per-turn B/E snapshot"; each `TurnChangeRecord` holds `baseline_sha`, `end_sha`, change counts and any `tracking_error`). The verified file change is therefore the net B-to-E diff for each file; no file snapshot is taken per tool call. The only ordered sequence is the turn's tool calls, with the arguments the transcript retains (for example `write_file`'s path and content, or a `shell_exec` command, whose file effects are not attributed to it).

Scope: step through the turn's file-related tool calls in transcript order beside each file's net B-to-E diff. A call's arguments are shown as what the agent asked for, never as the file's verified state after that call. Per-tool-call before/after capture is out of scope; it would need its own task and ADR first.

This task does not change `Workspaces/native_git.py` or `Workspaces/change_review_consent.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can select a completed turn and step through its file-related tool calls in transcript order, each linked to its tool call and run, beside the turn's net B-to-E diff for each changed file.
- [ ] #2 Each step shows the call's retained arguments labeled as requested input, not verified file state; each file shows its recorded B-to-E diff or an explicit missing, truncated, binary, or tracking-error state; current file bytes are never substituted as historical content.
- [ ] #3 Replay is read-only: it never re-executes tools, changes files, starts a model call, or performs Undo/restore.
- [ ] #4 Historical paths and content remain subject to current workspace/binding/exclusion authority, with unavailable/revoked sources represented safely.
- [ ] #5 The sequence remains bounded and responsive for large or partially recorded turns and preserves exact turn/session identity through navigation.
- [ ] #6 Targeted and mounted checks cover repeated edits to one file (several calls, one net diff), a `shell_exec` write with no attributed path, missing/binary/truncated evidence, a root with a `tracking_error`, access revocation, and session switching.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
