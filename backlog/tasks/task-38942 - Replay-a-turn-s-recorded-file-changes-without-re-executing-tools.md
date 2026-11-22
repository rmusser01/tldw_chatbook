---
id: TASK-38942
title: Replay a turn's recorded file changes without re-executing tools
status: To Do
created_date: 2026-11-05 16:09
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
- Docs/Architecture/notes-sync.md
- Docs/Architecture/console-file-authority.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users inspect how an agent changed files during a turn by stepping through the changes that were actually recorded. Make incomplete evidence visible instead of requiring users to reconstruct a sequence from the entire transcript.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can select a completed turn and step through its recorded file-change events in actual recorded order, with source tool/run links.
- [ ] #2 Each event shows the available recorded before/after diff or an explicit missing/truncated/uncertain-evidence state; current file bytes are not substituted as historical content.
- [ ] #3 Replay is read-only: it never re-executes tools, changes files, starts a model call, or performs Undo/restore.
- [ ] #4 Historical paths and content remain subject to current workspace/binding/exclusion authority, with unavailable/revoked sources represented safely.
- [ ] #5 The sequence remains bounded and responsive for large or partially recorded turns and preserves exact turn/session identity through navigation.
- [ ] #6 Targeted and mounted checks cover repeated edits to one file, missing/binary/truncated evidence, uncertain writes, access revocation, and session switching.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
