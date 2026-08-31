---
id: TASK-25905
title: 'Raw shell: unbypassable hardline command floor'
status: To Do
assignee: []
created_date: '2026-08-31 15:09'
updated_date: '2026-08-31 15:11'
labels:
  - security
  - tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The raw-shell approval card is the only thing standing between the agent and any command, and once a session grant is given nothing re-checks. Verified on origin/dev: Tools/raw_cli_executor.py:144 validate_raw_cli_request checks caller identity, shell name, size, timeout and cwd but never inspects what the command does; Agents/raw_shell_tool_provider.py:291 gates on permission state alone; and once approve_session is granted (raw_shell_tool_provider.py:48) subsequent commands run unreviewed for the rest of the Console session. A named grep for rm -rf, hardline, mkfs and fork bomb across tldw_chatbook returns no guard hits. Hermes runs an unbypassable hardline list plus a sudo-stdin guard before even its own yolo mode. This is a floor, not a replacement for the approval card.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A small set of catastrophic command shapes (root recursive delete, mkfs, dd to a block device, fork bomb, shutdown) is refused outright, before any permission state or session grant is consulted
- [ ] #2 The floor also applies to commands issued under an active approve_session grant - verified by a test that grants a session then attempts a hardline command
- [ ] #3 Detection is resistant to trivial obfuscation (quoting, variable indirection, whitespace padding) - adversarial cases are in the tests
- [ ] #4 A refusal states plainly which rule fired and is distinguishable from a user denial in the model-facing result
- [ ] #5 The floor is not user-configurable off; any user-supplied deny list is additive to it
- [ ] #6 False-positive safety: a corpus of ordinary developer commands (git, npm, pytest, rm of a project file) is asserted to pass unaffected
<!-- AC:END -->
