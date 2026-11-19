---
id: TASK-41511
title: "Four Console test files are red on dev and on no PR lane, and the skill-discovery capture file runs near its 180 s lane timeout"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Performance/test_console_skill_discovery_genuine_capture.py
- Tests/Chat/test_console_compaction_hold_cardless_sends.py
- Tests/Chat/test_polling_static_update_idempotence.py
- Tests/Chat/test_console_staged_context.py
dependencies:
- TASK-41140
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 15 by the console, console-review and testinfra streams:
- `Tests/Performance/test_console_skill_discovery_genuine_capture.py` (9 cases, about 100 s at `-n 4`, slowest 155.8 s against the 180 s lane timeout) is named in no workflow or census (checked at the stack head); TASK-41140 repaired its three red cases but did not lane it;
- `test_console_compaction_hold_cardless_sends.py` has 4 red nodes at base (TASK-33621.4 area);
- `test_polling_static_update_idempotence.py` has 2 red nodes at base (TASK-40315 area);
- `test_console_staged_context.py::test_staged_context_summary_normalizes_launch_text` expects `&amp;`, but TASK-32802.4 removed the escaping; it is red and in no census.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each file passes under a plain run
- [ ] #2 Each is on a PR lane (or the notes say why not), with its measured time
- [ ] #3 The capture file's slowest case is under 120 s at the lane's flags, or its timeout is raised with evidence
<!-- AC:END -->
