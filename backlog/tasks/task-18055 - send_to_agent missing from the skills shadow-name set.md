---
id: task-18055
title: send_to_agent missing from the skills shadow-name set
status: To Do
assignee: []
created_date: '2026-08-18'
labels: [agents, skills]
dependencies: []
priority: medium
---

## Description (the why)

`Tests/Library/test_library_skills_state.py::test_shadow_name_set_stays_in_sync_with_real_sources`
is red on dev: `send_to_agent` (added by the supervisor-fleet PR 3b work) is
not in `_SHADOWED_BUILTIN_NAMES`, so a skill installed under that name could
shadow the runtime tool undetected.

**This is the guard working as designed, not a regression of it.** TASK-13214
rebuilt it to report every gap across four sources in one failure precisely so
the next addition could not hide behind the last one; this is the next
addition. The test's own message says it must not be accepted as a baseline
failure (task-580), and the fix is one entry.

Noticed during TASK-17755's battery; that arc's diff touches nothing in the
test's import path.

## Acceptance Criteria (the what)

- [ ] `send_to_agent` is covered by `_SHADOWED_BUILTIN_NAMES` (or deliberately
      exempted with a documented reason)
- [ ] The guard passes on a clean dev checkout
- [ ] Any other name added since TASK-13214 closed is covered in the same
      pass — the guard reports all four sources at once, so run it and fix
      everything it names rather than only the one in this title
