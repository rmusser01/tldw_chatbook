---
id: TASK-42651
title: "A terminated fork child in test_bootstrap_root_cleanup can outlive its 30-second join under load"
status: To Do
assignee: []
created_date: 2026-11-28 16:53
updated_date: 2026-11-28 16:53
labels:
- follow-up
- flake
- tests
priority: medium
references:
- Tests/CI/test_bootstrap_root_cleanup.py
- Tests/real_profile_guard.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen once on 2026-11-27, in round 28's final gate (macOS, 18 CPUs), in the fast PR contract step at load 7-26: `Tests/CI/test_bootstrap_root_cleanup.py::test_owned_root_is_removed[fork-child-terminated]` failed with `assert 'child=None kept=True' == 'child=-15 kept=True'` after 30.05 s. The child script starts a `multiprocessing` fork child, waits for it to signal ready, calls `terminate()` and `join(30)`; the child was still alive after the join. The node passed in the same gate's whole `Tests/CI` run, and 3 of 3 times alone, both on that tip and on dev.

The fork child inherits `remove_sandbox_at_exit`'s SIGTERM handler (`Tests/real_profile_guard.py`), which every pytest session installs: it removes the root only in the owning process, resets the disposition and re-sends the signal to itself. Why the child did not die within 30 s is not known. Round 24 saw a similar macOS loss of SIGTERM around fork and exec (TASK-42470).

The handler runs in every test session, and the test's own comment notes that Chunking's `ebook_chapters` strategy terminates fork children the same way, so this may not be limited to the test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The race is named with evidence (a forced-delay or load reproduction), and the notes say whether the inherited handler affects real test sessions or only this test
- [ ] #2 It is fixed where it lives, and the test still asserts that a terminated fork child ends by SIGTERM while the parent's root is kept
- [ ] #3 A soak under load passes where the unfixed version fails at a measured rate
<!-- AC:END -->
