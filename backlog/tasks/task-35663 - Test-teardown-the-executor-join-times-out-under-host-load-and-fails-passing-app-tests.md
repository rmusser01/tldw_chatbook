---
id: TASK-35663
title: >-
  Test teardown: the executor join times out under host load and fails passing
  app tests
status: To Do
assignee: []
created_date: '2026-10-19 18:55'
labels:
  - tests
  - flaky-test
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After an async test that built an app, the retire_test_app_owners fixture in Tests/conftest.py joins the event loop's default executor with a timeout of WORKER_CANCELLATION_GRACE_SECONDS (3 s), so a native callback a cancelled worker left running cannot outlive config reset or the DB drain. On a heavily loaded host the join itself can exceed 3 s. Teardown then raises "Native app workers remain uncertain; config retirement refused", and a test whose body passed ends as an ERROR, so the plain pytest command exits 1.

Seen on 2026-10-19 while verifying TASK-34751. At host load 60-100 on 14 cores it hit dev ce4aa4a231's unchanged Study contract test (Tests/UI/test_study_flashcards_real_service_contract.py) as well as the branch's new Study tests. The review of TASK-34751 saw about half of its local runs end this way at load ~90. It has not been seen in CI yet. If it starts to, it will fail unrelated PRs in the merge queue, as the TASK-34751 flake did.

It is not yet known whether the slow joins are slow shutdowns of finished work or native work that really is still running.

A second, more repeatable case was seen later the same evening (dev 8136af6f6f, load 60-80): `Tests/UI/test_study_controls_geometry.py::test_flashcards_tab_shows_deck_picker_and_card_editor_controls[size0]`, the first app test in its file, ended as this ERROR with all 6 test bodies passing, on dev and on the TASK-34751 branch alike (dev passed once at lower load). Being first in its file may matter: whatever the first app of a process starts in the default executor may be what the join waits for.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Under heavy host load, it is shown with evidence which executor work the timed-out join was waiting for: finished work shutting down slowly, or a native callback still running
- [ ] #2 A test whose app workers have all finished no longer fails at teardown only because the executor join was slow under load
- [ ] #3 A native worker that is really still running still fails the test with the custody error, and a test pins that
<!-- AC:END -->
