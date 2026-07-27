---
id: TASK-1010
title: >-
  Three worker components are dead or broken on first call -- decide whether to fix or delete
status: To Do
assignee: []
created_date: '2026-07-27 12:30'
labels:
  - ui
  - dead-code
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while auditing `@work(thread=True)` async workers for TASK-981. Three components are not merely untidy — each is broken in a way that proves nothing exercises it. They were reachable only through code paths that never run, which is why the defects survived.

**1. `Widgets/Media_Creation/swarmui_widget.py::generate_image` could not have worked.** It called `loop.run_until_complete()` from inside the fresh event loop that `@work(thread=True)` on an `async def` already creates via `asyncio.run()`. Reproduced directly: `RuntimeError: Cannot run the event loop while another loop is running`. TASK-981 converted it to a plain `def`, matching the working pattern used twice elsewhere in the same file — but the widget appears never to be mounted, so the fix is untested in situ. Confirm whether the widget is live; if it is not, delete it rather than carrying a fixed-but-unused component.

**2. `Widgets/multi_item_review_window.py::_generate_analyses_worker` references `app.llm_api_client`, which does not exist.** Its worker is otherwise sound (self-contained awaits, no loop-bound sharing), so TASK-981 left it async. But an attribute that is not defined anywhere means this path cannot have run.

**3. `Subscriptions/textual_scheduler_worker.py::SubscriptionSchedulerWorker` fails Textual's own guard immediately.** `@work` asserts `isinstance(self, DOMNode)`, and this class is not a `DOMNode`, so calling `start_scheduler` raises `AssertionError` on the spot — reproduced. Left unfixed under TASK-981 because the component is deprecated and the ADR-019 migration is underway, but it should not sit in the tree pretending to work.

For each: establish whether anything constructs and uses it. Delete what is dead; fix and add a test for whatever is meant to be live. Do not leave a third state where the code looks maintained but cannot execute.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Each of the three is confirmed live or dead by finding its real construction site
- [ ] Dead components are deleted, including their tests and any registration
- [ ] Live components are fixed and covered by a test that would fail against the broken version
- [ ] `SubscriptionSchedulerWorker`'s status is resolved against the ADR-019 migration rather than left ambiguous
<!-- AC:END -->
