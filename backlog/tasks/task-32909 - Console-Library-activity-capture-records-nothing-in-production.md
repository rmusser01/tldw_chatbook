---
id: TASK-32909
title: Console Library-activity capture records nothing in production
status: To Do
assignee: []
created_date: '2026-09-22 01:15'
labels:
  - tier2-review
  - review-crashes
dependencies: []
parent_task_id: TASK-32892
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing the P0 in TASK-32892, and deliberately **not** shipped with it -- it is a second, distinct
defect that needs its own evidence.

`console_runtime.py` (~`:3661`) sets the Library provider factory with `kwargs.update(...)` rather than
`setdefault(...)`, overwriting the factory `UI/Screens/chat_screen.py` maintains.

**The overwrite itself is deliberate and must not be "fixed".** `Tests/UI/test_console_runtime_ownership.py`
classifies `_library_provider_factory` as `"app-owned-domain"` in its `_VIEW_HOOK_OWNERSHIP` table -- the
runtime owns that slot, and changing `update` to `setdefault` would contradict a pinned ownership contract.
Anyone attacking this from the `setdefault` angle will break that test and should stop.

**The real consequence is elsewhere.** `capture_kwargs` -- which produces `activity_attempt_id` and
`activity_sink` -- is the only producer of those arguments anywhere in the tree, and it runs **only** on the
screen's factory. Since the runtime's factory wins, those arguments never reach the provider in production.
So Console Library-activity capture records nothing, silently, on the shipped path.

## Why it was not fixed alongside the P0

The correct fix is for the runtime to build the capture kwargs itself from the store it already owns
(`ensure_chat_store`), preserving the ownership contract. That is a real change to the runtime's
responsibilities and deserves its own test -- and the test that would prove it end to end needs the app
running, which the review's ground rules forbid. Shipping it inside the P0 commit would have meant an
unverified behaviour change riding along with a verified one.

Note this defect lives on a path covered by `Tests/UI/`, which the PR gate does not run -- see TASK-32908.

Source: tier-2 code review 2026-09-21, found while implementing TASK-32892.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The runtime's factory supplies `activity_attempt_id` and `activity_sink` from the store it owns
- [ ] #2 `_VIEW_HOOK_OWNERSHIP`'s `app-owned-domain` classification still holds and its test passes unchanged
- [ ] #3 A test proves activity capture records a real attempt on the shipped Console path
- [ ] #4 The test is gate-free, or it is stated plainly that it can only run in CI
<!-- AC:END -->
