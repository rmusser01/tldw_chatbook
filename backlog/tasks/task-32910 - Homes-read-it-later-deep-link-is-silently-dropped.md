---
id: TASK-32910
title: Home's read-it-later deep link is silently dropped
status: To Do
assignee: []
created_date: '2026-09-22 01:40'
labels:
  - tier2-review
  - review-dead
dependencies: []
parent_task_id: TASK-32899
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while deleting dead code (TASK-32899), and **pre-existing** — not caused by those deletions.

Home's `review_read_later` suggestion emits a navigation context intended to land the user directly on their
saved-reading queue (`UI/Screens/home_screen.py:126`):

```python
return {MEDIA_NAV_CONTEXT_BROWSE_SUBVIEW: MEDIA_BROWSE_SUBVIEW_READ_IT_LATER}
```

The **only** consumer of `MEDIA_NAV_CONTEXT_BROWSE_SUBVIEW` was `MediaScreen.apply_navigation_context`. That
route does not resolve to `MediaScreen` — verified by *running* `resolve_screen_route()` over all 52 route
targets, none of which reaches it — it resolves to `LibraryScreen`, which reads `LIBRARY_NAV_CONTEXT_MODE`
keys instead and ignores this one.

So the user clicks the suggestion, navigates, and lands on the generic Library view rather than their
read-it-later queue. No error, no log line. Repo-wide, the key now appears in exactly three places: its
definition (`Constants.py:87`), its import, and its emission — **zero readers**.

This is a small UX defect with an outsized lesson: the deep link broke when the screen behind the route was
shadowed, and nothing caught it because a producer with no consumer is not a type error, not a test failure,
and not something any existing guard looks for.

## Fix

Translate the intent to whatever `LibraryScreen` actually reads, so the suggestion lands where it says it
will. Then either delete `MEDIA_NAV_CONTEXT_BROWSE_SUBVIEW` or give it a reader — do not leave a constant
that only gets written.

Worth considering as part of TASK-32906's remit: a check for navigation-context keys that are emitted and
never read is narrow, mechanically decidable, and would have caught this.

Source: tier-2 code review 2026-09-21, found while implementing TASK-32899.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The read-it-later suggestion lands the user on the read-it-later view
- [ ] #2 A test asserts the landing view, not merely that navigation occurred
- [ ] #3 `MEDIA_NAV_CONTEXT_BROWSE_SUBVIEW` either has a reader or is deleted
<!-- AC:END -->
