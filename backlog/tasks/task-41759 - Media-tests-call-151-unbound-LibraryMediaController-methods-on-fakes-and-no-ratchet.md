---
id: TASK-41759
title: "Media tests call 151 unbound LibraryMediaController methods on fakes, and no ratchet holds the count"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
priority: low
references:
- Tests/Architecture/test_screen_size_ratchet.py
- Tests/UI/test_library_multiselect_media.py
- Tests/UI/test_library_media_reader_flow.py
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The media series' rulings counted `LibraryMediaController.<method>(fake)` as "retargeted to the media controller", because the real constructor takes the screen plus about 74 lambdas. The rulings named the cost (fake coupling moves from the screen class to the controller class) and the follow-up: "a typed construction seam, after 3.8". The TASK-36952.3.8 report repeats it as "a typed construction seam or a LibraryMediaController unbound-call pin".

At dev 908e9840aa (2026-11-18), an AST count over Tests finds 151 calls of a `LibraryMediaController` instance method through the class with a fake first argument, in 9 files:
- `test_library_multiselect_media.py`: 62;
- `test_library_media_reader_flow.py`: 44;
- `test_library_media_trash.py`: 21;
- `test_review_set_walker.py`: 13;
- `test_library_crit9_media_reader.py`: 6;
- 5 more across four other files.

`_LIBRARY_UNBOUND_TEST_CALLS` (511) ratchets the same shape for `LibraryScreen`. Nothing ratchets it for the controller, so the count can grow freely. The fakes are `SimpleNamespace` objects that break whenever a body gains a seam. Three `test_library_multiselect_media.py` fakes lack `_library_media_editor_dirty` and are red on dev for that reason.

The constructor still takes 73 lambdas (`library_screen.py:3048`). 30 of them are accessors and setters for single screen fields, and they are what makes a real controller hard to build in a test.
AC #3 also covers the 30 field accessors and setters (the Notes ones among them) that TASK-41757 leaves out of its callback scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A ratchet pins the number of unbound LibraryMediaController method calls in Tests at its measurement and fails when it rises
- [ ] #2 A media test builds a real LibraryMediaController from typed test doubles, with no LibraryScreen
- [ ] #3 The controller reads the screen state it needs through typed state objects rather than per-field accessor and setter lambdas, or the notes record why a field stays
- [ ] #4 The unbound-call count is below 151 at close and the ratchet is re-pinned to it
<!-- AC:END -->
