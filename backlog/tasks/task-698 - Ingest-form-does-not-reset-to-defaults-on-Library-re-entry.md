---
id: TASK-698
title: Ingest form does not reset to defaults on Library re-entry
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 05:36'
updated_date: '2026-07-26 18:06'
labels:
  - ingest
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-entering the Library ingest canvas via a nav deep-link is asserted to leave the form equal to a freshly constructed one, and it does not. Investigation shows the screen resets correctly and the assertion is wrong: a mounted canvas can never equal a never-mounted form, because each option widget reports the value it rendered, so the per-type defaults land in the form as soon as the canvas composes. The test therefore measured the mount rather than the reset, and failed on a screen that was behaving. The reset itself discards everything from the previous visit, including options the user moved off their defaults.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Re-entering the ingest canvas leaves no state from the previous visit
- [x] #2 The existing deep-link re-entry test passes
- [x] #3 Whichever field was surviving is identified, and it is clear whether keeping it was intended
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Not a product bug. The reset works; the test was measuring the wrong thing.

What survived re-entry was type_options, holding {'analyze': False, 'chunk': True, 'chunk_size': '1000'} -- the capability schema's own defaults, with chunk_size as the string '1000' because number inputs round-trip through display text. The test's pre-fill never set type_options at all, so those values could not have come from the previous visit: they are generated when the canvas composes and its widgets report what they rendered.

Proved by probing the stronger question the original test never asked. Pre-filling options the user had deliberately changed (chunk_size='500', chunk=False) and re-entering returns them at their defaults ('1000', True), alongside a cleared path. So nothing from the prior visit survives -- not merely the typed text, but user overrides too.

Fixed the test rather than the screen: it now asserts every other field equals a fresh form, and separately that type_options contains only schema defaults. The pre-fill was strengthened to include those changed options, so the test exercises the invariant its own docstring claims ('no stale half-filled form') instead of the weaker text-only version.

Mutation-checked: disabling the reset in the deep-link branch fails the test on path/title/author/keywords. Worth noting type_options is NOT among the differing attributes under that mutation -- the mount re-seeds defaults whether or not the reset ran, which is further evidence it was never reset-driven state. The type_options assertion therefore guards a different regression (a future change that let user values persist through a mount), not the reset.

Files: Tests/UI/test_library_shell.py.
<!-- SECTION:NOTES:END -->
