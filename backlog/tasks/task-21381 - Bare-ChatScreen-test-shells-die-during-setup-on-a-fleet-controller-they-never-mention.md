---
id: TASK-21381
title: >-
  Bare ChatScreen test shells die during setup on a fleet controller they never
  mention
status: Done
assignee: []
created_date: ''
updated_date: '2026-08-24 00:14'
labels:
  - testing
  - test-integrity
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The second-largest failure cluster across the suite, and the only one that spans both the
UI and core halves.

Many Console tests build their screen with `ChatScreen.__new__(ChatScreen)` and hand-set
the few attributes the code under test reads, rather than mounting an app. That shell never
runs `__init__`, so it never receives the sub-controllers the Console decomposition
introduced — including the fleet lifecycle controller.

That stays invisible until a shell touches a seam that reaches one, and one of the most
ordinary lines such a test writes does: assigning the Console chat store is a property whose
setter builds the chat controller's dependencies, and one of those reads through the fleet
controller. So the shell dies while it is still being *set up*, raising an error that names
an attribute the test file never mentions and has nothing to do with the behaviour under
test. Every affected file was written by copying a helper that worked at the time.

The repository already has the right home for the fix — a shared module of controller stubs
whose stated preference is that new controller wiring be added there rather than hand-rolled
per test file. It simply has no fleet entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A bypassed-`__init__` ChatScreen shell can set its Console chat store without dying on a controller the test does not mention
- [x] #2 The shared stub is fail-loud by default in the same way its siblings are: a shell gets working behaviour only for seams its caller wired, and wandering into an unwired one names that seam
- [x] #3 Recovery is measured as a node-level before/after on the exact affected set, and shows no test newly failing
- [x] #4 A latent case that is not currently failing is distinguished from a safe one by evidence, not by whether it happens to be red today
- [x] #5 The pattern cannot silently return: a guard fails on a newly-introduced instance, names the offending function, and is itself proven to fail when the pattern is reintroduced
- [x] #6 The guard encodes the real invariant rather than one preferred call, so an equally safe alternative does not register as a violation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on pinned dev and confirm the seam from the traceback rather than the symptom.
2. Check every member the screen reads through the controller is real, so a raiser-wired
   controller satisfies references without hiding invocations.
3. Add the stub to the shared module, deriving its parameter list from the constructor
   signature so it cannot drift by typo.
4. Wire it at each affected shell, before the assignment that trips the seam.
5. Sweep for files carrying the pattern but not currently failing; verify rather than assume.
6. Add the ratchet guard, then prove it fires.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `stub_fleet_controller` to `Tests/UI/console_controller_stubs.py` and wired it at
every bare-shell helper that assigns the Console chat store, plus an architecture ratchet.

**The seam.** `ChatScreen.__init__` installs `_fleet` through
`Console_Modules/wiring.build_console_controllers`, so a `ChatScreen.__new__(ChatScreen)`
shell never has one. `screen._console_chat_store = store` is a property whose setter calls
`self._console_runtime().set_chat_store(...)`, which builds the chat controller's kwargs,
which reads `self._fleet._console_wake_user_priority` (`chat_screen.py:5577`). The shell
therefore dies during **setup**, with an `AttributeError` naming an attribute the test file
never mentions.

**Why a real controller rather than a mock (AC#2).** All ten `_fleet.<x>` members
`chat_screen.py` reads are real methods on `ConsoleFleetLifecycleController`, so a
controller constructed with raiser callables satisfies every *reference* while still failing
loudly, by name, at any seam a test actually *invokes*. That matches the module's existing
contract for its two sibling stubs. The 33 constructor parameters were derived from
`inspect.signature` rather than transcribed, so the list cannot drift by typo, and unknown
keywords raise `TypeError` rather than silently leaving a seam raising.

**Evidence (AC#3), node-level A/B on pinned dev `fb0a9601e`.**
- The ten files carrying the signature: **122 red before → 9 red after. 113 recovered,
  0 regressions.** The 9 residuals are distinct causes — seven compositor-paint assertions
  in the fleet panel and steering bar, one citation-caching assertion.
- `Tests/Chat` + `Tests/Architecture` after the change: 23 red / 7,140 passed, and **none
  of the 23 is in a file this task touched**.

**AC#4 — the latent cases.** Seven further files carry `ChatScreen.__new__` *and* a store
assignment without a stub. Rather than assume they were safe because they were green, they
were run: six are genuinely safe, and `Tests/Chat/test_console_attachment_riders.py` was
not — it had two failures of exactly this shape that the UI shard list did not contain.
Fixed, 8/8 pass. Total recovery **115**.

**AC#5/#6 — the ratchet.**
`Tests/Architecture/test_bare_chat_screen_shells_wire_the_fleet.py` fails on any function
that builds such a shell and assigns the store without wiring one, naming the offending
function. Mutation-proven: with a reintroduced offender it fails and names it; with the
offender removed it passes. Its allowlist is empty and shrink-only, and a second test
refuses stale allowlist entries so the ratchet cannot quietly stop ratcheting.

The guard's first draft was wrong in an instructive way. It flagged
`Tests/UI/test_console_native_chat_flow.py::_bare_console_screen`, which is green — because
it pre-assigns `screen._console_runtime_ref`, and `_console_runtime()` returns a pre-set ref
verbatim, so the kwargs build never happens. That is an equally valid, arguably cleaner fix.
The guard now encodes the invariant — *do not let the store setter build a runtime this
shell cannot satisfy* — rather than mandating one helper. A guard that fires on a safe
pattern gets allowlisted into uselessness.

Added: `Tests/Architecture/test_bare_chat_screen_shells_wire_the_fleet.py`.
Modified: `Tests/UI/console_controller_stubs.py`, and the shells in
`test_console_generation_actions.py`, `test_console_citation_sources.py`,
`test_console_composer_menu.py`, `test_console_live_work_handoffs.py`,
`test_console_rag_settings_modal.py`, `test_console_video_actions.py`,
`test_console_attachment_riders.py`. (`test_console_h3_image_edit.py` imports the
generation helper, so it was fixed by that one line.)
<!-- SECTION:NOTES:END -->
