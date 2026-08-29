---
id: TASK-23144
title: >-
  Bare ChatScreen shells miss the Library-activity controller and 46 tests die
  in setup
status: Done
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - console
priority: high
dependencies: []
---

## Description

46 tests fail with `AttributeError: 'ChatScreen' object has no attribute '_library_activity'` —
every test in `Tests/UI/test_console_citation_sources.py` (41) and
`Tests/UI/test_console_composer_menu.py` (5). Production is correct: the controller is installed by
`build_console_controllers`, which only runs from `ChatScreen.__init__`, and these tests build
shells via `ChatScreen.__new__(ChatScreen)`. They die during **setup**, before asserting anything.

The sharp part: this is the exact failure mode `Tests/Architecture/test_bare_chat_screen_shells_wire_the_fleet.py`
was written to ratchet (TASK-21381, which fixed 115 such failures across 8 files). The guard is an
AST scan hard-coded to look for `stub_fleet_controller` only, so it cannot see a **second**
controller entering the same kwargs build. Widening the guard is the durable half of this task —
without it the next controller added to that build repeats this.

## Acceptance Criteria

- [x] Both files pass, with bare shells wiring the Library-activity controller through a shared stub
- [x] The architecture ratchet asserts that **every** controller the chat-store wiring reads is
  stubbed — not just the fleet controller — so a newly added controller fails at the guard rather
  than in dozens of unrelated tests
- [x] The widened guard is proven by a negative control (removing a stub makes it fail)

## Evidence

Installed at `tldw_chatbook/UI/Console_Modules/wiring.py:751`. The setup chain that dies:
`screen._console_chat_store = ...` -> `_console_runtime().set_chat_store()` ->
`_ensure_console_chat_controller` (`chat_screen.py:5008`) -> `chat_screen.py:5116`
`"_library_provider_factory": self._library_activity.build_provider`. Shells built at
`Tests/UI/test_console_citation_sources.py:480`. Guard's blind spot: `_calls_fleet_stub`,
`Tests/Architecture/test_bare_chat_screen_shells_wire_the_fleet.py:87`.

Introduced by `d8d5f9f2b1` (2026-08-27) "feat(console): capture and review minimized Library
activity (TASK-19900.5) (#2154)", which updated 3 UI test files but not these two.

## Implementation Plan

1. Trace the setup death to the code that actually reads the controller (the filing's chain names
   `_ensure_console_chat_controller`; confirm before building a guard around it).
2. Add `stub_library_activity_controller` to `Tests/UI/console_controller_stubs.py`, following the
   existing message/image/fleet pattern exactly: every constructor callable defaults to a raiser.
3. Wire it into every bare shell the ratchet's scan reports, not just the two filed files.
4. Widen `Tests/Architecture/test_bare_chat_screen_shells_wire_the_fleet.py` from "the fleet
   controller is stubbed" to "every controller the store setter reads is stubbed", with the
   required set DERIVED from production rather than listed.
5. Prove the widened guard with a negative control in both halves (a fixture missing its stub; a
   controller missing from the mapping), restoring by edit, never by checkout.
6. Full runs of the touched files, `Tests/Architecture/`, and `./scripts/preflight.sh`.

## Implementation Notes

Both halves done. The 46 filed failures are green, and the guard that should have caught them is no
longer able to be blind to a third controller.

**The chain is one step shorter than the filing says.** `screen._console_chat_store = ...` never
reaches `_ensure_console_chat_controller`: `ConsoleRuntime.set_chat_store` is a plain slot write.
The read happens on the way *in*, in `_console_runtime()` -> `ensure_console_runtime(view=self)` ->
`ConsoleRuntime.attach_view` -> `_bind_view_hooks` -> `ChatScreen.console_view_hooks()`, whose
`_library_provider_factory` entry is `chat_screen.py:5116`. Same line, different caller — and it
matters, because a guard written around `_ensure_console_chat_controller`'s kwargs would have
demanded `_retrieval` and `_session` stubs that this path never touches.

**The stub.** `stub_library_activity_controller` mirrors its three siblings: seven constructor
callables, all defaulting to raisers, plus the `app_instance` snapshot assert with the `NO_APP`
escape. Only the *attribute* `build_provider` is read on the setup path (the factory is stored for
later, never called), so raisers everywhere is exactly enough — and it earned its keep immediately:
`test_zero_only_count_cache_does_not_refresh_unchanged_transcript` really does reach
`sync_transcript` -> `ensure_store`, and said so by name instead of taking a silent no-op.

**The widened guard** (`Tests/Architecture/test_bare_chat_screen_shells_wire_their_controllers.py`,
renamed from `..._wire_the_fleet.py`). The old guard asked "is `stub_fleet_controller` called?" — a
name it could only have got from the production of its day, which is why a controller added beside
`_fleet` walked past it. The new guard derives the question: `controllers_the_store_setter_reads()`
performs `screen._console_chat_store = None` on a bare shell, records the attribute the
`AttributeError` names, installs a stand-in, and repeats until the assignment succeeds. Nothing
about how production spells the read is encoded anywhere. A **fresh shell each round** is
load-bearing: `_console_runtime()` caches its runtime on the shell, and `attach_view` only re-reads
when the view changed, so re-assigning on the same shell reports success after the first missing
name — the false answer was observed while building this.

The one hand-written thing left is `CONTROLLER_STUBS` (slot -> stub helper; only a person knows
which helper builds which controller), held to set-equality with the derived set in both
directions, per `test_framework_armed_clock_inventory.py`'s `EXPECTED_*` model. A positive control
runs the mapped helpers against a bare shell and asserts the assignment then succeeds, so a mapping
that names the wrong helper cannot pass either.

**The ordering rule** (added in review; Qodo on #2179). The widened scan still collected helper
names from the whole function, so a fixture that assigned `_console_chat_store` *first* and stubbed
afterwards was accepted — and still died in setup, because the setter attaches the view and reads
both controllers as it runs. That is this same defect one level down: a check that looked stronger
than it was. `_runs_before` now credits wiring only where it provably precedes **every** assignment
in the function (each assignment is its own attach). Same scope decides by source order — `with` /
`if` / `for` bodies read top-to-bottom, and the loop case stays right because wiring below the
assignment is late on the first iteration, which is the one that dies. Wiring in an *enclosing*
scope counts only if it precedes the `def`/`lambda` statement, not merely the assignment: nothing
can call a nested function before its `def` has run, but a call site sitting in between would
falsify the weaker test. Everything else — wiring deferred inside a nested function, or in a
sibling scope — is refused, because the order then hides behind a call site the scan does not
follow; being wrong that way costs one hoisted line, and being wrong the other way is what let 46
tests die in setup. The `_console_runtime_ref` escape hatch is held to the same rule, since
`_console_runtime()` only honours a ref that is already there. The rule proves *order*, not
reachability — a stub above the assignment but inside an `if` still counts.

**Negative controls** (all restored by edit):

- Moved `_video_action_screen`'s two stub calls below its assignment; the ratchet went red with
  `stub_fleet_controller -- called at line 88, which is not provably before the
  _console_chat_store assignment at line 87`, and the fixture itself then died with the original
  `AttributeError: 'ChatScreen' object has no attribute '_library_activity'` — the setup death the
  guard now predicts statically. Six permanent controls pin the rule in both directions
  (`test_the_scan_*`), mutation-tested: forcing `_runs_before` to always-True reds the four
  ordering controls, always-False reds the two acceptance controls. `_scan_test_tree()` output is
  byte-identical to the pre-rule baseline, so no correct fixture is newly flagged.
- Dropped `stub_library_activity_controller` from `_bare_promote_screen`; the ratchet went red with
  `Tests/UI/test_console_composer_menu.py::_bare_promote_screen (missing:
  stub_library_activity_controller)`.
- Dropped `_library_activity` from `CONTROLLER_STUBS`; the derivation went red with `read by
  production but unmapped here: ['_library_activity'] -- _library_activity (built by
  build_console_controllers)`, and the positive control failed with the original
  `AttributeError: 'ChatScreen' object has no attribute '_library_activity'`.

**Scope.** The filing named 2 files; the ratchet's own scan finds 11 store-setting shells, and 6 of
them were dying the same way (`Tests/Chat/test_console_generation_actions.py` 40 red,
`Tests/UI/test_console_live_work_handoffs.py` 4, `Tests/Chat/test_console_video_actions.py` 4,
`Tests/Chat/test_console_attachment_riders.py` 2, `Tests/UI/test_console_rag_settings_modal.py` 1).
All are fixed, because the widened guard cannot be green while they are not. Two reds unmasked by
the repair are NOT this root cause and are recorded rather than absorbed:
`test_console_citation_sources.py`'s transcript double needed `selected_message_id` /
`set_fork_eligibilities` / `set_model_thinking_visible` from `5d9b4bec5a` (#2152) — same defect
TASK-23146 files against a *different* file, fixed here only because AC#1 demands this file be
green; and `test_console_generation_actions.py` is left at 15 red on `_raw_cli` from `0f67f3b952`
(#2151), an unfiled dev red on `handle_console_message_action`, a path the store setter never
takes, so the guard correctly does not require a stub for it.
`Tests/UI/test_console_live_work_handoffs.py` keeps 4 of its original 8 reds (3 Watchlists
destination tests whose screen fails to load, 1 rail-structure assertion) — also pre-existing and
unrelated.

Verified: `Tests/UI/test_console_citation_sources.py` 41 failed/14 passed -> 55 passed;
`Tests/UI/test_console_composer_menu.py` 5 failed/33 passed -> 38 passed; the widened ratchet 10
passed with every negative control exercised; `Tests/Architecture/` 375 passed / 3 failed, all
three pre-existing (they read only `tldw_chatbook/` and `test_screen_size_ratchet.py`, neither
touched here); `./scripts/preflight.sh` all checks passed.

Modified: `Tests/UI/console_controller_stubs.py`,
`Tests/Architecture/test_bare_chat_screen_shells_wire_their_controllers.py` (renamed),
`Tests/UI/test_console_citation_sources.py`, `Tests/UI/test_console_composer_menu.py`,
`Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_console_rag_settings_modal.py`,
`Tests/Chat/test_console_attachment_riders.py`, `Tests/Chat/test_console_generation_actions.py`,
`Tests/Chat/test_console_video_actions.py`.
