---
id: TASK-2852
title: Library evidence handoff can land in a locked Console with no receipt
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 06:32'
labels:
  - library
  - search-rag
  - console-handoff
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-04, observed at dev `6ffa56516`, fresh profile with no provider
configured).

Search/RAG → select evidence → "Use in Console" navigated to Console's locked onboarding
("Get started / Composer unlocks after setup") with zero receipt of the selection — no staged-
evidence chip, no toast, no trace. The flagship Library→Console handoff silently ate the user's
selection and stranded them on a setup screen that never mentions it.

The staged-evidence strip DOES exist on a configured Console (shipped in PR #1320); this is the
unconfigured edge: the handoff is neither gated nor warned when Console cannot accept work.
Re-verify at current dev before implementing (RAG-truth PR #1385 merged since observation).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When Console is locked (setup incomplete/no provider), "Use in Console" either warns before navigating (naming that evidence is saved and what unlocks it) or is disabled with that reason at the button
- [x] #2 If navigation proceeds, the locked Console surface shows a visible receipt that Library evidence is staged and will be usable after setup
- [x] #3 The configured-Console path is unchanged (staged-evidence strip still appears; regression-covered)
- [x] #4 Live TUI verification on a fresh profile confirms the chosen behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the repro at HEAD on a fresh scratch profile (no provider): confirm the RAG-truth PR #1385 merge did not change the symptom.
2. Trace the handoff seam: library_screen.py::_stage_library_rag_result_in_console -> app.open_console_for_live_work -> HandoffChannel.CONSOLE_LIVE_WORK -> ChatScreen._pending_console_launch_context -> the staged-evidence strip (PR #1320). Confirm the strip DOES update even while locked, but is visually hidden behind ConsoleSetupModal's opaque overlay (mode == "card").
3. AC#2 (receipt on the locked surface): reuse the existing seam (the launch context PR #1320 already reads) rather than a new channel. Add a pure receipt-text builder next to ConsoleLiveWorkLaunch, render it as a new Static inside ConsoleSetupModal (visible only while blocking and non-empty), wire it from ChatScreen._sync_console_setup_modal.
4. AC#1 (pre-nav notice): add a reusable pure predicate (console_setup_is_blocking) beside build_console_setup_card_state so Library and Console share one source of truth for "would this land on the blocking card". Library computes it from app_config (best-effort, not the freshest-config reload Console itself uses) and calls app.notify() before navigating when true; navigation still proceeds (advisory, not a gate).
5. AC#3 regression: do not touch the staged-evidence strip or its state builder; add regression tests asserting it renders unchanged once Console is configured.
6. TDD: write tests first for the pure predicate/receipt builder, the widget-level notice rendering, the end-to-end locked/configured screen tests, and the Library pre-nav notify tests -- then implement.
7. Live-verify both a fresh (locked) and a dummy-key (configured) scratch profile in tmux, per the plan's Global Constraints.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-verified the repro live at HEAD (dev merge of PR #1385) on a fresh scratch profile: Library
Search/RAG -> select evidence -> "Use in Console" still lands on the locked Console with the
blocking "Get started" setup-modal overlay covering the whole workbench (rail + transcript +
composer). The staged-evidence strip (PR #1320) DOES update -- `_pending_console_launch_context`
is set correctly -- but ConsoleSetupModal's opaque overlay (mode == "card") visually covers it,
so the repro held unchanged.

Implemented the plan's (a)+(b) combination, reusing PR #1320's existing seam rather than a new
notification channel:

(a) Locked-Console receipt (AC#2): `console_setup_staged_receipt(launch)` in
`Chat/console_live_work.py` builds a one-line receipt ("<source> evidence staged - finish
provider setup to use it.") from the SAME `ConsoleLiveWorkLaunch` the staged-evidence strip
already reads -- generic across every "Use in Console" caller (Library, Watchlists, Schedules,
Artifacts, ...), not Library-specific, since they all funnel through the same
`open_console_for_live_work` -> `CONSOLE_LIVE_WORK` handoff -> `_pending_console_launch_context`
seam. `ConsoleSetupModal` gained a `staged_evidence_notice` param on `sync_card_state()` and a
new Static line (`#console-setup-modal-staged-notice`, new CSS rule in `_agentic_terminal.tcss`,
regenerated bundle) shown only while the card is actually blocking. `ChatScreen._sync_console_
setup_modal` wires it from `self._pending_console_launch_context`.

(b) Pre-navigation notice (AC#1): `console_setup_is_blocking(readiness, has_model,
first_send_completed)` in `Chat/console_onboarding_state.py` delegates to `build_console_setup_
card_state` (one source of truth, not a hand-rolled copy of its branches) so Library and Console
can never disagree about "would this land on the blocking card". `LibraryScreen._console_setup_
would_block()` rebuilds the same readiness/model/first-send inputs from `app_config` (a
best-effort snapshot read, not Console's freshest-config reload -- this is an advisory hint, the
receipt in (a) is the source of truth) and `_stage_library_rag_result_in_console` calls
`app.notify(LIBRARY_RAG_USE_IN_CONSOLE_LOCKED_NOTICE, severity="information")` before navigating
when true. Navigation still proceeds either way -- advisory, not a gate, since the evidence
really is staged and usable once setup completes.

AC#3 (configured-Console regression): untouched code path, regression-covered by a new
screen-level test plus a live-TUI pass with a dummy `[api_settings.openai] api_key` -- the
staged-evidence strip and inspector-rail Sources card render exactly as before PR #1320 shipped
them; no real send was issued.

Tests (TDD: written first, watched fail via a save-diff/git-checkout/git-apply revert-restore
cycle against the six touched implementation files, confirmed pass after restore):
- Tests/Chat/test_console_onboarding_state.py: 4 new tests for `console_setup_is_blocking`.
- Tests/UI/test_console_setup_card_fit.py: 3 new widget-level tests for the staged-notice line.
- Tests/UI/test_console_live_work_handoffs.py: 2 new pure tests for `console_setup_staged_
receipt` + 2 new end-to-end screen tests (locked receipt, configured-path regression).
- Tests/UI/test_product_maturity_gate16_library_search_rag.py: 2 new tests for the Library
pre-nav notify (locked warns, configured stays silent).
Full targeted suite green: 65 (console_live_work_handoffs) + 65 (gate16) + 24
(onboarding_state+setup_card_fit) + 110 (three sibling Console suites, regression check) + 32
(rag handoffs + shell RAG/console subset) all passing; `Tests/Library --collect-only` sweep:
1077 collected, 0 errors.

Live TUI verification (both required profiles, per the task's constraints):
- Fresh profile (locked): created a real note via the UI, ran Search/RAG, selected the evidence,
  pressed "Use in Console" -- landed on the locked "Get started" card now showing "Library
  Search/RAG evidence staged - finish provider setup to use it." directly under the title.
  Screenshot-equivalent captured via tmux capture-pane.
- Dummy-key profile (`[api_settings.openai] api_key = "sk-test-dummy-not-real"`): same flow
  landed on a fully unlocked Console with the staged-evidence strip ("Staged for next send - 1
  source / Incident Review Task 2852 - note") and inspector-rail Sources card rendering exactly
  as PR #1320 shipped -- no real send was issued.
- The pre-navigation toast (AC#1) could not be reliably caught in a tmux screenshot across three
  separate app sessions and an 80ms-granularity poll (the App-level notification and the
  screen-swap both settle inside the same sub-100ms window tmux capture-pane cannot straddle);
  its correctness is instead pinned by the Textual `run_test()`-harness tests above, which mount
  the real screen, press the real button, and assert on the real `app.notify()` call -- a more
  precise check than a screenshot race for a transient toast.

Files changed: tldw_chatbook/Chat/console_onboarding_state.py, tldw_chatbook/Chat/console_live_
work.py, tldw_chatbook/Widgets/Console/console_setup_modal.py, tldw_chatbook/css/components/
_agentic_terminal.tcss (+ regenerated tldw_chatbook/css/tldw_cli_modular.tcss),
tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/UI/Screens/library_screen.py,
tldw_chatbook/Library/library_rag_state.py, plus the four test files listed above.

Docs: no Docs/User_Guide/ page documents this handoff's copy specifically (the Console and
Library pages describe the feature at a level this change doesn't affect); none updated.
<!-- SECTION:NOTES:END -->
