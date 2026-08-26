---
id: TASK-15970
title: User-wins-ties probe is blind to a live-typed composer draft
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 01:17'
updated_date: '2026-08-14 06:03'
labels:
  - fleet
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3a-2 residue-arc live verification (2026-08-13, scratch profile, real Anthropic): a wake fired straight through a held composer draft. The draft 'drafting my next thought' was typed via real terminal keys after a send, was visibly rendered in the composer pane, and was still present after the wake -- yet the instrumented probe logged 'probe: composer=True draft=""' at attempt time and the wake delivered and stamped ~6s after settle (run 2fd18ecf / conv d6dee392, stamp 2026-08-14T01:00:28Z; second repro run identical). ChatScreen._console_wake_user_priority reads ConsoleComposerBar.draft_text(), which returns the canonical SEGMENTS once _segments_initialized -- the live-typed text was not in them. Task 7 (feat/fleet-autowake @e38e62a2f) verified the same flow deferring for a full 50s, so this regressed between that branch's base and current dev (composer draft/segment plumbing changed in the PR #1554 era). The existing wiring test passes because load_draft writes segments directly -- the same harness-shortcut trap as task-15862's context bug. Note the same blindness likely affects every draft_text consumer, not just the wake probe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The user-wins-ties probe defers a due wake while a draft typed with real keys (not load_draft) is present
- [x] #2 A test drives the probe through real typed input and fails on the blindness before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce RED with real key input through the production on_key path (not load_draft)\n2. Diagnose where typed state lives vs where the probe reads\n3. Fix the probe to read the composer the user actually types into\n4. Mutation-test the fix\n5. Live re-verify the deferral with typed text
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
FIXED in commit 8d8b0c2be; the suspicion was WRONG and the fix landed elsewhere.
The suspected PR #1554-era segment plumbing was verified CORRECT under real key
input (a real-keys harness test passes against unmodified production). The real
mechanism: a navigation issued while a pushed screen sits above the Chat screen
pops the MODAL off the screen stack, leaving the Chat screen resident-but-hidden
with a live controller — the live 'probe: composer=True draft=""' was the HIDDEN
screen's probe reading its own empty composer while the user typed into the
DISPLAYED screen's. Fix: _console_wake_user_priority resolves the composer via
_console_wake_probe_composer — the displayed screen's composer whenever the
displayed screen is a (different) Console screen, its own otherwise, with the
task-15862 active_app-ContextVar lesson applied to self.app resolution. Tests
(Tests/UI/test_console_fleet_wake_hidden_screen.py): real pilot.press keys, real
nav construction of the two-screen state, RED pre-fix on the probe AND on the
outcome (the hidden coordinator's _attempt delivered through the held draft);
[correction 2026-08-14, task-16300: that "real nav construction" built the
two-screen state through a screen-stack LEAK — switch_screen pops only the top,
so navigating under a pushed screen left the old Chat screen resident. The leak
is fixed, so navigation can no longer produce two live Console screens; the two
tests now build the geometry directly with push_screen and stay RED without this
fix (mutation re-run at the fix commit), keeping the probe's cross-screen
resolution pinned as defence in depth. The fix itself is unchanged and still
correct — a Console screen covered by a modal reads its own composer, which is
where the user's draft is.]
mutations M1/M2 killed. Live re-verified on a scratch profile vs real Anthropic:
a draft typed with real keys during the spawning turn held the due wake ~90s
(child done, stamp NULL throughout), clearing it delivered in ~2s. Evidence:
.superpowers/sdd/2026-08-13-supervisor-fleet-pr3a2-autowake/wake-integrity-report.md
+ wake-integrity-frames/. The underlying stack leak is filed as task-16210.
<!-- SECTION:NOTES:END -->
