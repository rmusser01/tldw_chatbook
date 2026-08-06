---
id: TASK-2314
title: First-run wizard honors early Escape
status: Done
assignee: []
created_date: '2026-08-04'
labels:
  - onboarding
  - ux
  - uat-2026-08-04
dependencies: []
priority: low
---

## Description (the why)

UAT: the first-run wizard advertises "Esc finish later" but silently ignores
Escape until it has fully settled — early keypresses during the opening
seconds are dropped, making the wizard feel frozen.

UAT finding F0a. (Positive to preserve: the explicit "Skip — explore on my
own" one-click path, and the Escape→confirm asymmetry.)

**Investigation correction (review finding I2, post-implementation):** the
UAT's own literal hypothesis above — Escape silently dropped/queued during
an early mount window — was investigated live (tmux, fresh profile,
Python-driven poll-then-press harness) and **was not reproduced**. A single
Escape sent within ~5ms of the wizard's first paint reliably opened the
finish-later confirmation every time; no mount-window drop exists for the
literal first render, so there was never anything to "queue." The real,
adjacent defect that produces the same "feels frozen, Escape did nothing"
symptom is different and is described in AC#1 below and in the
Implementation Notes: a reflexive SECOND Escape (pressed because the first
press gave no immediate visual feedback while several heavy wizard steps
were still settling) lands on the confirmation dialog's own Escape-cancels
binding and silently dismisses it back to "Keep going." That is the
mechanism this task actually fixes.

## Acceptance Criteria (the what)

- [x] A single Escape pressed at any point after the wizard becomes visible
      — including immediately on first paint — reaches the finish-later
      confirmation dialog. (The originally-reported "early Escape is
      silently dropped / queued if mid-mount" mechanism was investigated
      live and NOT reproduced: no mount-window gap exists between the
      wizard's first paint and its Escape binding being active.)
- [x] A reflexive second Escape, pressed within a brief settling window of
      the confirmation dialog's own appearance, does not silently dismiss
      it back to "Keep going" — this is the actual mechanism behind the
      UAT's "frozen" symptom. A genuine, later Escape (after the window
      elapses) still cancels the dialog normally; Escape is never made
      permanently inert.
- [x] Regression tests cover: a single Escape with no settle pause still
      opens the confirmation; a rapid back-to-back double-Escape at first
      render still reaches and holds the confirmation open; the settling
      window expires so a later, deliberate Escape still cancels; a
      Cancel-button click is never swallowed regardless of timing.

## Implementation Plan (the how)

1. Reproduce live (tmux, fresh profile): confirm whether a single Escape
   pressed the instant the wizard's first paint appears is actually lost,
   or whether the "frozen" symptom is caused by something else (e.g. a
   second reflexive Escape landing on the confirmation dialog's own
   Escape-cancels binding while the user is still deciding whether the
   first one registered).
2. Once the real mechanism is identified, fix it at the narrowest point
   that preserves every other documented behaviour: the "Skip — explore on
   my own" one-click path (WelcomeStep, untouched) and the Escape→confirm
   asymmetry (Escape must still open a confirmation, never finish
   immediately).
3. Add a Pilot-based regression test in `Tests/Wizards/
   test_first_run_setup_wizard.py` (the file already hosts
   `test_escape_asks_for_confirmation_instead_of_dismissing`) that presses
   Escape during/adjacent to the wizard's first render and asserts the
   finish-later confirmation flow is reached and not silently reverted.
4. Live-verify in tmux: press Escape immediately on first paint, and again
   as a rapid double-press, and confirm the confirmation dialog is
   reached and stays up.

## Implementation Notes

The advertised binding was never broken. Live reproduction (tmux, fresh
profile, `TLDW_CONFIG_PATH`-isolated) with a Python-driven poll-then-press
harness showed a SINGLE Escape sent within ~5ms of the wizard's first
paint ("Welcome to tldw chatbook") reliably opened the "Finish setup
later?" confirmation every time — no mount-window drop exists for the
literal first render.

**Actual defect, reproduced live:** `ConfirmationDialog` binds Escape to
`action_cancel_dialog` ("dismissing is always the safe outcome", by
design, and correct for every other caller of this shared widget). The
wizard is pushed while several heavy steps are still settling (10
composed steps, the full provider catalog). A user who presses Escape
once and perceives no immediate feedback over that render lag
reflexively presses it again — and that SECOND press lands directly on
the now-topmost `ConfirmationDialog`'s own Escape binding, silently
dismissing it back to "Keep going". Net effect: the wizard is exactly
where it started, with no visible sign anything happened — the reported
"silently ignores Escape ... feels frozen". Confirmed by sending two
Escape presses within 5ms of the wizard's first paint: the dialog closed
itself and the wizard was back on screen, `dialog visible after DOUBLE
immediate escape: False`.

**Fix:** `_SettlingGuardedConfirmationDialog` (new, in
`FirstRunSetupWizard.py`), a thin `ConfirmationDialog` subclass used only
for this wizard's finish-later flow. It overrides the Escape BINDING only
(a distinct action, `cancel_dialog_if_settled`, never
`action_cancel_dialog` itself — the Cancel button's `on_button_pressed`
still calls the base action directly, so a deliberate click stays
instant regardless of timing) to swallow a press landing within 0.5s of
the dialog's own `on_mount`. 0.5s comfortably exceeds a reflexive
double-tap and is comfortably shorter than the time it takes to actually
read the dialog's message, so a genuine second, deliberate Escape after
the window elapses still cancels normally — live-verified both ways in
the same session. The "Escape → confirm" asymmetry (Escape opens a
confirmation rather than finishing immediately) and the "Skip — explore
on my own" one-click path are both untouched.

One existing test (`Tests/UI/test_first_run_wizard_live_contract.py::
test_escape_finish_later_dismisses_and_next_boot_resumes_via_toast`)
asserted the dialog's exact class name (`"ConfirmationDialog"`) as its
premise; corrected to `isinstance(app.screen,
_SettlingGuardedConfirmationDialog)`, which is both true and a strictly
looser/more honest check of the actual contract (a confirm dialog is up).

### Verification

* New tests in `Tests/Wizards/test_first_run_setup_wizard.py` (+4):
  a single Escape with no settle pause still opens the confirmation
  (guards against a fix that swallows early Escapes wholesale); a rapid
  double-press (`pilot.press("escape", "escape")`, back-to-back with no
  pause) still reaches and stays on the confirmation, not reverted; the
  grace window expires (a genuinely later Escape still cancels); a
  Cancel-button click is never swallowed regardless of timing.
* Mutation-verified, both restored byte-exact (md5): (1) reverting
  `action_cancel` to plain `ConfirmationDialog` turned the 3 new
  isinstance-based tests red; (2) unconditionally swallowing every Escape
  in `action_cancel_dialog_if_settled` turned
  `test_escape_still_cancels_the_dialog_once_it_has_actually_settled` red
  without affecting the others (proving each test pins a different half
  of the guard).
* Gates: `Tests/Wizards/` **288 passed**; `Tests/UI/
  test_first_run_wizard_live_contract.py` + `test_product_maturity_
  phase1_first_run.py` + `Tests/Chat/test_console_onboarding_state.py`
  **51 passed, 1 pre-existing failure** —
  `test_speech_step_install_button_visible_at_120x40_without_scrolling`
  times out waiting for a real `ModelArtifactService` layout in this
  sandbox; zero diff against `origin/dev` in this test file and no code
  path from this change reaches `SpeechSetupStep`, so it is unrelated and
  pre-existing.
* Live verification (tmux, fresh profile, 235x52): single immediate
  Escape → dialog appears (`found_at≈3.5s`, escape sent ~5ms later,
  dialog visible). Rapid double immediate Escape (pre-fix) → dialog
  closes itself, wizard shows with no visible sign anything happened.
  Same double-press, post-fix → dialog stays up
  (`dialog visible after DOUBLE immediate escape (post-fix): True`).
  A further Escape after a genuine 1s pause still dismisses back to the
  wizard ("Keep going") — the grace window does not make Escape inert
  forever.

### Files

* `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` —
  `_SettlingGuardedConfirmationDialog`, `action_cancel`.
* `Tests/Wizards/test_first_run_setup_wizard.py` (4 new tests).
* `Tests/UI/test_first_run_wizard_live_contract.py` (premise correction).
