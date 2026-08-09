---
id: TASK-3314
title: Unify ingest consent inline — retire the guardrail modal
status: Done
assignee:
  - '@claude'
created_date: '2026-08-08 20:30'
updated_date: '2026-08-09 15:49'
labels:
  - library
  - ingest
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved by the owner via task-3310 (ruling 2). The critique's "one consent grammar" question: guaranteed failures gate inline while missing tooling raises a blocking modal — the worse outcome gets the quieter treatment, and consent changes shape with failure type. The owner ruled to fold tooling-warning consent into the inline commit/gate grammar and retire `IngestGuardrailModal` entirely (its rendering was fixed in tasks 3300/3304, so this is a consolidation, not a bug fix). The inline preflight warnings already carry per-warning copy-install-command buttons (task-3304), so the modal's information is already on the canvas — only its consent step remains to move.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Starting an import with active tooling warnings requires an explicit second confirmation carried by the inline commit/gate grammar (the repo's incumbent two-press pattern), naming how many files may fail
- [x] #2 `IngestGuardrailModal` and its tests are removed; no modal appears on any Start path
- [x] #3 The copy-install-command affordance remains reachable at the inline warnings
- [x] #4 Starts with no warnings are unchanged (single press); Esc/blur resets a pending confirm state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. State layer: `start_confirm_armed` kwarg + field on the canvas state. When armed AND
   the gate is open AND tooling warnings are active, the gate line
   (`start_quiet_line`) becomes the explicit confirm copy naming the blast radius:
   "⚠ Press Start again to import anyway — N file(s) may fail." N comes from a new
   `count_warning_affected_files(preflight)` helper (distinct staged files whose type
   group depends on a warned feature; replaces the modal's `_affected_counts`).
2. Screen: two-press consent inside `_submit_library_ingest_form`, MIRRORING the
   queue's incumbent Clear-finished mechanism exactly: screen-attr state carrier
   (`_library_ingest_start_confirm_armed` + `_armed_at`), first press arms + in-place
   gate update only, 0.3s double-press dead zone, second press submits. An
   armed-against warnings snapshot is kept so a preflight result carrying DIFFERENT
   warnings disarms (the same "the thing you armed against changed" rule that disarms
   Clear finished on registry mutations) while the Enter-in-path re-trigger landing an
   IDENTICAL forecast does not steal the pending confirm.
3. Reset set (what invalidates the forecast clears the consent): genuine path text
   change; a fresh preflight result with different warnings; preflight invalidation
   (submit/Clear/reset); rail-switch away (pause + reset); per-type option edits; Esc
   while armed (disarms and STAYS on the canvas — the consent "no"; a second Esc
   leaves as before).
4. Render: the confirm state lives in `_update_library_ingest_gate`'s domain — content
   via `start_quiet_line`, warning treatment via a `-ingest-start-confirm` class
   toggle (canvas DEFAULT_CSS, `$warning` token + bold; the ⚠ glyph keeps it
   monochrome-legible). Enter-submit shares `_submit_library_ingest_form`, so
   Enter,Enter gets the same two-press semantics for free (pinned).
5. Remove `IngestGuardrailModal`, its push site, `_affected_counts`, and the
   now-unused `ModalScreen` import; drop `_do_submit_ingest`'s `confirmed` param.
6. Tests: mine `Tests/UI/test_library_ingest_guardrail_modal.py` assertion-by-assertion
   (disposition table in Implementation Notes) — submit-flow tests migrate to a new
   inline-consent suite, count pluralization + copy-reachability + theme-token checks
   migrate to state/canvas suites, modal-chrome geometry retires with the modal.
   Update `Tests/integration/test_library_ingest_flow.py`'s modal test to the inline
   flow. Rewrite the guardrail section of
   `Docs/User_Guide/library/import-and-export.md` for inline consent.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Shipped**: one consent grammar. Start with active tooling warnings is now an inline
two-press at the gate line — first press converts it to "⚠ Press Start again to import
anyway — N file(s) may fail." (warning-styled, `-ingest-start-confirm` class,
`$warning` + bold via canvas DEFAULT_CSS; the ⚠ glyph keeps it monochrome-legible) —
second press submits. `IngestGuardrailModal` is deleted (class, push site,
`_affected_counts`, the `ModalScreen` import, and its test file).

- **Mechanism mirrored from Clear finished** (task-2015/2160, read before writing):
  screen-attr carrier (`_library_ingest_start_confirm_armed` / `_armed_at`), arming
  changes ONLY the gate line in place (no recompose, no scroll — the confirm appears
  under the finger/caret), a 0.3s double-press dead zone
  (`_START_CONFIRM_DEAD_ZONE_SECONDS`, its own constant so the two knobs stay
  independent), and "the thing you armed against changed" disarms. The armed flag is
  never trusted by the renderer: the state builder gates it on
  start_enabled + active warnings, so a stale carrier can't paint consent copy.
- **Blast radius N**: new `count_warning_affected_files` in `library_ingest_state.py`
  (distinct staged files whose type group depends on a warned feature; a file lives in
  exactly one group, so group-count summation is a distinct count). Pluralized;
  count-less defensive variant when no staged group claims a warned feature.
- **Reset set** (derived from what invalidates the forecast): genuine path edit
  (Changed handler), pre-flight invalidation (submit/Clear/reset), rail-switch pause, a
  fresh result whose WARNINGS differ from the armed-against snapshot (an identical
  re-trigger result keeps the consent — otherwise Enter,Enter could never submit, since
  Enter re-triggers pre-flight), per-type option edits, path-field blur, and Esc.
  **Esc while armed disarms and STAYS on the canvas** (the consent "no"); a second Esc
  leaves for the hub as before.
- **Enter,Enter pinned**: Enter routes through the same `_submit_library_ingest_form`,
  so the keyboard path carries identical two-press semantics (pilot test renders the
  confirm line, asserts identity across the in-place hot path, then submits).
- **Guardrail test file disposition** (assertion-by-assertion, per the
  deleted-assertion lesson):
  - MIGRATED → `Tests/UI/test_library_ingest_inline_consent.py`:
    `test_submit_with_blank_path_warns_to_import_not_ingest` (as-is);
    `test_submit_with_warnings_shows_guardrail_modal` → first press arms, never
    submits, never pushes any screen; `test_submit_confirm_guardrail_calls_submit` →
    second press past the dead zone submits with the source kwarg;
    `test_submit_without_warnings_calls_submit` → single press unchanged + never arms;
    `test_submit_clears_the_stale_preflight_summary` (as-is);
    `test_guardrail_modal_pluralizes_file_counts` → confirm-copy pluralization
    ("1 file may fail" / "2 files may fail", never "1 files");
    `test_affected_counts_aggregates_by_feature` → `count_warning_affected_files`
    distinct-file semantics; `test_guardrail_modal_css_uses_theme_tokens` → the
    confirm class's canvas CSS carries `$warning`, no black/gray literals;
    `test_guardrail_modal_copy_command` → copy affordance reachable AT the inline
    warnings while armed (pilot pin; the always-on canvas copy behavior itself was
    already pinned by test_library_ingest_structural's MI-17 pair).
  - RETIRED with the modal (chrome-specific, no inline analogue):
    confirm/cancel-button callbacks, Escape-dismisses-the-modal (the dismiss(False)
    ast bug pin), compact-modal geometry, action-buttons-single-line, actions
    reachable at 7 warnings on 24 rows, warning-list scrolls-not-clips,
    Cancel-not-destructive variant convention.
  - RETIRED as duplicate: `test_guardrail_warning_line_never_echoes_label_as_its_own_hint`
    — the inline builder's identical rule is already pinned by
    `test_build_warning_lines_does_not_repeat_the_label`
    (Tests/Library/test_library_ingest_state.py).
  - Integration: `test_guardrail_modal_shows_when_pdf_deps_missing` REWRITTEN as
    `test_inline_consent_gates_start_when_pdf_deps_missing` (first press: no modal, no
    job, armed; second press: job lands in the real registry).
- **Docs**: `Docs/User_Guide/library/import-and-export.md` — "Consent for risky
  imports" section replaces the dialog paragraph; Start-row copy, warning-fix task,
  keyboard section (Enter,Enter + Esc-declines), and a task-3313/3314 stamp; historical
  "Verified against" stamps left as history.
- **RED evidence**: the new suite ran before implementation — 16 failed, headlined by
  `Expected 'push_screen' to not have been called. Called 1 times.` with
  `IngestGuardrailModal` in the call args (the exact old behavior).
- **Mutation evidence**: making the first press submit directly
  (`if False and …warnings`) → 2 RED (arm test + Enter,Enter pilot); making the disarm
  a no-op → 5 RED (invalidate, different-warnings result, both Esc tests, path edit).
  Both restored via Edit; suites re-verified green after restore.
- **Files**: `tldw_chatbook/UI/Screens/library_screen.py` (two-press submit, disarm
  hooks, gate-updater class toggle, modal deletion),
  `tldw_chatbook/Library/library_ingest_state.py` (`start_confirm_armed` field+logic,
  `count_warning_affected_files`),
  `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (DEFAULT_CSS, compose-time
  class), `Tests/UI/test_library_ingest_inline_consent.py` (new, 19 tests),
  `Tests/integration/test_library_ingest_flow.py` (rewritten modal test),
  `Tests/UI/test_library_ingest_guardrail_modal.py` (deleted),
  `Docs/User_Guide/library/import-and-export.md`.
- **Verification**: consolidated ingest battery 644 passed; nav audit subset 10
  passed; `test_library_shell -k ingest` 28 passed / 14 failed, all 14 the
  pre-existing task-3315 `_ingest_local_stt_jobs` harness drift (verified per-failure).
- **Owner-review notes**: the confirm treatment lives in the canvas widget's
  DEFAULT_CSS rather than the built bundle (kept out of `css/` on the
  never-hand-edit-the-bundle rule; tokens only); path-field BLUR parks a pending
  confirm (AC#4's wording) — the one visible consequence is that arming via Enter and
  confirming via a Start CLICK re-arms once instead of submitting, which reads as
  consent-conservative rather than broken.
xhigh review + live-verify round (2026-08-09): AC#4's blur clause was wrong, and the notes above
called the consequence "consent-conservative rather than broken" -- live it is simply broken.
(a) Blur no longer disarms. The original reasoning assumed the mouse flow always blurs BEFORE the
arming press, which holds only when the FIRST press is the click. Arm with Enter in the path field
and the gate line says "Press Start again to import anyway"; the Start CLICK that copy asks for
blurs the path field on its way in, so the disarm fired between the gesture and the press handler
and the second press merely RE-ARMED. Nothing could ever submit by the route the copy prescribes.
A blur carries no information about the forecast, so it is not an invalidation; every genuine
invalidator (path edit, option edit, a fresh pre-flight with different warnings, pre-flight
invalidation, Browse pick, rail switch, Esc) still disarms.
(b) Browse… could submit file B under file A's consent. The picker callback wrote `form.path`
directly, so the recomposed Input's re-announcement equalled the form's copy and
`handle_library_ingest_path_changed`'s echo guard dropped it -- the one seam that disarms on a path
change never ran. New `_adopt_library_ingest_path` is the single seam for every NON-typing path
writer: it sets the field and disarms when the value actually changed.
Tests (Tests/UI/test_library_ingest_inline_consent.py): Enter-arm -> Start-click submits (the click
must be scrolled into view first -- `pilot.click` addresses screen coordinates and the Start button
sits below the fold at 170x48, so an unscrolled click makes the test vacuous); a bare blur keeps the
confirm; a Browse pick of a second file disarms. Mutation check (restore the blur disarm) sends the
first two RED.
<!-- SECTION:NOTES:END -->
