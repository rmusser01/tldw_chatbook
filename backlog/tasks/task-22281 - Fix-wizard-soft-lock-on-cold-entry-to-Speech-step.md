---
id: TASK-22281
title: Fix wizard soft-lock on cold entry to Speech step
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:13'
updated_date: '2026-08-25 06:28'
labels:
  - ux
  - wizard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT finding F-1 (output/ux-review-setup-wizard/findings.md): on a fresh profile with the onnx-asr extra absent, reaching the Speech step for the first time via Next kills all keyboard input (Next/Back/Esc/Tab/Ctrl+P); render loop stays alive; app must be killed. Reproduced 2/2 cold, 0/2 warm (Resume path works). Full track is unusable past step 6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cold full-track walk on a fresh profile with optional deps absent reaches the Tools step using only the keyboard
- [x] #2 Root cause is documented in task notes; cross-checked against the H-3 30s validating-provider hang for a shared cause
- [x] #3 A regression test asserts the failure mechanism cannot recur (mechanism-level, not race-symptom)
- [x] #4 Existing wizard tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-establish the cold tmux repro on a fresh profile (UAT harness)\n2. Instrument the RAG->Speech transition: screen stack, focus target, worker state at each point\n3. Identify the input sink (phantom modal vs focus black hole vs binding failure)\n4. Fix at the root; re-run cold repro 3x\n5. Mechanism-level regression test; cross-check H-3 hang for shared cause\n6. Full wizard test suite + live cold walk to Tools
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: SetupWizardContainer.show_step()'s F-B focus fix focuses a child of the incoming step immediately after SpeechSetupStep's first on_show schedules refresh(recompose=True) via _ensure_loaded; the recompose detaches the focused widget and Textual 8.2.8 leaves app.focused pointing at the detached node, so every key event dispatches into a dead message pump and no binding (container/screen/app) ever resolves. Confirmed with file-probe instrumentation (persistent log only records diagnostics.* events; loguru stderr is swallowed by the TUI). Warm entries skip the lazy-load recompose (_loaded gate), which is why Resume worked.

Fix: SetupStep.refresh() override schedules _heal_orphaned_focus via call_after_refresh on every recompose=True — heals only when focus is detached/None and the step is visible; restore order: same-id widget in new tree, preferred_focus(), first focusable, nav bar. Covers all step recomposes (first-show load, load-completion, provider discovery), where a one-shot fix at the focus site would be re-orphaned.

H-3 cross-check: NOT the same mechanism. H-3's 'Validating provider' hang has live input and live UI (dialogs/toasts still worked); F-1 is dead input with live rendering. H-3 remains open under TASK-21145.

Evidence: regression test test_cold_full_track_speech_entry_keeps_keyboard_alive (failed pre-fix with focus orphaned on detached #setup-speech-use-from-disk, passes post-fix); full live-contract file 80/80; live tmux cold walk on fresh profile now reaches Built-in tools (Step 7 of 10) and Ctrl+B returns. Lessons entry added to backlog/docs/lessons-textual.md.

Files: tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py (SetupStep.refresh + _heal_orphaned_focus), Tests/UI/test_first_run_wizard_live_contract.py (regression test + STEP_RAG/STEP_SPEECH imports), backlog/docs/lessons-textual.md.
<!-- SECTION:NOTES:END -->


## Renumbering provenance

Renumbered from TASK-21139 on 2026-08-25 per the 2026-08-21 owner rule
(TASK-19601): upstream's "Restore Windows checkout for Backlog task paths"
(created 2026-08-23) is the older arrival and keeps the id; this task
(created 2026-08-25 on the fix/setup-wizard-uat branch, PR #2101) is the
younger and renumbers. Git commit messages retain the historical id;
code comments, tests, lessons entries, and dependent task references were
updated to TASK-22281.
