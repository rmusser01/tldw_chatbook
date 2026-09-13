# Approved Console Regression Repairs Implementation Plan

> **For agentic workers:** Use superpowers:dispatching-parallel-agents for independent file owners, with TDD and independent specification/code review before integration.

**Goal:** Fix the five approved production defects without weakening the remaining test inventory or resource checks.

**Architecture:** Reuse existing controller/dialog lifetimes, action widths, tracked resume timers and trace descriptors. Keep data and state with current owners.

**Tech Stack:** Python 3.12 test environment, Textual 8, SQLite, pytest, Ruff.

**Approved specification:** `Docs/superpowers/specs/2026-09-06-approved-console-regressions-design.md`.

ADR required: no. ADR paths: ADR-033 application-session ownership and ADR-097 semantic trace ledger. Reason: bounded regression fixes to existing contracts, with no new boundary or storage policy.

## Execution and verification

Worktree: `/private/tmp/tldw-chatbook-dev-review-851c4e2`, starting at `6039d7a70e`.
Python: `/private/tmp/tldw-31651-env.rIizsz/bin/python`.
Test environment: `PYTEST_DEBUG_TEMPROOT=/private/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-review-pytest-DzlFOe`.
Use `python -m pytest -q <exact files> --junitxml=/private/tmp/<task>-<phase>.xml`;
use `/private/tmp/tldw_fd_inventory_probe.py` for final native resource attribution.
No full-suite run or resource/size/deadline relaxation. Commit only root-reviewed
cohorts; never stage another owner's unfinished code.

### TASK-31817 — owned dictation retry lifetime

Files: `tldw_chatbook/UI/Console_Modules/dictation.py`, only the dictation suspend
call in `tldw_chatbook/UI/Screens/chat_screen.py`, and `Tests/UI/test_console_dictation.py`.

- [x] Strengthen the retry double to require live retained audio; reproduce confirm/decline failures.
- [x] Add exact owned-dialog suspend, cancellation, ordinary navigation and unmount controls.
- [x] Implement the smallest controller-owned exact-dialog lifetime exception, capturing the suspend decision before asynchronous cleanup can race dismissal; retain default teardown behavior and repaint canonical mic state.
- [x] Run complete dictation and cached-screen suspension files; review spec compliance and cleanup races, then static checks.

### TASK-31928 — Stop action containment

Files: `tldw_chatbook/Widgets/Console/console_composer_bar.py`, `Tests/UI/test_console_pending_attachment_stash.py`, and the existing composer layout test file where relevant.

- [x] Reproduce original Stop failure; establish ordinary composer focus before synthetic Send and assert physical row/Stop containment to isolate horizontal clipping.
- [x] Include the existing Redirect width while visible, preserving idle space, order and visibility semantics. Reflow the draft after run-state width changes; cap the existing optional attachment label to remaining narrow-row space without weakening control reachability or existing draft-budget rules. The initial fixed +10 experiment regressed 80-column idle space, so retain dynamic accounting instead.
- [x] Verify actual click/cancel at unchanged 160x48 and 0.5-second click deadline; run complete attachment and relevant composer/layout files.
- [x] Review two representative rendered/geometry states and static layout scan; no forced hidden scrolling or unrelated styling changes.

### TASK-31929 — warm CHAT handoff replay (root owns)

Files: `tldw_chatbook/UI/Screens/chat_screen.py` ordinary resume timer list only, `Tests/UI/test_console_chat_handoff_resume.py`, `Tests/UI/test_uat_first_time_character_chat.py`.

- [x] Add a focused real warm-return CHAT regression with exact same-screen/session/acknowledgement checks; show RED. Add no-handoff and hide-before-timer controls.
- [x] Add `self.set_timer(0.15, self._consume_pending_chat_handoff)` to the existing tracked list under `not ordered_resume_active`.
- [x] Run focused controls and complete reuse/handoff/UAT files. Full UAT GREEN follows TASK-31931; do not disable capture to bypass it.
- [x] Review first-mount, saved-startup, cancellation and claim behavior.

### TASK-31930 — late empty-stack event (root owns)

Files: `tldw_chatbook/app.py` ContentsRebuilt handler and `Tests/UI/test_persona_buddy_app_mount.py`.

- [x] Call the real handler with an empty stack and assert no scheduling/no exception; verify RED.
- [x] Short-circuit with `if self.screen_stack and message.screen is self.screen:` before scheduling.
- [x] Verify empty, matching and stale screen controls, then full buddy app-mount and parallel-run files; review exact event ownership.

### TASK-31931 — synthesized system provenance

Files: `tldw_chatbook/Chat/console_chat_controller.py`, an existing durable trace/controller test file, and regression coverage using the existing UAT (do not edit its handoff ownership concurrently).

- [x] Add real durable Capture-On tests for unsaved leading system, saved system, ordinary unsaved active and nonleading system descriptors; show the typed category failure before the change.
- [x] Determine the contiguous leading `role == system` boundary and choose RENDERED_SYSTEM only for unsaved fallback rows in that slice; preserve existing saved descriptors and ACTIVE_REQUEST fallback elsewhere.
- [x] Run full changed durable test file and related prepared/provenance tests, then the complete real character UAT with TASK-31929.
- [x] Review fail-closed behavior, descriptor/category alignment and absence of capture bypasses.

## Integration completion

- [x] Independent plan/spec review before implementation; independent spec then code-quality review of completed changes.
- [x] Run complete affected files in bounded selections with saved logs and native SQLite attribution.
- [x] Run unchanged screen-size guard and explicitly report its pre-existing failure separately.
- [x] Update task acceptance criteria only when verified, record approved-design resolution and exact evidence, commit/push and update draft PR #2427; do not merge.

## Verification notes and justified refinements

- Independent review added exact owned-modal foreground-loss cleanup and exact
  settled-revision assertions. Both gaps have regression coverage and review.
- A pristine initial session may legitimately be repurposed by Start Chat. The
  new warm-return fixture instead seeds an existing conversation and verifies it
  remains unchanged; the existing UAT covers pristine-session behavior.
- Adjacent fixture ownership is completed under TASK-31927 using existing
  cleanup APIs. Seven complete files pass 253 resource-clean; root integrated
  owner controls and handoff/UAT/attachment/width files pass 56 resource-clean.
- Full buddy/parallel/live-handoff files pass 134 resource-clean; durable,
  prepared request and provenance files pass 91. Existing size, two route-census
  guards, and an unrelated Retry Speech narrow-layout failure are explicitly
  retained in the checkpoint, not waived or presented as passing.
- No new ADR. Exact evidence and remaining qualifications are in
  `backlog/docs/dev-test-review-checkpoint-2026-09-05.md`.
- Publication verified: repair commit `f663423048` pushed to the existing branch;
  draft PR #2427 description updated successfully. No merge was requested or run.
