---
id: TASK-36694
title: Handle each Speech Recognition button activation once
status: Done
created_date: 2026-10-24 17:33
labels:
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md
assignee:
- '@codex'
updated_date: 2026-10-24 21:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One ordinary Start Dictation button event is handled by the child and then manually forwarded to it again by the Speech container. Capture starts and immediately stops. Give handled child messages a single action delivery.

### Evidence and scope

Verified after PR #3228 merged, against dev `96cae7eec00473bbfb5687ccfd0685eba2785add`. One real Pilot click delivers the same Button event object twice: false-to-true then true-to-false; start_calls=1, stop_calls=1 and the final UI is idle. Existing keyboard-action control starts once and remains listening. The prior slice-41 QA correctness side-find has no matching backlog owner.

Book basis: PDF145–146 / printed136–137 explains message bubbling and stopping; PDF253 / printed244 uses event.stop() to keep a handled button from reaching its parent.

Affected owners: [Dictation_Window_Improved.py](../../tldw_chatbook/UI/Dictation_Window_Improved.py), [STTS_Window.py](../../tldw_chatbook/UI/STTS_Window.py).

### Existing ownership and decisions

The September27 slice41 QA correctness side-find1 (data/items.md:4507) recorded the duplicate dispatch, but is not a numbered finding with a task owner. TASK-33284/PERF25 owns blocking-latency findings and a ≤50ms performance outcome, not this correctness defect. TASK-16266 and TASK-15470 cover view-switch lifecycle and settings flushing.

Related tasks: [TASK-16266](../tasks/task-16266%20-%20Harden-Speech-view-switching-and-first-run-test-seam.md), [TASK-33284](../tasks/task-33284%20-%20PERF-25-Other-screens---Evals-Change-Review-Speech-video-player-splash.md), [TASK-15470](../tasks/task-15470%20-%20Config-persistence---batch-writes-and-take-them-off-the-click-path.md).

ADR required: no.
ADR paths: [025-shared-stt-artifacts-and-runtime-routing](../decisions/025-shared-stt-artifacts-and-runtime-routing.md), [031-tui-keybinding-and-footer-hint-conventions](../decisions/031-tui-keybinding-and-footer-hint-conventions.md).
Reason: Routine correction of handled-message delivery within existing Speech routing; no new application structure or keybinding policy.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A single mouse click or normal Enter/Space activation of Start starts one dictation session and leaves the foreground view listening until a distinct Stop action.
- [x] #2 Stop and other handled Speech Recognition buttons execute their intended action once, without a bubbled event reversing the result.
- [x] #3 Other Speech views and container-owned controls retain their routing and ADR-031-compliant keyboard behavior.
- [x] #4 Mounted controls exercise the actual parent and child handlers with the same event identity and bounded audio collaborators, including ordinary Start and Stop.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR paths: [ADR-025](../decisions/025-shared-stt-artifacts-and-runtime-routing.md) and [ADR-031](../decisions/031-tui-keybinding-and-footer-hint-conventions.md). Reason: handled-message delivery and explicitly required focused Space activation within existing Speech routing; no new terminal chords, global keys or service boundaries.
1. Add mounted mouse/Enter/Space Start and distinct Stop controls with actual STTS parent/Dictation child message delivery, bounded recorder I/O and real service where relevant; cover other handled and unhandled/container controls.
2. Retain duplicate-delivery RED evidence, then consume only messages handled by the child while preserving other routing. Provide focused Space with a private owner-local Button subclass while retaining inherited Enter (stock Textual8.2.8 Button binds Enter only). Preserve the reviewed retirement/capture-publication implementation from TASK-36693.
3. Run focused new and neighboring Speech controls, inspect private-child teardown, and run Ruff/format/whitespace checks.
4. Obtain task-scoped Superpowers requesting-code-review, register every new regression in the required UI census, update Speech architecture notes and record implementation evidence. Final hosted PR qualification is part of overall delivery.
Master plan: Docs/superpowers/plans/2026-10-24-textual-lifecycle-and-import-followups.md (Task2).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Handled Speech Recognition Button.Pressed messages now stop at the child after their intended action; unknown/container controls retain normal bubbling. All seven action controls use a private Button subclass for focused Space activation alongside inherited Enter, meeting the explicit AC without global or reserved-key changes. Existing capture retirement, callback dispatch/generation and Stop behavior remain unchanged.

Verification: 13 new mounted mouse/Enter/Space, Pause/Resume, Clear, actual clipboard/export/Troubleshoot and same-event routing cases plus 6 neighboring capture controls passed (19 parent cases and 19 clean private children, zero failures/errors/skips). Source hashes, Ruff/format/whitespace and complete split-census coverage were verified. All13 new cases are registered on the required UI gate; overall hosted PR qualification remains pending. Superpowers task review approved, with no blocking findings. No full local suite or real microphone/model/network/OS clipboard certification.

ADR required: no; [ADR-025](../decisions/025-shared-stt-artifacts-and-runtime-routing.md) and [ADR-031](../decisions/031-tui-keybinding-and-footer-hint-conventions.md) govern existing ownership and focused controls. Updated Docs/Architecture/speech.md and the UI census/ratchet. Source/tests: UI/Dictation_Window_Improved.py and Tests/UI/test_dictation_button_delivery.py.
Hosted qualification complete for PR #3243, code head087a75bab335485b2058f9a010c6f872c57ebc9b: https://github.com/rmusser01/tldw_chatbook/actions/runs/40132324399 (all25jobs success) and https://github.com/rmusser01/tldw_chatbook/actions/runs/40132324401 (Perf Guard success). Actual completed job logs prove all81 new UI selectors and20 parser cases executed, with17 successful UI shards and maximum986s, below1020s planning target/1200s cap. Pinned Canvas and all actual repository source guards pass on the hosted runner. No allowance, timeout, shard count or original census entry changed.

Superpowers requesting-code-review approved each task, the whole branch, and the single final fix re-review; M1/M2 are addressed with no remaining Critical/Important/Minor finding. All remote comment/review/thread pages were checked: no actionable finding or unresolved thread. All four acceptance criteria, implementation plan/notes, existing ADR links and relevant architecture/QA/incident documentation are complete. Native legacy/profile/Windows/static/size qualifications remain disclosed in Docs/superpowers/qa/2026-10-24-textual-followups-36693-36698.md; no full local suite or unrelated debt closure is claimed. This CLI closeout and verification-doc commit contains metadata only; its final PR head will receive fresh hosted CI before delivery.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->