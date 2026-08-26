---
id: TASK-14808
title: Deliver the visible Console prompt queue experience
status: Done
assignee: []
created_date: '2026-08-10 06:11'
updated_date: '2026-08-10 20:32'
labels:
  - console
  - agents
  - ux
dependencies:
  - TASK-14806
references:
  - backlog/decisions/046-visible-bounded-console-prompt-queue.md
  - backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
documentation:
  - Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
  - Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the already-safe bounded queue through an honest keyboard-first Console shelf and manager, document it, and prove the complete experience live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a turn is accepted, Send becomes Queue and Enter or the button admits the exact text draft; Preparing, Queue full, Send, slash-command, attachment, staged-evidence, and boundary-race states use the approved routing and preserve refused drafts exactly.
- [x] #2 A one-row queue shelf and focused manager expose count, state, safe previews, edit, reorder, remove, clear waiting, pause, resume, retry, skip, resume next, retry stopped turn, review, and current-context confirmation with stable focus and stale-revision recovery.
- [x] #3 Collapsed and background presentations reveal only content-free counts and states; previews strip controls, escape Rich markup, and truncate by terminal cell width, while full prompt text is fetched only for the entry actively edited.
- [x] #4 Composer, mouse, keyboard, voice, hands-free, and programmatic sends share one UI dispatcher and one exact per-session Textual chain worker, while chat_screen size and method ratchets do not increase.
- [x] #5 F1 help, fleet and session markers, Console user guides, collapsed-composer documentation, and close, leave, and quit documentation match the shipped queue vocabulary and controls.
- [x] #6 Mounted tests prove rendering, key routing, draft transactions, revision-gated polling, safe previews, privacy, lifecycle-dialog integration, and neighboring control geometry at 80x24, 100x30, and 160x40.
- [x] #7 An isolated live-provider walkthrough verifies sequential draining, management, approval, pause, failure, Stop recovery, background isolation, one final notification, and close, leave, and quit warnings.
- [x] #8 Unsent queue text remains absent from database persistence, screen snapshots, prompt history, diagnostics, and logs; accepted queued turns persist through the normal message path.
- [x] #9 The manager remains pinned to its owning session ID and queue revision across tab switches, never retargets intents to the viewed session, and renders precomputed previews without fetching unselected prompt bodies.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the current composer, session-tab, conversation-row, send, voice, hands-free, help, and geometry seams; TASK-14801's intended Console reference is unavailable because that ID is occupied by unrelated roleplay work, so use the current mounted code as authority and document exact mount points.\n2. Add ConsolePromptQueueRegion and ConsolePromptQueueUIController under UI/Console_Modules, wire them with late-bound dependencies, and keep chat_screen.py limited to shrinking compatibility delegation.\n3. Join Enter/button/voice/hands-free/programmatic draft dispatch to typed sent/queued/refused outcomes with exact stash restoration and one per-session chain worker.\n4. Add the session-pinned revision-aware manager modal and all edit/reorder/remove/clear/pause/resume/retry/skip/context-review recovery actions while materializing only the actively edited prompt body.\n5. Integrate shelf, collapsed/background labels, F1 help, marker legend, lifecycle focus preservation, responsive TCSS, and user documentation without a new shortcut or setting.\n6. Add pure/mounted/joined/lifecycle/privacy/geometry tests, run mutation checks and reached suites, then complete isolated live verification, backlog notes, DoD, and final program audit.\n\nADR required: no new ADR\nADR path: backlog/decisions/046-visible-bounded-console-prompt-queue.md and backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md\nReason: ADR-046 already fixes ownership, interaction, recovery, privacy, and lifecycle behavior; ADR-031 fixes the keyboard and help constraints.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

- Added the body-free queue presentation/controller seam, always-mounted shelf,
  session-pinned revision-aware manager, exact draft transaction handling, and one
  per-session Textual chain worker. Queue admission validates before mutation; refused
  and pre-acceptance races preserve the exact draft transaction.
- Joined Enter/button/dictation/hands-free/programmatic paths, added content-free Qn
  session and conversation markers, and kept full prompt bodies out of background,
  collapsed, persistence, history, diagnostics, and app-log surfaces.
- Added state-specific failure/Stop/context recovery, destructive confirmations,
  current-context review fencing, background-session retry correctness, narrow-terminal
  manager geometry, and user/F1/lifecycle documentation.
- Live QA used an isolated `task14808_live_queue` profile and localhost delayed
  OpenAI-compatible endpoint. It verified preparing-race refusal, ordered draining,
  manager rendering, pause-after-turn/resume, Stop, accepted-turn persistence, and
  absence of queued canaries from app stdout/stderr. It found and fixed a handoff gap
  where an occupied-but-not-yet-accepted slot briefly advertised Send; that state now
  renders Preparing. Temporary servers and the 7,942 generated QA/test artifacts were
  stopped and removed after verification.
- A second isolated `task14808_live_openai` walkthrough exercised native OpenAI tool
  calls with local tools enabled. A real `list_characters` call mounted the approval
  card; a background tab exposed only `Q3` plus the approval marker; approval timeout
  left the stopped turn available for separate retry while the queue advanced
  `3 -> 2 -> 1`. Session close reported transcript/live-turn/unsent-prompt counts,
  and leave and quit each presented their queue-loss warnings while Stay preserved
  the work.
- Verification: 24 coordinator tests, 34 registry tests, 34 focused UI tests, 20
  selected controller tests, 18 core mounted queue tests, a 41-test bounded changed-
  seam pass, the full 69-test workspace-context file, and both live walkthroughs.
  The final review also made the existing stylesheet assertion read UTF-8 explicitly,
  eliminating a Windows cp1252 collection/runtime failure. Full-tree collection succeeds after
  installing the declared `playwright` and `trafilatura` web-search dependencies.
  Focused Ruff, Python compilation, and `git diff --check` pass. The feature leaves
  `chat_screen.py` at its branch-start line count and the same 625 methods. The
  repository's stale absolute screen ratchet still reports 19,167
  versus 17,727 even though this branch reduces the screen by one line; an unrelated
  browser-toggle characterization still differs from the controller state it
  asserts.

ADR required: no new ADR. ADR-046 and ADR-031 remain the governing decisions.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

The visible Console prompt queue is implemented, documented, live-verified, and green
across its focused and repository-collection verification gates. It is bounded to ten
per session, preserves refused drafts, drains sequentially, keeps unsent text
process-memory-only, and protects queued work across session close, Console leave, and
application quit.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] All acceptance criteria are complete and implementation deviations are recorded.
- [x] Automated pure, coordinator, mounted UI, lifecycle, privacy, and geometry tests cover the feature.
- [x] Focused Ruff, Python compilation, diff validation, and full-tree collection pass.
- [x] Console user guides, F1/help vocabulary, ADR links, and implementation notes are current.
- [x] Self-review and isolated live-provider verification are complete.
- [x] No new lesson entry is required; the live handoff gap is captured in these implementation notes and its regression test.
<!-- DOD:END -->
