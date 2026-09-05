# Library Assembly Cleanup Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement the two independently reviewed tasks below in the existing shared review worktree.

**Goal:** Restore the unchanged Library line and method ratchets by completing existing controller ownership and extracting unchanged assembly.

**Architecture:** Conversations behavior remains in its existing controller. The screen calls that owner directly instead of eight obsolete private forwarding methods. A separate assembly function constructs the same six existing controllers, in the same order and at the same initialization position, with explicit late-bound keyword dependencies.

**Tech Stack:** Python, Textual, pytest, AST characterization.

ADR required: no
ADR path: N/A
Reason: Mechanical cleanup and assembly relocation implement DESIGN.md section 7 and the approved 2026-08-02 screen-decomposition design without changing ownership, events, DOM, or behavior.

## TASK-31734: Conversations private forwarding cleanup

Files: `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/Architecture/test_library_conversations_wiring.py`, `Tests/UI/test_library_shell.py`.

- [x] Baseline the existing Conversations characterization and architecture tests; retain the rebased Notes results separately.
- [x] Extend the existing pruned-name regression for `_library_conversation_focus_region`, `_library_conversation_escape_label`, `_adopt_library_conversation_state_selection`, `_carry_selected_conversation_into_snapshot`, `_selected_conversation_record`, `_library_conversation_page_needs_recovery`, `_fail_library_conversation_request`, and `_notify_library_conversation_unavailable`. Confirm RED before deleting them.
- [x] Remove those eight one-line delegates. Retarget existing screen callers and the sole direct UI test call to `_conversations_controller`; preserve the controller bodies and behavioral assertion ASTs.
- [x] Run Conversations UI/architecture tests and relevant shell conversation tests. Review the diff, record exact line/method counts, request parent review, and commit only scoped files.

## TASK-31735: Ordered existing controller assembly

Files: `tldw_chatbook/UI/Screens/library_screen.py`, new `tldw_chatbook/UI/Library_Modules/wiring.py`, new `Tests/Architecture/test_library_controller_assembly.py`.

- [x] Characterize the six existing controller constructors, exact order, keyword names and post-construction late binding through real `LibraryScreen` construction.
- [x] Move the contiguous constructor block into `build_library_controllers(screen)`. Keep all controller bodies unchanged and preserve its call position between Skills state initialization and reader-preference loading. Keep load-bearing screen controller imports used by static forwarders.
- [x] Compare all six normalized constructor call ASTs against the pre-move source. Check that no controller state or event ownership changed.
- [x] Run all five existing controller architecture files, new assembly checks, both Library ratchets, Conversations/Export/Collections characterization, relevant RAG/Skills coverage, Notes responsive/retention and entry-reuse coverage.
- [x] Format/lint new files and touched screen ranges, compare preexisting screen findings, check diff whitespace, record final counts, and request parent review before scoped commit. Parent approved the exact shared Library ratchet tightening to 41,303 / 1,301 and owns integrated verification.

No new CSS or DOM nesting is introduced. No timeout, exact scroll, focus, editor identity, geometry, or ratchet assertion is relaxed. Notes light-theme and compact-export painted failures remain separate bug repairs.
