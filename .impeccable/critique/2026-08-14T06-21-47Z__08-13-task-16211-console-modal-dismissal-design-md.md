---
target: TASK-16211 Console modal dismissal design
total_score: 30
max_score: 40
na_heuristics:
p0_count: 0
p1_count: 3
timestamp: 2026-08-14T06-21-47Z
slug: 08-13-task-16211-console-modal-dismissal-design-md
---
# TASK-16211 Console modal dismissal critique

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of System Status | 3 | Protected states are visible, but pending async cancellation needs defined feedback. |
| 2 | Match System / Real World | 4 | Escape and backdrop cancellation match familiar modal conventions. |
| 3 | User Control and Freedom | 3 | Exits improve, but Settings can strand its undo affordance. |
| 4 | Consistency and Standards | 3 | One grammar is strong; exceptional Escape stacks need explicit contracts. |
| 5 | Error Prevention | 2 | Hit testing is careful, but repeated async cancellation and side effects remain unsafe. |
| 6 | Recognition Rather Than Recall | 2 | Context-sensitive Escape states need focus and feedback postconditions. |
| 7 | Flexibility and Efficiency | 4 | Keyboard, pointer, and visible controls coexist. |
| 8 | Aesthetic and Minimalist Design | 4 | The change adds no visual clutter. |
| 9 | Error Recovery | 3 | Dirty and staged artifacts are protected; Settings undo is not yet. |
| 10 | Help and Documentation | 2 | Footer/help discoverability remains inconsistent. |
| **Total** | | **30/40** | **Good foundation; safety gaps remain.** |

## Design Specificity Verdict

The proposal is strongly authored for Chatbook and Textual rather than a generic modal recipe. Its combined DOM/geometry boundary, descendant overlay handling, Prompt Workbench states, enhanced-file-picker Escape stack, staged-video ownership, and exact typed results fit the product's keyboard-first, recoverable Console. The remaining gaps are equally product-specific: Console Settings owns immediate side effects and an in-modal undo token; async cancellation can race Textual's screen stack; and Console reachability includes modal-to-modal launch paths.

The deterministic detector returned zero findings because it scans web markup rather than Python or Markdown. Source inspection found 27 Console-owned `ModalScreen` classes and seven directly shared types, plus transitively reachable nested modals that the current direct-launch definition misses. This detector result is non-probative, not evidence that the design is complete.

## Overall Impression

The core interaction is right and substantially improves user control. The biggest opportunity is to make the shared contract a real state machine—single-shot, top-screen-only, focus-restoring, and explicit about side effects—rather than only a hit-test helper.

## What's Working

1. Combined DOM ancestry and geometry correctly treats Textual overlays as inside even when they render outside the dialog box.
2. Generated-video dismissal now protects the only staged artifact instead of mapping Escape directly to destructive discard.
3. Modal-specific typed results and callbacks stay at the modal boundary rather than leaking into `ChatScreen`.

## Priority Issues

### [P1] Console Settings cancellation can strand recovery state

`ConsoleSettingsModal` can immediately reset branch memory and retain the only undo token on the modal, while compaction may continue asynchronously. Closing through Cancel, Escape, or backdrop can remove the recovery affordance without undoing the mutation.

**Fix:** Add Settings to the exception table. Define reset-token and in-flight-compaction states; either externalize Undo into Console or guard close with explicit choices. Define whether compaction continues, cancels, or blocks close and show continuing work if applicable.

### [P1] Async cancellation needs single-shot and top-screen guarantees

Textual invokes matching handlers across the MRO, and async cancel callbacks create a window for repeated Escape/click input. A late `dismiss()` can pop whichever screen is currently on top.

**Fix:** Set a cancellation latch before any await, consume subsequent gestures, route every terminal dismissal through `dismiss_once`, and verify the modal is still mounted and is the active top screen immediately before dismissal. Test held Escape, rapid backdrop clicks, and delayed callbacks.

### [P1] Console modal reachability must be transitive

Direct `ChatScreen` construction misses modal-to-modal flows such as `CancelConfirmationDialog` from the prompt queue and `ChangeRevertConfirmModal` through `ChangeReviewScreen`.

**Fix:** Inventory the transitive launch graph: `ChatScreen` to screen/modal to nested modal. Record opener, type, content boundary, safe hook/result, guards, and focus postcondition. Explicitly justify exclusions.

### [P2] Focus and gesture postconditions are under-specified

Prompt Workbench's hidden dirty guard does not currently restore editor focus, and the capacity guard and nested-overlay cases have no required focus destination. Changing Workbench Escape from Back to Close also changes learned keyboard behavior.

**Fix:** Add a transition table for clean, dirty, guard-visible, active-work, transient-overlay, and cancellation-pending states. Every transition states the result, feedback, and focus destination. Escape on a visible dirty guard must use the same Keep Editing path as its button.

### [P2] Event tests need real Textual semantics

Textual replaces absent screen coordinates with local coordinates, `Pilot.click` cannot emit non-primary buttons, and base/subclass click handlers can both run. A mock-only coordinate test can falsely pass while real dispatch double-dismisses.

**Fix:** Separate pure hit-classifier tests from mounted dispatch tests. Dispatch real button-2/3 events explicitly, exercise the full MRO, count callbacks for all existing backdrop handlers, and cover double-click/click-through. Treat `event.widget is None` or an explicit preserved provenance marker as fail-closed rather than relying on impossible `screen_x is None` state.

## Persona Red Flags

- **Alex (power user):** Changing Prompt Workbench Escape from Back to Close may discard learned navigation context; held Escape can over-pop nested screens without a latch.
- **Sam (keyboard/accessibility):** Focus restoration is not yet an acceptance criterion, so closing a guard can leave focus on a hidden widget. Shortcut hints are inconsistent.
- **Riley (stress tester):** Reset memory then backdrop-click strands Undo; repeated Escape during an awaited callback can trigger duplicate side effects; direct-only inventory omits nested destructive confirmations.

## Minor Observations

- Existing Composer, RAG Settings, and Image Viewer click handlers all need reconciliation because Textual dispatches matching handlers across the MRO; exact-once testing should cover all three, not only Image Viewer.
- `ConsoleSettingsModal` has a non-dismissal click handler for redirected Textual-Web Select events that must survive the mixin.
- `VideoPlayerScreen` has no bounded wrapper; treating only a child as content would misclassify ordinary player cells as backdrop.
- A pure mixin is safer than a second `ModalScreen` base for `EnhancedFileOpen`'s deep inheritance chain.
- The cross-module, long-lived dismissal grammar should amend ADR-031 rather than merely cite it.

## Questions to Consider

- Is matching visible Cancel truly safe when that button can strand an undo token?
- Should backdrop ever close a modal while billed or long-running work is active?
- Is Prompt Workbench Escape fundamentally Back or Close?
- What exact widget receives focus after every cancellation outcome?
