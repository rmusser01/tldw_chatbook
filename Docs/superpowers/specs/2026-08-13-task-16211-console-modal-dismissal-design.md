# TASK-16211 — Safe Console modal dismissal design

## Status

Approved in conversation on 2026-08-13 and amended on 2026-08-14 after an
independent UX/source audit. The user approved the three-choice Settings reset
guard described below.

## Problem

Console modals are inconsistent. Most Console-owned `ModalScreen` classes bind
`Escape`, but several shared dialogs opened from Console do not, and only a few
support clicking the backdrop. Users can become trapped in a modal or must find
its visible Cancel control even though both `Escape` and a backdrop click are
standard no-action exits.

This must be fixed without turning dismissal into an accidental save, confirm,
delete, or draft discard.

## Goals

- Every `ModalScreen` reachable from Console has a keyboard cancel path and a
  primary-button backdrop cancel path.
- Both paths are semantically identical to that modal's safe Cancel or Close
  request, including exact result values, callbacks, guards, and focus
  postconditions.
- Existing safeguards for dirty state, active work, nested controls, and
  destructive confirmations remain intact.

## Non-goals

- The required `ConsoleSetupModal` overlay is not dismissible. It is an embedded
  setup gate rather than a `ModalScreen`, and closing it early would leave the
  Console unusable.
- This task does not redesign modal layouts or general focus order. It may add
  the narrowly required destructive-close guard and must restore focus after
  transient guards and modal dismissal.
- This task does not make arbitrary non-Console screens dismissible.
- This task does not make backdrop clicks confirm destructive actions.

## Scope

Scope is defined by actual Console reachability, not by filenames alone.

1. All Console-owned `ModalScreen` implementations in
   `tldw_chatbook/Widgets/Console`, including the model popover and prompt
   workbench.
2. Shared modal components that Console opens directly, currently including
   contextual Workbench help, dictionary/world-book pickers, confirmation
   dialogs, video cost and capacity decisions, the video viewer, and
   `EnhancedFileOpen` for attachments and `EnhancedFileSave` for generated
   video export.
3. Every transitively reachable modal launched from those screens, such as
   rename-from-session-switcher, prompt-queue cancellation, and change-revert
   confirmation, where only the top screen may be cancelled.

Shared components receive intrinsic safe-cancel behavior wherever they are
reused. That is intentional: a confirmation dialog must not return different
cancel semantics depending on which screen opened it.

The focused inventory is the fixed-point union of all `ModalScreen` subclasses
discovered under `Widgets/Console`, explicit shared modal types constructed by
current `ChatScreen` launch paths, and nested modal launches from every screen
already reached. The baseline includes 27 Console-owned modal types; shared
Workbench help, dictionary/world-book pickers, confirmation dialogs, video
viewer, `EnhancedFileOpen`, and `EnhancedFileSave`; and
`ChangeRevertConfirmModal` reached through `ChangeReviewScreen`.
`ConsoleSetupModal` remains the explicit non-screen
exclusion.

The inventory test asserts the complete current set. Every row records modal
type, opener, content boundary, safe hook/result, guard state, and focus
postcondition. This keeps exceptional behavior explicit rather than hidden in
a broad base-class assumption and catches multiline declarations such as
`ConsoleModelPopover`, which a simple anchored text search misses. Adding a
future direct or nested launch requires updating the table; this task does not
add a new Console screen-launch abstraction solely for inventory maintenance.

## Design

### Shared dismissal boundary

Add one small reusable mixin at the widget boundary rather than a second
`ModalScreen` base. A participating modal declares its bounded content widget,
an Escape request hook, and a terminal safe-cancel hook. The terminal hook
defaults to `dismiss(None)` but may return another safe value, await an existing
cancellation callback, or reveal a guard. Separate Escape routing is necessary
for the enhanced file dialogs and Prompt Workbench, whose transient state must be
handled before terminal cancellation.

The shared behavior owns only two concerns:

1. the common `Escape` cancel action;
2. deciding whether a primary-button click is on the backdrop.

It does not know how a particular modal saves, confirms, navigates, or guards
dirty state.

The shared layer is single-shot. It tracks three separate states: a temporary
cancellation-pending latch, a permanent cancel-side-effect commitment, and a
permanent terminal-dismissal commitment. It sets the pending latch before any
await and consumes repeated Escape/backdrop gestures while pending. An awaited
callback is committed before invocation and can never run again, even if a
nested screen makes the later dismissal stale. After that nested screen closes,
a new request may retry only terminal dismissal. Immediately before dismissal
the mixin verifies that the modal is still mounted and is the app's active top
screen. A stale callback must never pop a newer nested screen or invoke a cancel
callback twice.

All pre-existing screen-level click handlers are either removed in favor of
the mixin or made to delegate to it. This includes Composer Menu, RAG Settings,
and Image Viewer; Console Settings' unrelated redirected-`Select` click
recovery remains intact. Textual dispatches matching handlers across the MRO,
so `event.stop()` alone is not an exact-once guarantee.

The safe hook also preserves modal-owned cleanup that precedes current
cancellation. The contract inventory records those pre-cancel hooks explicitly,
including Character/Style/Session Switcher debounce cancellation and Citation
Sources request-generation invalidation. The shared default is used only where
the existing Cancel path has no such side effect.

### Backdrop classification

A click is inside the modal when either of these is true:

- the original event widget is the content widget or one of its descendants;
- the click's screen coordinates fall inside the content widget's region.

The DOM check keeps descendant overlays such as Textual `SelectOverlay` options
logically inside the modal even when their rendered region extends beyond the
dialog box. The geometry check keeps blank padding and non-widget-painted areas
inside the dialog from being mistaken for backdrop.

A click cancels only when `event.button == 1`, its target and screen-relative
coordinates can be classified, and it fails both inside checks. Production
Textual click events always expose integer screen coordinates because missing
values are normalized to local coordinates; the classifier still accepts an
explicit unknown provenance from direct callers and fails closed. The event is
stopped and default handling is prevented before cancellation. Mounted tests
must exercise full MRO dispatch and verify that rapid/double backdrop input
cannot click through to the revealed screen.

### Escape behavior

`Escape` routes to the same safe cancellation hook as the visible Cancel or
Close button once no modal-owned transient surface has claimed it. Existing
picker navigation bindings remain unchanged.

When a descendant transient overlay, such as an expanded `Select`, owns
`Escape`, the first keypress closes that overlay. A subsequent `Escape` closes
the modal. This preserves standard nested-control behavior and avoids throwing
away a larger form merely to close a dropdown.

`EnhancedFileOpen` and `EnhancedFileSave` already implement a deliberate,
screen-owned Escape stack:
path input, search, recent files, and bookmarks close in that order before the
picker dismisses with `None`. That `action_smart_dismiss` contract remains
unchanged. A primary backdrop click, by contrast, matches the picker's visible
Cancel button and dismisses the whole picker with `None`; it does not merely
peel one of those internal surfaces. Clicks on those surfaces still classify
as inside the picker and never cancel it. The terminal Escape branch, backdrop,
and visible Cancel all use the shared single-shot/top-screen dismissal path;
the existing file-dialog `dismiss` override still persists recent-location
state.

### Modal-specific behavior

- Confirmation dialogs cancel with `False`, never `None` or `True`, and run any
  existing cancel callback.
- Ordinary pickers and viewers cancel with `None`.
- Prompt Workbench dismissal uses its existing Close path. Dirty edit/recipe
  state displays the existing discard guard; active improvement work keeps its
  current cancellation behavior. In a clean nested Workbench mode, `Escape`
  closes the whole modal rather than performing Back navigation; the visible
  Back button remains the navigation control. When the dirty guard is already
  visible, `Escape` invokes the exact visible Keep Editing path, hides the
  guard, and restores editor focus. A backdrop click cannot bypass the explicit
  Discard control. The implementation records and tests transitions for clean,
  dirty, guard-visible, improving, cancelling, and nested-control states,
  including result, feedback, and focus destination.
- Console Settings treats immediate branch-memory reset and active compaction
  as exceptional states. When a reset undo token exists, Cancel, `Escape`, and
  backdrop reveal one three-choice guard: **Undo and close** invokes the exact
  undo operation and closes only after it succeeds; **Keep reset and close**
  accepts the reset and closes; **Return** leaves the modal open with the Undo
  affordance intact. If undo has expired, the guard remains open with the
  existing recovery message rather than pretending the reset was reversed.
  While Compact now is active, a close request reveals a separate two-choice
  acknowledgement: **Close anyway** stops waiting in this modal and warns that
  provider work may continue and may still be billed; **Return** keeps the
  modal open on the existing progress state. The implementation must not claim
  that an already-dispatched provider call was cancelled, and reopening
  Settings reloads the latest durable memory state rather than trusting the
  abandoned modal worker.
- Nested modal dismissal removes only the top `ModalScreen`.
- The image viewer keeps its intentional click-anywhere-to-close behavior. Its
  existing click handler is reconciled with the shared handler so dismissal is
  invoked once.
- `ConsoleVideoCapacityModal` has no passive Cancel result: Keep/Retry may write
  or evict managed videos, Save writes externally, and Discard destroys the
  only staged result. Its generic Escape/backdrop hook therefore opens a
  topmost discard-confirmation guard rather than returning any of those three
  terminal actions. The staged artifact and outer capacity modal remain owned
  and alive while the guard is open. Cancelling the guard returns to the
  capacity choices; only its explicit Discard confirmation returns
  `"discard"`. This replaces the current unsafe Escape-to-discard shortcut and
  prevents a generic dismissal gesture from silently losing generated output.
  Repeated gestures cannot stack duplicate guards, and cancelling the guard
  restores focus to the capacity modal's safest reason-specific default.
- Successful modal dismissal restores the opener's previous focus when that
  widget remains mounted. If it does not, Console falls back to its composer
  input. Cancelling any in-modal guard restores the exact editing or decision
  control named by that modal's inventory entry.
- On a viewport where content fills the entire screen there is no backdrop to
  click; `Escape` remains the available dismissal path.

## Alternatives considered

### Duplicate handlers in every modal

This is locally simple but repeats coordinate and cancel logic across many
files. It is easy for future modals to omit and already caused the current
inconsistency. Rejected.

### Intercept at `ChatScreen`

The underlying screen does not own events while a `ModalScreen` is active and
cannot safely infer modal-specific cancel values, callbacks, or dirty guards.
Rejected as the wrong boundary.

### Geometry-only hit testing

This matches two existing Console implementations but breaks Textual overlays
that render outside their owning content region. Rejected in favor of combined
DOM and geometry checks.

## Testing

Focused Textual Pilot tests will prove:

- `Escape`, visible Cancel/Close, and backdrop clicks return the same safe value;
- backdrop clicks close representative ordinary, boolean, async-callback, and
  nested modals;
- clicks inside content and on expanded `Select` options do not dismiss;
- direct classifier calls with unknown provenance and explicitly dispatched
  non-primary Textual clicks do not dismiss;
- held/repeated Escape, rapid backdrop clicks, delayed async callbacks, and
  full MRO dispatch invoke one callback and pop only the intended top screen;
- Prompt Workbench dirty state shows its discard guard for `Escape` and
  backdrop dismissal, while Keep Editing restores editor focus;
- Console Settings reset close requests present the approved three choices,
  failed Undo remains guarded, and active compaction requires the separate
  Close anyway acknowledgement without claiming the provider call was
  cancelled;
- `EnhancedFileOpen` and `EnhancedFileSave` preserve their overlay-first Escape
  stack while backdrop cancellation returns `None` immediately;
- generated-video capacity Escape/backdrop requests keep the staged artifact
  alive until explicit discard confirmation, and cancelling that guard returns
  to the choices;
- nested dismissal removes only the top screen;
- Composer Menu, RAG Settings, and Image Viewer dismiss exactly once while the
  Settings redirected-`Select` click recovery remains functional;
- ordinary dismissal restores opener focus or the Console composer fallback,
  and every cancelled guard restores its recorded focus target;
- `VideoPlayerScreen` treats the whole player as content, adds Escape cleanup,
  and never classifies ordinary frame/status/hint cells as backdrop;
- an explicit inventory of Console-reachable modal types participates in the
  transitive contract, including `EnhancedFileOpen`, `EnhancedFileSave`,
  `ConsoleVideoCapacityModal`, nested `CancelConfirmationDialog`, and
  `ChangeRevertConfirmModal`, while `ConsoleSetupModal` is explicitly excluded
  with its reason.

The tests will be mutation-checked by disabling the shared backdrop branch,
removing the pending latch/top-screen check, and changing a boolean
confirmation's cancel result.

## ADR check

ADR required: yes; amend the existing ADR rather than create a duplicate.

ADR path:
`backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`.

Reason: the reusable cross-module dismissal interface and long-lived Console
interaction grammar meet the repository's ADR threshold. ADR-031 is amended to
own this extension alongside its existing keybinding and truthful-hint rules.
