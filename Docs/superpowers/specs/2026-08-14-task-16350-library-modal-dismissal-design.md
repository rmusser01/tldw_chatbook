# TASK-16350 — Safe Library modal dismissal design

## Status

Approved in conversation on 2026-08-14 after two source and interaction review
passes, then hardened after a dual-agent critique. The approved refinements
cover controller-injected launch paths, enhanced-picker compatibility,
mutation-time close rejection and feedback, stable opener-focus eligibility,
concrete-class behavior, deterministic picker precedence, and real Textual
overlay/key dispatch.

## Problem

Library modals do not share one trustworthy cancellation grammar. Some bind
`Escape`, some expose only a visible Cancel button, and most do not treat a
primary-button backdrop click as a safe cancellation request. A user can
therefore become trapped in a dialog or must hunt for a visible button even
though the equivalent Console surfaces already support both gestures.

A mechanical backdrop handler would be unsafe. Library reaches destructive
Prompt deletion, skill trust, model installation, Git trust and push
authorization, file pickers with transient child surfaces, and a Prompt
collection manager whose create/rename callbacks are asynchronous. Generic
dismissal must never become confirmation, data loss, or a race past active
work.

## Goals

- Every `ModalScreen` transitively reachable from Library has a keyboard and
  backdrop safe-cancel path.
- Terminal `Escape`, primary-button backdrop click, and the visible Cancel or
  Close control converge on the same typed negative result and cancellation
  side effects.
- Descendant overlays, transient picker surfaces, active mutations, destructive
  decisions, and focus restoration keep their existing safety boundaries.
- The launch inventory is exact enough that a future direct, injected, or
  nested modal cannot silently escape the contract.

## Non-goals

- This task does not redesign Library layouts, modal copy, success actions, or
  the Library navigation model.
- Inline confirmation rows and other bounded widgets that are not
  `ModalScreen` instances are excluded; their existing explicit controls remain
  unchanged.
- Backdrop and `Escape` never confirm deletion, installation, trust,
  authorization, save, selection, or mutation completion.
- This task does not introduce a Library-screen event interceptor, an app-wide
  modal registry, a new modal framework, or a new dependency.
- This task does not make every modal in the application dismissible. Shared
  components changed for Library receive one intrinsic safe contract, with
  representative non-Library compatibility coverage.

## Reachability and inventory

Scope is the fixed-point launch graph rooted at the production Library screen,
not a filename or import-name heuristic. “Fixed point” here is deliberately
narrow: an explicit table names the production-default owner files/functions,
their presenter calls, and declared injected presenter seams. The test resolves
modal values passed through those supported edges and joins them with direct
production-route tests for controller/factory seams; it is not a generalized
Python call-graph analyzer. The inventory must inspect:

1. modal constructors in `LibraryScreen`;
2. helper methods and controller callbacks declared as launch owners, including
   `LibraryPromptCollectionsController.push_modal`;
3. nested Library widget owners, including File Notes workspace and Git panel
   classes;
4. modal-to-modal launch edges; and
5. concrete runtime subclasses, even when their safe behavior is inherited.

The reviewed starting population contains:

| Family | Concrete reachable modal types | Safe terminal result |
| --- | --- | --- |
| Skill trust | `SkillTrustPassphraseModal`, `SkillTrustBootstrapModal` | `None` |
| Model artifacts | `ModelInstallModal` | `False` |
| Shared file pickers | `FileOpen`, `FileSave`, `SelectDirectory` | `None` |
| Prompt deletion | `PromptDeleteConfirmationModal` | `PromptDeleteDecision(False, fingerprint)` |
| Database Note folders | `LibraryNoteFolderNameDialog`, `LibraryNoteFolderTargetDialog` | `None` |
| Prompt collections | `PromptCollectionManagerModal` | `None` |
| File Notes | `FileNotesRootDetailsDialog`, `FileNotesConflictCompareDialog` | `None` |
| File Notes Git | `SessionGitTrustDialog`, `PushEndpointDetailsDialog`, `PushDestinationAuthorizationDialog` | `False`, `None`, `False` respectively |
| Already-safe shared surfaces | `WorkbenchHelpPanel`, `PromptVariablesDialog`, `ConfirmationDialog` and concrete reachable subclasses | Existing typed negative |

That table is a reviewed seed, not permission for an incomplete test. The
implementation-time inventory must compare every discovered edge with the
declared launch table in both directions. Injecting an undeclared modal into a
controller path, a nested owner class body, or a modal's own launch helper must
make the inventory test fail. A concrete subclass that overrides `compose()`
must be mounted even when its parent already participates, because an inherited
content selector can otherwise fail only at runtime.

Every declared row records the concrete type, production owner and presenter
edge, factory, stable content selector, visible negative control, exact negative
result or predicate, positive-result type, active guard, focus postcondition,
and whether a non-dismissible exception is allowed. Every current Library row
is dismissible; the exception column exists so a future gate cannot be hidden
in prose. `EnhancedFileOpen` and `EnhancedFileSave` are compatibility-only
clients of shared-base adoption rather than Library-reachable inventory rows.

## Design

### Reuse the shared dismissal boundary

Library modal classes adopt the existing
`tldw_chatbook.Widgets.modal_dismissal.SafeModalDismissMixin`. Each modal
declares a stable `SAFE_MODAL_CONTENT` selector and routes its existing visible
Cancel or Close action through the same safe request used by terminal Escape
and backdrop dismissal.

The mixin remains deliberately small. It classifies backdrop clicks, enforces a
single-shot top-screen dismissal for the current mount generation, and invokes
the modal-owned cancellation action. It does not know how to save, trust,
install, delete, authorize, or complete a mutation. Existing modal-specific
cleanup and typed results remain in the owning modal.

Textual walks matching handlers across the MRO. Adopting the mixin therefore
requires removing explicit lifecycle `super()` calls or duplicate click/cancel
handlers that would execute the same behavior twice. Mounted lifecycle tests
must prove one mixin mount/unmount and one dismissal for the real concrete
classes. The same rule applies to decorated inherited handlers: in
`SelectDirectory`, the decorated `DirectoryNavigation.Changed` handler
explicitly calls the independently decorated base handler, so adoption must
ensure breadcrumb/error/recent-hook behavior runs exactly once rather than
through both Textual MRO dispatch and a manual parent call. The base currently
also clears the same error twice through `_on_directory_changed()` and a
separate decorated `_clear_error()` handler. Keep `_on_directory_changed()` as
the single owner of error clearing and remove the redundant handler rather than
preserving duplicate work behind the new contract.

Shared-base adoption also changes the inheritance beneath the already-safe
`EnhancedFileDialog`, which currently declares
`SafeModalDismissMixin, BaseFileDialog`. After `FileSystemPickerScreen` adopts
the mixin, `EnhancedFileDialog` must inherit only `BaseFileDialog`; retaining
the explicit mixin would create an inconsistent MRO and break imports. Its
`action_smart_dismiss`, `_SUPPRESSED_BASE_HANDLERS`, stable content selector,
recent-location persistence, typed results, and enhanced open/save behavior
remain unchanged. `EnhancedFileOpen` and `EnhancedFileSave` are mandatory
compatibility tests, including an application import smoke test.

### Backdrop classification

A primary-button click is inside when the original event widget is the content
widget or one of its descendants, or when its screen coordinates fall inside
the content region. Descendant ownership keeps an expanded `SelectOverlay`
inside even when it paints outside the bounded panel; geometry keeps padding and
blank cells inside the panel from looking like backdrop.

Only a classifiable primary-button click outside both boundaries is a backdrop
request. Non-primary clicks and inputs with unknown provenance fail closed.
The event is consumed before cancellation so a rapid second click cannot reach
the newly revealed Library surface.

Stable content selectors are added where needed. `ModelInstallModal` uses its
existing `.model-install-modal` class because its outer container ID is
caller-configurable. Folder and File Notes dialogs receive explicit stable
outer IDs rather than positional selectors.

### Escape precedence

Escape follows this order:

1. the topmost descendant overlay or transient child surface handles it;
2. a modal-owned active-work or destructive guard may reject or transform the
   close request; then
3. the modal requests its exact terminal safe negative result.

An expanded `SelectOverlay` therefore closes before its owning modal. Tests
must dispatch a real Escape key through Textual rather than call a helper
directly.

The shared file-picker base needs source-aware behavior:

- a focused descendant overlay receives Escape first; after that, terminal
  picker Escape uses the deterministic order path editor, search, then recent
  locations before dismissing the picker;
- visible Cancel and a primary-button backdrop click are terminal cancellation
  and return `None` immediately; and
- clicks on the path editor, search, recent list, picker options, and a real
  `SelectOverlay` remain inside.

Because the base picker is reused broadly, the safe contract belongs on
`FileSystemPickerScreen`, not on Library-only wrappers. Representative existing
non-Library `FileOpen`, `FileSave`, and `SelectDirectory` flows must remain
green, including their success values and current navigation behavior. The
base picker currently exposes no-op recent-location hooks; this task preserves
that seam and does not add persistence. The implementation must reconcile both
the current `SelectDirectory.on_mount()` explicit `super()` call and its
decorated navigation-change parent call with Textual's full-MRO event dispatch.

### Modal-owned negative results

- Skill passphrase/bootstrap and ordinary detail/folder dialogs return `None`.
- `ModelInstallModal` returns exact `False`; cancellation cannot start install
  work or satisfy an acknowledgement.
- `PromptDeleteConfirmationModal` returns
  `PromptDeleteDecision(False, request.fingerprint)`, preserving stale-result
  validation while never selecting Delete.
- `SessionGitTrustDialog` and `PushDestinationAuthorizationDialog` return exact
  `False`; endpoint details return `None`.
- `PromptCollectionManagerModal` returns `None` on cancellation and never
  publishes its positive browse/membership result from a generic gesture.
- Existing safe shared dialogs retain their current exact negative values and
  callbacks.

### Prompt collection mutation guard

The current create/rename button handlers await the mutation callback on the
screen message pump. That can delay Escape/backdrop dispatch until after
`_mutation_in_flight` resets, allowing a close gesture made during the mutation
to execute later as though it were new.

Create, rename, and their mutation retry path will synchronously claim the
existing in-flight state, disable mutating, selection, Done, and retry controls,
and start the awaited mutation in a screen-owned Textual worker. Visible Cancel
remains enabled solely as a guarded close request; its handler, Escape, and
backdrop input are consumed while the flag is true and cannot dismiss the
modal. Completion may repaint only the same mounted generation; a stale
completion cannot clear or dismiss a remounted instance. The task does not add
a generalized operation manager.

The first close request received during one active mutation updates the existing
outcome/status line to “Finish the current collection change before closing.”
Later close requests during that same mutation are consumed without stacking
notifications or changing state. Mutation progress and final success/failure
still replace that line through the existing outcome path; no toast or second
guard state machine is added.

Catalog loading keeps its existing independent request-token behavior unless a
focused regression proves it shares the same close race.

### Focus restoration

The shared mixin currently holds only a weak reference to the opener. Library
can recompose its canvas while a modal is open, replacing that widget with a
new instance that has the same stable ID. At mount, the mixin will record both
the weak reference and the opener's non-empty widget ID. After safe dismissal
it evaluates the exact object and any ID-resolved replacement with the same
eligibility predicate: mounted and attached, displayed and visible, enabled,
and focusable. It restores the exact object only when eligible; otherwise it
resolves that ID on the revealed screen and focuses the single eligible match.
It never guesses a different Library control.

The existing Console composer fallback remains Console-specific. A Library
modal with neither the original opener nor one exact ID match leaves focus to
the revealed screen's normal policy rather than focusing an unrelated action.
Nested modal cancellation restores focus to the control that opened the nested
screen when that exact identity remains valid.

Existing File Notes callbacks that also restore focus are removed when the mixin
becomes the single owner, or are retained only with a mounted exact-once test
proving the duplicate path is harmless and idempotent.

### One-shot and stale-input safety

The existing top-screen, pending-latch, mount-generation, and click-through
shield rules remain authoritative. Repeated Escape/backdrop/Cancel input may
produce one negative callback and remove one screen. A delayed mutation or
cancel callback from an earlier mount cannot dismiss, refocus, or clear state
on a later presentation of the same modal instance.

## Testing

Use focused Textual tests only; do not run broad test directories or the full
repository suite for this task.

1. Add an explicit, bidirectional Library owner-edge inventory covering direct,
   controller-injected, nested-widget, and modal-to-modal presenters. Mutation
   tests inject an undeclared alias edge into each non-obvious owner category.
   Direct production-route tests cover injected controller/factory seams; do
   not build a generalized Python call-graph analyzer.
2. Give every concrete reachable modal one contract row and mount it. For every
   row, assert the selector exists and verify visible Cancel, terminal Escape,
   and primary backdrop return its exact negative value or predicate. No result
   family may stand in for an untested concrete sibling.
3. Dispatch non-primary and inside-content clicks through Textual and prove no
   dismissal or callback.
4. Drive a real expanded `SelectOverlay`: first Escape closes the overlay and
   keeps the modal; the next Escape safely closes the modal. While the overlay
   is expanded, prove an overlay-option click remains inside, while a true
   backdrop click and the visible Cancel control remain terminal safe-negative
   requests rather than merely peeling the overlay.
5. Drive real file-picker path/search/recent state and real key dispatch. Verify
   terminal Escape peels transient state, while backdrop and visible Cancel
   return `None` immediately. Include one combined-state case that proves the
   fixed path-editor, search, recent precedence after descendant overlays.
6. Gate Prompt collection create/rename callbacks with `asyncio.Event`. During
   the gate, dispatch Escape, backdrop, and a real `pilot.click` on the enabled
   visible Cancel control and prove the modal remains. The first request updates
   the existing status line exactly once, later requests do not stack feedback,
   the operation settles once, and no queued close fires afterwards. Also prove
   stale completion cannot mutate a remounted presentation.
7. Recompose the underlying Library canvas while a modal is open and prove
   dismissal focuses the replacement widget with the same ID. Prove that a
   missing/duplicate/ineligible ID and an ineligible still-mounted original do
   not focus another control.
8. Verify Textual full-MRO mount/unmount, cancellation, and decorated inherited
   navigation handlers execute exactly once. Include `SelectDirectory` and any
   existing subclass with its own handlers; assert one breadcrumb/recent-hook
   update and one error-clear update per real `DirectoryNavigation.Changed`
   event.
9. Run representative non-Library shared-picker regressions alongside the
   Library tests to pin typed success values and existing navigation behavior;
   do not turn the base's no-op recent-location hooks into a new feature.
10. Run the mounted `EnhancedFileOpen`/`EnhancedFileSave` safe-dismiss suites
    and an application import smoke test after removing the now-redundant
    explicit mixin base. Preserve enhanced smart-dismiss order, handler
    suppression, persistence, and typed results.

Mutation checks must show the tests go red when the backdrop branch, exact
negative result, mutation guard, stable-ID restoration, or one fixed-point
inventory edge is removed, then return green after restoration.
Implementation Notes record the exact mutations performed so the RED evidence
remains auditable.

## Alternatives considered

### Intercept dismissal in `LibraryScreen`

Rejected. A covered screen is the wrong event boundary and cannot know each
modal's typed result, transient state, active mutation, or nested-screen
ownership.

### Add duplicate Escape and click handlers to every modal

Rejected. It repeats the subtle backdrop, one-shot, generation, and focus rules
and would continue drifting. The existing shared mixin is the established
boundary.

### Wrap shared file pickers only at Library call sites

Rejected. Library uses the same picker classes as the rest of the app, and
wrapper-specific MRO handlers would compete with inherited Textual event
handlers. The shared base should have one intrinsic safe contract, protected by
representative compatibility tests.

### Store only the opener object for focus restoration

Rejected. Library recomposition can legitimately replace the object while
preserving its semantic widget ID. Exact ID re-resolution is the smallest
stable fallback and avoids a Library-specific focus heuristic.

## ADR check

ADR required: yes; amend the existing ADR rather than create a duplicate.

ADR path:
`backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`.

Reason: task-16350 extends the established cross-module modal cancellation
contract to a second long-lived application surface and changes the shared
file-picker and focus-recovery interfaces. ADR-031 already owns this interaction
grammar.
