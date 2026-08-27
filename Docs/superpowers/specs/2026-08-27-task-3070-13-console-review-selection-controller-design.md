# TASK-3070.13 Console Review and Selection Controller Design

**Status:** design direction approved by the owner 2026-08-26; written amendment
prepared for final review 2026-08-27

**Task:** `TASK-3070.13 - Extract Console review and selection workflow ownership`

**Depends on:** TASK-3070.12 and the approved
`2026-08-23-console-decomposition-wave6-closeout-amendment.md`

## Context and Amendment Trigger

The Wave 6 closeout amendment characterized a 26-method, 1,114-line
review/selection family at its historical base. It also required a later child to
stop and amend rather than silently reclassify the family if a rebase invalidated
that exact inventory.

That condition now applies. TASK-3070.12 merged, and an independent per-turn
changed-files simplification removed these ten historical methods before
TASK-3070.13 began:

- `_build_console_changed_files_state`
- `_console_changed_files_scope`
- `_console_changed_files_section_enabled`
- `_dispatch_console_changed_files_worker`
- `_land_console_changed_files`
- `_land_console_changed_files_empty`
- `_on_console_change_review_dismissed`
- `_sync_console_changed_files_if_scope_changed`
- `_sync_console_changed_files_section`
- `handle_console_changed_files_selected`

Recreating deleted changed-files coordination would restore obsolete behavior.
Marking the task superseded would leave seven surviving policy methods on
`ChatScreen` and fail the parent task's ownership objective. This design amends only
the current implementation boundary for TASK-3070.13; the original inventory and
arithmetic remain immutable historical evidence.

The exact implementation base is `origin/dev` `ee8dc24115`. At that revision,
`ChatScreen` is 17,599 physical lines with 539 direct methods. The surviving coherent
family is 16 methods and 840 physical lines.

The focused pre-change baseline is 111 passing tests out of 115. The four failures
are pre-existing stale unit assertions in `test_console_annotation_markers.py`:
assistant-turn grouping now nests annotation rows inside the top-level
`assistant-turn` row, while those assertions still search only top-level rows. The
mounted annotation, persistence, feedback, note, review-note, trajectory, and
architecture coverage passes. This task may correct those four test traversals but
must not change product behavior to satisfy them.

## Goals

1. Give the seven surviving policy methods one explicit, non-DOM owner.
2. Preserve the three Textual event/action boundaries as complete delegates of at
   most five physical lines each.
3. Retain the six exact DOM, modal, composer, dismissal, and ADR-068 review-note
   methods on `ChatScreen`.
4. Move annotation preview/load identity and selection-feedback exclusion state to
   the new owner without mirrored state or stale compatibility behavior.
5. Preserve Git, run-store, note, annotation, prompt-queue, privacy, cancellation,
   and user-visible behavior.
6. Remove at least 399 physical `ChatScreen` lines and seven direct methods without
   raising a ratchet.

## Non-goals

- Reintroducing the deleted changed-files section or its worker/state machinery.
- Redesigning change review, annotations, review notes, feedback copy, note format,
  selection UX, trajectory presentation, or persistence schemas.
- Moving Git, database, run-store, prompt-queue, or transcript rendering authority.
- Moving ADR-068's review-note fetch, optimistic edit/delete wrappers, modal, or
  forced inline preview reload out of `ChatScreen`.
- Moving composer quote insertion, selection-menu dismissal, or change-review screen
  presentation out of `ChatScreen`.
- Adding a mixin, dynamic method facade, ambient screen reference, sibling-controller
  reference, or shadow compatibility state.
- Running a local full test suite. Required GitHub Actions remain the broad
  integration gate.

## Current-Base Method Inventory

Physical spans are measured from the exact implementation base. The classification
is exhaustive and binding for this task.

| Method | Lines | Classification | Reason |
|---|---:|---|---|
| `_console_change_review_provider` | 35 | move | collaborator and run-state policy; no DOM |
| `_console_change_review_run_id` | 35 | stay | reads the mounted transcript before store fallback |
| `_console_change_review_workspace_roots` | 27 | move | execution-context policy; no DOM |
| `_console_review_notes_flow` | 217 | stay | explicit ADR-068 screen-owned modal/fetch/mutate/reload flow |
| `_console_selection_feedback_flow` | 44 | move | modal-result sequencing, audit ordering, and prompt dispatch |
| `_console_selection_quote_requested` | 24 | stay | Textual event plus composer DOM insertion |
| `_create_console_selection_note` | 52 | move | validation, provenance, persistence, and privacy policy |
| `_dismiss_console_selection_menus_outside_transcript` | 64 | stay | screen DOM ancestry and presentation cleanup |
| `_load_console_annotation_previews` | 28 | move | off-thread load, stale-conversation guard, and re-keying policy |
| `_open_change_review` | 55 | stay | constructs and pushes the presentation screen |
| `_record_console_feedback_event` | 60 | move | durable audit/annotation policy and immediate preview update |
| `_sync_console_annotation_discovery` | 34 | move | conversation transition and worker-dispatch policy |
| `action_open_trajectory_view` | 62 | delegate | Textual binding must remain on the screen |
| `on_console_review_notes_requested` | 31 | stay | ADR-068 event ownership and inflight contract |
| `on_console_selection_feedback_requested` | 48 | delegate | Textual `@on` boundary must remain on the screen |
| `on_console_selection_note_requested` | 24 | delegate | Textual `@on` boundary must remain on the screen |
| **Total** | **840** | **7 move / 3 delegate / 6 stay** | |

The seven move methods contain 280 lines. The six stays contain 426 lines. Capping
each complete delegate at five lines leaves at most 441 screen lines from this
family, so the minimum net removal is 399 lines and seven direct methods. The
conservative post-extraction screen is therefore at most 17,200 lines and 532 direct
methods. Actual results may be smaller; the projection is not a new ratchet.

## Considered Approaches

### 1. Dedicated `ConsoleReviewSelectionController` (chosen)

Create `tldw_chatbook/UI/Console_Modules/review_selection.py`, construct one
controller through the existing `build_console_controllers()` seam, and install it as
`screen._review_selection`. Move the seven policy methods and the controller-side
implementations behind the three delegates.

This creates one inspectable owner, follows `DESIGN.md` section 7, preserves the
framework boundary, and supports direct tests with plain fakes.

### 2. Close TASK-3070.13 as superseded

This correctly avoids restoring deleted changed-files code but leaves seven policy
methods and their mutable state on `ChatScreen`. It does not satisfy the parent
ownership acceptance criterion and is rejected.

### 3. Recreate the historical 26-method family

This would regress the independently simplified per-turn review path solely to make
old inventory arithmetic match. It is rejected.

### 4. Mixin or broad screen delegates

A mixin preserves ambient access to all screen state. Delegating every moved method
preserves direct-method inventory and dual navigation paths. Both hide rather than
improve ownership and are rejected.

## Ownership and Module Boundary

`ConsoleReviewSelectionController` owns:

- change-review provider construction and live workspace-root resolution;
- annotation discovery transitions, background loading, stale-result rejection, and
  native-message re-keying;
- selection-feedback mutual exclusion, comment result handling, message composition,
  durable audit ordering, annotation sidecar writes, and prompt-queue dispatch;
- selection-note validation, title/provenance derivation, off-thread persistence,
  privacy-safe failure logging, and notification policy;
- trajectory snapshot orchestration and background build sequencing.

The controller does not query the DOM, focus widgets, construct or push Textual
screens/modals, reference `ChatScreen`, or hold sibling-controller objects. External
work is expressed through named, late-bound callables installed in
`UI/Console_Modules/wiring.py`.

`ChatScreen` retains exactly:

- `_console_change_review_run_id`;
- `_console_review_notes_flow`;
- `_console_selection_quote_requested`;
- `_dismiss_console_selection_menus_outside_transcript`;
- `_open_change_review`;
- `on_console_review_notes_requested`.

It also retains the three binding methods as complete delegates:

- `action_open_trajectory_view`;
- `on_console_selection_feedback_requested`;
- `on_console_selection_note_requested`.

Each delegate may stop its event where applicable and invoke one controller entrypoint.
It contains no validation, inflight, worker, persistence, modal-result, or dispatch
policy and spans at most five physical source lines.

## Construction and Dependencies

`build_console_controllers()` remains the single construction API and adds
`screen._review_selection`. Its docstring and controller count are updated; no second
wiring path is introduced.

Named callables cover these capabilities:

- current/ensured Console chat controller and store;
- current/ensured agent bridge and current run state;
- native Console messages;
- worker scheduling and UI-thread callback marshaling;
- prompt-queue dispatch;
- feedback-comment modal wait;
- trajectory-screen presentation;
- native transcript synchronization;
- user notifications.

The presentation callbacks may close over the screen in wiring, but the controller
does not retain or discover the screen. The trajectory presenter owns lazy
`TrajectoryScreen` import/construction and `app.push_screen`; the controller owns
only the snapshot inputs, worker sequencing, and handoff. The feedback-modal callback
similarly owns Textual modal construction/wait while returning only the comment
result to the controller.

Sibling behavior is reached through purpose-named callables, never by passing
`screen._agent`, `screen._prompt_queue`, or another controller object.

## State and Compatibility

The controller is the sole owner of:

- `annotation_loaded_conversation: str | None`;
- `annotation_previews: dict[str, tuple[str, ...]]`;
- `selection_feedback_inflight: bool`.

`ChatScreen` keeps temporary private compatibility names
`_console_annotation_loaded_conversation`, `_console_annotation_previews`, and
`_console_selection_feedback_inflight` as fail-loud read/write descriptors:

- reads and writes forward to `screen._review_selection`;
- access before controller wiring raises an actionable `RuntimeError`;
- no descriptor or screen instance stores a shadow copy;
- mutable-map identity is preserved so existing in-place preview updates and focused
  tests continue to observe the controller-owned map.

`_console_review_notes_inflight` remains screen-owned. The review-note flow may call
the controller's annotation loader during its forced reload, but it continues to own
the modal, database-bound edit/delete wrappers, conversation-current guard, and
post-change transcript refresh required by ADR-068.

## Runtime Flows

### Annotation discovery and reload

The existing transcript sync path calls the controller directly with the active
store. The controller compares the persisted conversation id with its loaded id,
clears previews on transitions, and schedules one named worker. The loader performs
the database read off-thread, rejects a result after a conversation switch, maps
persisted ids back to native message ids, and replaces the preview map.

The screen-owned review-note flow continues to force an inline reload while idle by
awaiting the controller loader, then synchronizing the transcript. This preserves the
live-verification fix recorded in ADR-068.

### Selection feedback

The Textual handler stops the event and passes action, quote, and anchor id to the
controller. The controller rejects blank or duplicate in-flight requests, arms its
guard, and schedules the existing non-exclusive worker flow. The flow waits through
the named modal callback, composes identical feedback text, records audit and optional
annotation off-thread before dispatch, sends through the named prompt-queue callable,
and releases the guard on every exit.

Audit failure remains non-fatal. Comment annotations update the controller-owned
preview map immediately. Raw selected text never enters new logs or diagnostic fields.

### Selection note

The Textual handler stops the event and delegates the quote. The controller rejects
blank input, schedules a worker, validates the bounded untrusted text, derives the
same title and provenance content, writes through the existing database off-thread,
and emits markup-safe confirmation. Failure logging continues to expose title length
only, never selected content.

### Change review

Existing callers request provider and workspace roots directly from the controller.
The provider still joins on the same agent conversation id and receives live
run-active callbacks. Workspace roots still derive from the active turn execution
context and degrade to `None` on missing collaborators or exceptions.

`_open_change_review` and `_console_change_review_run_id` remain screen-owned. They
retain transcript-first run-id resolution, lazy `ChangeReviewScreen` import,
constructor arguments, notification, and `push_screen` presentation.

### Trajectory view

The Textual action delegates to the controller. The controller resolves the current
persisted conversation, title, run database, and snapshot builder; emits the same
empty-state/building notifications; and builds off-thread. The named UI-thread and
presentation callbacks push the same live-tail `TrajectoryScreen`. No controller DOM
or Textual screen dependency is introduced.

## Error Handling and Privacy

The extraction preserves the existing degrade posture:

- missing change-review collaborators return `None` rather than raising;
- annotation load failures warn without logging note bodies;
- feedback audit failure never prevents feedback dispatch;
- selection-note failures never log selected text or derived title;
- stale annotation loads never paint previews from a prior conversation;
- every feedback exit releases mutual exclusion;
- UI callbacks remain marshaled through the existing app boundary.

No API key, prompt body, quote, note content, raw database row, or provider failure is
added to diagnostics, notifications, or logs.

## Verification Strategy

Implementation follows red-green-refactor and uses focused local gates only.

### RED-first ownership tests

Add a source-inspected architecture suite proving:

- all seven exact move methods are absent from `ChatScreen` and owned by
  `ConsoleReviewSelectionController`;
- the six exact stays remain on `ChatScreen`;
- the three exact delegates retain their binding/action surfaces, contain only the
  allowed handoff, and fit the five-line ceiling;
- the controller has no Textual DOM queries, screen reference, mixin, dynamic facade,
  sibling-controller object, Git authority, or database implementation;
- wiring is the sole constructor and supplies the reviewed named dependencies;
- compatibility descriptors are read/write, fail loudly before wiring, and store no
  shadow state;
- the current-base family spans and conservative 399-line / seven-method removal are
  met without a ratchet increase.

### Isolated controller tests

Use plain fakes to cover:

- change-review provider/root success and graceful degradation;
- annotation transition clearing, stale-load rejection, native-id re-keying, and load
  failure behavior;
- feedback blank/duplicate guards, cancel, dispatch composition, audit-before-send,
  annotation preview update, and `finally` release;
- note validation, provenance/title caps, off-thread persistence, markup-safe success,
  and privacy-safe failure;
- trajectory missing-conversation behavior and worker-to-presentation handoff.

### Mounted and repository regression tests

Retarget focused tests only where ownership or patch handles move. Correct the four
stale annotation-marker assertions to inspect the nested assistant-turn rows while
keeping their product expectations unchanged. Run the related annotation, selection,
change-review opener, trajectory, turn-undo, controller-wiring, and architecture
suites. Do not run a local full suite.

Use bounded manual mutation checks after a checkpoint commit for the highest-risk
policy branches: stale annotation result rejection, feedback inflight release,
audit-before-dispatch ordering, and selection-note privacy logging. Each temporary
semantic edit must make its exact focused node fail, then be restored with
`apply_patch` and rerun green. The implementation plan will name the precise edits and
commands.

Run targeted Ruff check/format on modified Python, isolated compile on modified
production modules, diagnostic inventory generation/validation, backlog task-ID
validation, `git diff --check`, and the Wave 6/current-boundary architecture gates.
Review the final diff for behavior drift, accidental authority transfer, privacy
widening, or unrelated changes.

## Delivery Sequence

1. Freeze the current 7/3/6 boundary and state/wiring contracts with RED tests.
2. Add the controller, state descriptors, and plain-fake tests.
3. Move annotation, change-review, feedback, note, and trajectory policy in coherent
   slices, keeping focused tests green after each slice.
4. Install the three complete delegates and retarget direct call sites/patch handles.
5. Correct the four stale nested-marker unit traversals without production changes.
6. Run focused behavioral, mutation, architecture, static, privacy, diagnostic,
   backlog-ID, and diff gates.
7. Record task evidence and rebase on the latest `origin/dev`. If that rebase changes
   the reviewed 16-method family, invalidates 7/3/6 ownership, or breaks the
   conservative reduction, stop and amend again before delivery.

## ADR Check

ADR required: no.

ADR path: `backlog/decisions/068-console-text-selection-and-annotations.md`.

Reason: this task applies the accepted screen/controller ownership rule in
`DESIGN.md` section 7 and preserves ADR-068's explicit screen ownership for review
note fetching, optimistic edit/delete wrappers, modal presentation, and forced inline
reload. It changes neither storage/schema, Git or database authority, service
contracts, security/privacy policy, dependency/tooling, nor long-lived UX structure.
