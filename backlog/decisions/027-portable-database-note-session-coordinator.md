# ADR-027: Portable Database Note Session Coordinator

Status: Accepted
Date: 2026-07-30
Related Task: [TASK-1333](../tasks/task-1333%20-%20Adapt-Library-Notes-for-lossless-60x20-workflow.md)
Supersedes: N/A

## Decision

Database Notes use a host-independent `DatabaseNoteSessionCoordinator` for the
active note's in-memory editing and persistence session.

The coordinator lives in
`tldw_chatbook/Library/library_notes_session.py` and owns:

- the persisted Database Note baseline and optimistic-lock version;
- the canonical raw title, body, and keywords draft;
- monotonic draft and saved revisions;
- dirty, saving, error, and editor-conflict state;
- serialized and coalesced save requests;
- conflict-resolution generation and operation gating;
- untouched-new-note eligibility and stale-operation gating;
- conditional Reload and revision-safe Overwrite;
- the pending-work flush result used by navigation guards;
- immutable session snapshots and typed outcomes for its host.

The coordinator depends on an injected asynchronous Database Note session port
for detail fetch and versioned save operations. The port is limited to the
existing Database Note service boundary. It does not expose SQLite handles,
Textual widgets, File Notes, filesystem sync policy, navigation, focus, or
global application state.

The port returns one normalized detail value containing note id, exact title,
exact body, semantic keyword tokens, optimistic-lock version, and
created/modified metadata. A host adapter may combine existing service calls
to construct that value, but the coordinator never observes a partially loaded
detail.

`LibraryScreen` remains the Adapt host and owns:

- Library route and compact/wide stage selection;
- Navigator, Editor, Preview, Context, Create, and Sync transitions;
- Textual focus, caret, selection, and scroll capture/restoration;
- Textual worker/timer lifecycle and autosave debounce scheduling;
- the adapter from the existing Database Note service to the coordinator port;
- visible status/effect presentation and navigation decisions.

The host calls the coordinator for draft mutation, save, conflict resolution,
and flush. It renders immutable coordinator snapshots through
`LibraryNotesCanvas`. Navigation never infers persistence success from widget
state; it proceeds only from a successful coordinator flush outcome.

`LibraryNotesCanvas` stays presentation-only. It receives explicit state,
emits user intent, and neither calls the Database Note service nor owns
persistence tasks.

`library_notes_state.py` retains pure immutable display/session types and
transition helpers shared by the coordinator and presentation. It owns no
asyncio, services, database access, or Textual lifecycle.

The coordinator is scoped to Database Notes. ADR-021's
`FileNotesSessionController` remains a separate file-authority controller and
does not share a generic write path with Database Notes. A later dedicated
Notes workbench may reuse the Database Note coordinator through the same port
without moving Database ownership into the UI host.

## Context

The Library Notes Adapt work adds a canonical draft, revisioned save queue,
conflict-operation gating, navigation flush, and recompose-safe state
rehydration. Keeping all of that orchestration in the existing
`LibraryScreen` would further concentrate permanent Notes behavior in a screen
that already owns many unrelated Library canvases.

Pure transition helpers alone do not make the asynchronous save/conflict state
machine portable. The later dedicated Notes workbench would have to extract or
rewrite coupled worker, conflict, and flush behavior from `LibraryScreen`.

At the same time, Adapt must not migrate route ownership, create the dedicated
workbench early, change Database Note storage, or merge Database and File Notes
authority. A small host-independent coordinator isolates only the session logic
that must survive that later move.

## Required Boundaries

- The coordinator imports no Textual screen, widget, message, worker, timer,
  focus, CSS, or navigation type.
- The coordinator imports no File Notes repository, controller, recovery
  store, filesystem sync engine, or filesystem mutation type.
- The session port exposes only the minimum async detail-fetch and
  optimistic-version save operations needed by the active Database Note.
- The coordinator never constructs or owns the application Database Note
  service. The host injects the port adapter.
- Autosave debounce timing remains host-owned; serialization and coalescing
  after a save request reaches the coordinator are coordinator-owned.
- Only one persistence or editor-conflict resolution operation may be active
  for a session. Every completion is gated by note identity, session
  generation, and operation token.
- The coordinator owns whether the current note is the untouched result of the
  active create token. The Textual host may execute the existing delete service
  only after receiving that gated discard eligibility; the coordinator does
  not absorb general note deletion or creation ownership.
- Before discard or general Delete, the host must obtain typed destructive
  admission from the coordinator. Admission atomically blocks draft mutation,
  save/autosave, and duplicate destructive operations until Cancel, failure, or
  completion. The host revalidates note identity, session generation, version,
  and any create token immediately before its service call.
- Raw drafts remain raw in session state. Persistence payload construction must
  not truncate, strip, or rewrite title/body/keyword content. A transformation
  or limit violation returns a typed validation veto, makes no service call,
  retains dirty state, and cannot report Saved.
- A successful explicit no-op Save clears untouched-new-note discard
  eligibility because it records the user's intent to keep the note.
- Coordinator snapshots contain no live widget, task, lock, service, database,
  or application references.
- The coordinator returns typed outcomes for saved, validation-vetoed, failed,
  conflicted, missing-note, reload-vetoed, destructive-admission, and
  flush-vetoed states. It does not navigate or focus widgets in response.
- A whole-screen Textual recompose may replace the canvas but not the
  coordinator object owned by the still-live `LibraryScreen`.
- Controlled route replacement must cross the coordinator flush barrier.
  Abrupt process termination and crash-draft durability remain outside Adapt.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep all session orchestration in `LibraryScreen` | Deepens an already large screen, spreads the save invariant across UI handlers, and makes the later workbench migration riskier. |
| Use only pure state helpers | Pure transitions are valuable but do not encapsulate asynchronous save serialization, coalescing, conflict tokens, or flush outcomes. |
| Put the coordinator inside `LibraryNotesCanvas` | Couples persistence to presentation, makes recomposition unsafe, and violates ADR-011's explicit-state widget boundary. |
| Build the dedicated Notes workbench now | Expands Adapt into the later Shape phase and changes route/application structure before the compact safety foundation is proven. |
| Share one generic controller with File Notes | Database and File Notes have different authorities, recovery guarantees, repositories, and conflict semantics under ADR-021. |

## Consequences

### Benefits

- The no-silent-loss invariant has one testable owner.
- Library resize/recompose code cannot accidentally become the persistence
  authority.
- The later dedicated Database Notes workbench can reuse the coordinator
  without rewriting save/conflict behavior.
- Database and File Notes remain explicitly separate while allowing a future
  host to present both sources.
- Coordinator concurrency tests can run without mounting Textual.

### Accepted trade-offs

- Adapt adds one focused module and an injected port instead of keeping all
  logic in one screen.
- `LibraryScreen` still owns UI lifecycle, focus, navigation, and debounce
  timing, so moving the host later remains real work.
- Wide and compact presentations may expose different arrangements of the same
  actions, but they consume one coordinator state.
- Introducing the coordinator does not add history, undo, backlinks, file
  authority, or crash-draft persistence.

## Links

- [Library Notes Adaptive 60×20 Design](../../Docs/superpowers/specs/2026-07-30-library-notes-adaptive-60x20-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-015: Shell Destination IA](015-shell-destination-ia.md)
- [ADR-021: File-Backed Notes Disk Authority and Recovery Replica](021-file-backed-notes-disk-authority-and-recovery.md)
- [ADR-022: Textual 8 Runtime Floor](022-textual-8-runtime-floor.md)
