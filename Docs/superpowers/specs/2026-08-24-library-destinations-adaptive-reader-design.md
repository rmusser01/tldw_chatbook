# Library Destinations Adaptive Reader Migration Design

Date: 2026-08-24

Status: Approved

ADR: [ADR-086](../../../backlog/decisions/086-library-adaptive-reader-shell.md)

Related: [Library Media NetNewsWire reader design](2026-08-23-library-media-netnewswire-reader-design.md), [ADR-084](../../../backlog/decisions/084-library-media-reader-ia.md), [Library compose-once design](2026-08-13-library-compose-once-design.md)

## Summary

Migrate Library Conversations, Notes, Prompts, and Skills to the reader-shaped interaction model
shipped by Library Media. Each destination will present three stable spatial roles:

- **Library** — the existing destination rail, independently collapsible.
- **Destination list** — Conversations, Notes, Prompts, or Skills, independently collapsible.
- **Work pane** — the permanent destination-owned reader, editor, trust surface, or workflow.

The migration introduces one Library-local adaptive structural shell. It shares pane composition,
grips, responsive geometry, focus-region registration, and preference plumbing while leaving each
destination's records, services, builders, controllers, validation, mutations, conflicts, trust,
imports, and recovery behavior authoritative. It does not create a generic editor or action
framework.

The programme is delivered in four PRs, in order: Conversations, Notes, Prompts, then Skills. The
first PR extracts the shared shell from Media and migrates Conversations. Media keeps its domain
behavior; the only intended visual geometry change is the approved rule that reclaimed Library
width expands the destination list toward its comfort cap before flowing to the work pane.

## Goals

- Give the five primary Library browsing destinations one coherent scan/select/work grammar.
- Keep the destination list mounted while reading, editing, importing, syncing, reviewing trust,
  or recovering from an error.
- Let users collapse Library and the destination list independently without hiding the permanent
  work pane.
- When Library collapses, use its released width to show fuller destination titles and details
  before giving remaining surplus to the work pane.
- Preserve every existing user-facing capability and its current domain authority.
- Keep selection and loaded work identity truthful under rapid traversal and slow workers.
- Preserve deliberate layout choices while allowing temporary responsive adaptation.
- Remain usable from 80x24 through wide desktop terminals with keyboard-first operation.
- Deliver four independently releasable migrations rather than one all-destinations cutover.

## Non-goals

- Building an application-wide split-pane framework.
- Sharing implementation with Watchlists in this programme.
- Creating a schema-driven editor, generic field renderer, generic action registry, or workflow DSL.
- Redesigning destination databases, sync rules, trust policy, import formats, or lifecycle semantics.
- Inventing new primary actions or making previously read-only supporting files editable.
- Redesigning Media content modes, backend scope, or destructive-action contracts.
- Migrating Collections, Search/RAG, Ingest, Export, Trash, Study, or File Notes into this shell.
- Replacing transient confirmations, file pickers, credential prompts, or system dialogs with
  in-pane imitations.

## Current state and authoritative owners

Media already uses a permanent Reader beside its Items list and has a pure responsive-layout
resolver, independently collapsible panes, generation-fenced detail loading, and persisted manual
pane preferences. Conversations, Notes, Prompts, and Skills still use destination-specific canvas
takeovers in which selecting or creating an item can replace the list with a reader or editor.

The existing destination state builders and controllers remain the source of truth:

- Conversation records and saved message history remain owned by the conversation service and
  current conversation state.
- Database Notes retain their editor, templates, sync, conflict, import, utilities, and recovery
  contracts.
- Prompts retain their existing local/server fields, history, collections, provenance, import,
  validation, and lifecycle behavior.
- Skills retain their local store, trust boundary, import, supporting-file, and recovery contracts.
- `LibraryScreen` remains the orchestration owner and continues to use scoped canvas replacement
  under the compose-once contract. The adaptive shell does not become a second router.

Before each migration, its PR must record a before/after capability inventory. An existing action
may move, but it may not disappear, change authority, or acquire new semantics without separate
approval.

## Architecture and ownership

### Library-local adaptive shell

`LibraryAdaptiveReaderShell` is a structural Textual widget with three content slots and two
full-height grips. It owns only:

- mounting the Library, destination-list, and work-pane regions;
- keeping both grips mounted, focusable, clickable, and keyboard operable;
- applying effective visibility and exact cell widths in place;
- reporting settled shell-width changes without causing data work;
- exposing stable region identities to the application's existing global focus cycle; and
- composing late-bound destination widgets from current state snapshots.

The shell accepts builders or current widget/state inputs at composition time. It must not cache
removable widget instances or reconstruct destination state internally. Region content remains
concrete: Conversations, Notes, Prompts, Skills, and Media each own their list and work widgets.

### Pure geometry policy

A pure Library layout module resolves one effective layout from:

- available shell width after application chrome;
- requested open/collapsed state;
- requested fixed or custom widths;
- the destination list's minimum, target, comfort, and maximum widths;
- the active work mode's minimum and comfort widths;
- the prior effective layout for hysteresis; and
- an optional pane explicitly requested by the user.

No layout calculation may query a database, start a worker, rebuild destination content, or write
preferences.

The initial shared baselines are:

| Role | Protected minimum / escape floor | Fixed target | Comfort / maximum |
| --- | ---: | ---: | ---: |
| Library | 24 / 0 when collapsed | bounded 3:13 projection, 31 fallback | 34 default / 48 custom |
| Destination list | 32 / 0 when collapsed | 40 | 56 / 72 |
| Read-only work mode | 44 / 0 compositor escape | flexible | receives remaining width |
| Editor work mode | 48 / 0 compositor escape | flexible | 56 / remaining width |
| Each grip | 5 fixed | 5 fixed | 5 fixed |

Protected work-mode minimums drive responsive collapse decisions. If the entire shell is narrower
than both grips plus that protected minimum, the work pane may compress toward its zero-column
compositor escape floor so the shell remains reachable instead of overflowing. Destination
contracts may raise a mode minimum when existing controls require it, but may not lower the
48-column editor minimum without focused geometry evidence.

### Destination ownership

The shell does not know field schemas, record types, actions, validation rules, draft models, trust
states, or service calls. Each destination owns:

- its list query, paging, sorting, existing bulk-selection capability, and stable row identity;
- selected, pending, and loaded item identity;
- work modes and mode-specific state;
- draft, save, discard, conflict, recovery, and destructive-action behavior;
- empty, loading, stale, unavailable, and error copy; and
- the mapping from existing capabilities to visible controls.

This boundary permits internal destination changes without changing the shell and permits shell
geometry changes without touching domain services.

## Information architecture and pane behavior

The work pane is permanent. Library and the destination list are independent collapse targets.
Each collapsed pane leaves a five-column, full-height grip with an action-labelled direction. The
work-pane header also keeps compact, labelled restore controls reachable when one or both optional
panes are collapsed.

Default wide layout uses a bounded fractional Library width and fixed destination-list targets.
The Library default follows the ordinary 3:13 Library-to-canvas proportion, rounded
deterministically as `floor((3W + 8) / 16)` from the positive adaptive-shell content width `W`
and clamped to 24–34 cells; exact halves round upward. A zero-width pre-layout shell retains an
all-zero effective sentinel. Its representative new/reset preference value is 31.
Custom widths remain opt-in, normalized, and clamped to their declared 24–48 range. The geometry
resolver distinguishes:

1. **Requested layout** — persisted manual visibility and normalized widths.
2. **Responsive override** — temporary collapses or compression required by current width.
3. **Effective layout** — the rendered result.

Responsive decisions never overwrite requested state. Stable thresholds and a hysteresis band
prevent collapse/reopen oscillation near boundaries. Requested panes return automatically when
space becomes available.

Resolution follows these rules:

1. Reserve both grips and protect the active work mode's usable width.
2. Render the Library pane at its bounded fractional default (or explicit custom width) and the
   list pane at its fixed target when they fit.
3. On shortfall, auto-collapse Library before the destination list.
4. Whenever Library is effectively collapsed and the list remains open, allocate reclaimed width
   to the list up to its 56-column comfort cap; allocate the rest to the work pane.
5. Auto-collapse the list only when the active work mode still cannot remain usable.
6. An explicit open gives the requested pane temporary priority. If all roles cannot fit, the
   other optional pane becomes effectively collapsed while requested preferences remain intact.
   Destination-owned automatic priority follows the same branch: the default Notes Navigator
   keeps Items open and permits Work compression at narrow widths, while Notes editor/work-owned
   states release that priority so the normal collapse order protects editing space.
   With the two five-cell grips reserved, explicit Library priority holds its 24-cell floor at
   `W=34` and compresses to `max(W - 10, 0)` below it (`W=33` yields Library 23, Items 0, Work 0).
7. Manual grip toggles persist requested visibility. Custom widths persist only through an explicit
   Settings save; the grips remain collapse/expand controls and are not drag handles.
8. Automatic collapse, adaptive list expansion, window resize, and effective priority do not
   persist.

Adaptive expansion begins from the normalized requested list width. A saved custom width above the
56-column comfort cap is retained whenever it fits; the comfort cap limits automatic growth from a
smaller requested width and never shrinks a larger deliberate choice.

Library visibility is one shared preference across these destinations. Destination-list
visibility and preferred width are remembered separately for Media, Conversations, Notes, Prompts,
and Skills. Preference ownership is explicit:

| Scope | Config section | Keys |
| --- | --- | --- |
| Shared Library geometry | `[library.reader]` | `library_open`, `custom_widths_enabled`, `library_width` |
| Media list | `[library.media_reader]` | existing `items_open`, `items_width` |
| Conversations list | `[library.conversations_reader]` | `items_open`, `items_width` |
| Notes list | `[library.notes_reader]` | `items_open`, `items_width` |
| Prompts list | `[library.prompts_reader]` | `items_open`, `items_width` |
| Skills list | `[library.skills_reader]` | `items_open`, `items_width` |

`custom_widths_enabled` is one shared opt-in: when false, the bounded fractional Library default and
normalized fixed destination-list targets apply while saved custom values remain available for
later re-enabling. Explicit Library widths from 35 through 48 are intentional overrides and are
not clamped to the default 34-cell ceiling. When `[library.reader]` is absent, shared
normalization reads `library_open`, `custom_widths_enabled`, and `library_width` from the existing
`[library.media_reader]` section. No eager disk rewrite occurs. The first explicit shared toggle or
Settings save writes `[library.reader]`; Media then reads the shared section first while retaining
the legacy fallback. Settings labels shared Library geometry separately from per-destination list
geometry.

## Destination work surfaces

| Destination | Default mode | Additional modes | Work-pane contract |
| --- | --- | --- | --- |
| Conversations | Read | Info | Complete saved transcript, full-conversation Find, metadata, existing export, and Open in Console. Read-only. |
| Notes | Edit | Preview, Info | One draft shared by editor and preview. Info contains properties, persisted metadata, Chatbook utilities, sync state, and separated destructive actions. |
| Prompts | Basic | Advanced, Info | One lossless draft across Basic and Advanced. Info contains history, collection, provenance, and lifecycle state. |
| Skills | Overview | Edit, Trust, Files | Read-first overview, explicit editing, revision-specific trust review, and supporting files. Files remain read-only unless already editable. |

### Conversations

Read shows the complete saved transcript while mounting long histories progressively so opening a
large conversation does not block the event loop or create an unbounded first frame. Find searches
the complete saved conversation, reports match count, and brings a matching message into the
mounted window. It does not silently limit itself to currently rendered messages. Info shows the
existing conversation metadata and actions. Open in Console uses the existing handoff contract and
does not modify the saved record.

### Notes

Mounting the default Edit mode does not make a note dirty. The first user-authored change does.
Edit and Preview share one item-owned draft; Preview renders the current draft rather than only the
last persisted body. Info distinguishes saved metadata from unsaved draft changes. Templates,
create, import, sync, conflict resolution, and recovery occupy the work pane while the note list
remains mounted. Existing confirmations and file/system pickers may appear transiently above it.

### Prompts

Basic and Advanced are two projections of one lossless item-owned draft. Basic may hide advanced
fields but saving from Basic cannot reset or delete them. Validation identifies the owning mode and
can move focus to the invalid field. Info presents persisted history, collection, import
provenance, and lifecycle state without pretending an unsaved draft is already historical.
Create/import workflows remain in the work pane.

### Skills

Overview is the default read-first surface. Edit is an explicit state. Trust identifies the exact
reviewed revision or fingerprint and follows the existing trust policy when trust-relevant content
changes; a prior review that no longer applies is visibly stale. Files identifies its source and
remains read-only unless the current capability expressly supports modification. Import, create,
trust review, and recovery stay in the work pane.

### Shared action placement

Filter, sort, paging, existing bulk selection, and list-level actions remain in the destination
list.
Existing destination primary actions remain visible in the work pane when applicable. Secondary
item actions may use a compact More surface, but destructive actions are separated, labelled, and
confirmed rather than buried as ordinary menu items. Mode controls stay keyboard reachable and
collapse to a compact selector instead of wrapping at narrow widths.

## Selection, loading, and data flow

List selection and loaded work identity are separate. Selecting a row updates the list immediately
and starts a detail load fenced by destination, stable item id, revision or mutation epoch, and a
monotonically increasing generation.

Until the matching load succeeds, the work pane either shows an initial loading state or retains
the previous item beneath an explicit banner such as:

> Loading “B”… showing “A” until ready.

Save, delete, trust, and other identity-sensitive actions are disabled while selected and loaded
identities differ. Late success or failure may update the work pane only when the complete fence
still matches. A stale result may enter a cache only when the cache is keyed by immutable item id
and matching revision; otherwise it is discarded. Destination changes, deletion, mutation,
revision changes, and screen unmount invalidate pending generations.

Switching destinations preserves each destination's list query, page, selection, work mode, safe
scroll state, requested list geometry, and recoverable session state. Drafts are owned by
destination plus item id. Navigating away uses the destination's existing save, discard, conflict,
or recovery contract rather than retaining an ambiguous anonymous draft. This does not introduce a
multi-item draft registry: navigation completes only after the current destination's existing
dirty-draft contract resolves.

Create, import, sync, template selection, trust review, and recovery replace only the work-pane
content. The list remains mounted unless a narrowly scoped mutation genuinely requires a short,
labelled lock. Where an existing capability inventory includes bulk mode, it keeps the last singly
loaded item as a labelled read-only preview, states whether that item is included in the bulk set,
and disables item-specific mutations. Bulk controls stay in the list. No destination gains bulk
selection merely by adopting the shell.

Keyboard behavior follows the application contract:

- `/` focuses the active destination filter only when focus is outside an editor or text input.
- Screens do not bind `F6`; their visible region identities participate in the existing global
  focus cycle.
- Escape closes the nearest transient state first, then steps backward within the active work
  mode, then moves toward the list/Library region before leaving the destination.
- Focus never moves into a collapsed or replaced region.

## Loading, errors, and recovery

Each pane owns its operational state so a failure in one region does not blank the destination.

- An initial list failure shows a scoped retry without disturbing the work pane.
- A background list refresh failure retains the last successful rows with a stale marker and
  preserves filter, paging, selection, and loaded work.
- Empty collection, filtered-empty, no selection, initial loading, background refresh, stale data,
  and failure are distinct states. Filtered-empty offers Clear filters.
- A detail failure preserves the previous item with the identity-mismatch treatment, names the
  failed selection, and offers a fenced retry.
- External deletion or revision conflict never silently selects another item. Recoverable draft
  content can be copied, exported, or handled through the destination's existing recovery path
  before dismissal.
- Drafts survive recoverable load, save, sync, validation, and conflict failures.
- Mutations lock only the affected item and action, prevent accidental duplicate submission, and
  leave a receipt identifying target and outcome. Retry is offered only when repetition is safe.
- Optional capabilities explain the missing dependency or configuration without exposing secrets
  or unsafe paths.
- Preference failures are deduplicated, preserve a usable session layout, and do not interrupt
  content work.

Threaded worker cancellation is best-effort. Generation invalidation and refusal of late results
are mandatory even when the underlying work cannot be stopped.

## Focus and accessibility

- Selection, loaded identity, pending, dirty, stale, conflict, trust, unavailable, and error states
  use visible text rather than color alone.
- Grips and restore controls expose the pane name, current action, current width where applicable,
  and keyboard instructions without changing geometry when focused.
- Mode controls expose selected state and remain reachable without traversing the document body.
- Non-navigational actions retain focus on their initiating control when it still exists.
- Async completion may move focus only when the initiating workflow is still active and the user
  has not moved focus elsewhere.
- Successful creation focuses the new item's primary field only while that creation workflow still
  owns focus intent.
- Actionable status remains visible until resolved and uses the application's established
  notification conventions without claiming unsupported terminal screen-reader behavior.

## Delivery sequence

This is one programme specification delivered through four PRs:

1. **Conversations** — extract the Library-local shell from Media, preserve Media domain behavior,
   add adaptive list expansion, and migrate the complete Conversations reader.
2. **Notes** — migrate the note list and Edit/Preview/Info work pane while preserving templates,
   sync, conflict recovery, import, and utilities.
3. **Prompts** — migrate the list and lossless Basic/Advanced/Info work pane while preserving
   import, history, collections, provenance, validation, and lifecycle behavior.
4. **Skills** — migrate Overview/Edit/Trust/Files after the shared structure has stabilized while
   preserving import, trust review, supporting files, and existing capability boundaries.

The shared extraction stays inside the Conversations PR rather than creating a fifth
infrastructure-only PR. Each later PR must be independently releasable and may depend only on
already merged earlier work.

The Conversations PR has three reviewable internal checkpoints: first extract the shell with
unchanged Media behavior and green Media regressions; then add shared preferences plus the approved
adaptive list expansion; only then wire Conversations as the second consumer. A later checkpoint
may not mask a regression in an earlier one.

Global extraction with Watchlists remains out of scope. After both the Library programme and the
shipped Watchlists reader have concrete usage evidence, a separate ADR may evaluate whether any
application-wide primitive is justified.

## Testing and verification

All implementation follows test-driven development. Each new behavior is first observed failing
against the pre-change implementation, and every guard is mutation-checked where practical.

### Shared geometry and shell

- Unit and property tests cover requested versus effective state, fixed/custom normalization,
  adaptive list expansion, explicit-open priority, mode minimums, hysteresis, idempotent resize,
  and restoration when width returns.
- Geometry assertions prove each mounted region and control stays within the screen rather than
  checking only `display` and text.
- Textual Pilot tests cover composition identity, collapse/restore controls, keyboard and pointer
  activation of the grips, Settings-driven custom-width refresh, focus evacuation, compact mode
  controls, and global focus-region participation.
- Spies prove resize and geometry resolution perform no database reads, start no destination
  workers, and write no preferences.
- Preference tests pin the exact config ownership, legacy Media fallback, absence of eager rewrites,
  and first explicit shared save. They also prove only completed user actions persist and responsive
  adaptation does not.
- Media regression coverage runs in the Conversations PR and pins all existing Media modes,
  actions, selection/loading identity, bulk behavior, recovery, and preference compatibility.

### Destination behavior

Every migration covers:

- its signed-off before/after capability inventory;
- list query, paging, stable selection, and bulk mode where present in its capability inventory;
- selected versus loaded identity and every stale-response fence dimension;
- initial, refreshing, stale, empty, filtered-empty, loading, error, conflict, unavailable, and
  deletion states;
- dirty-draft ownership and destination-specific recovery;
- mode changes, compact mode controls, and hidden-field safety;
- focus preservation without async focus theft; and
- destination change and unmount while workers are still running.

Conversations additionally measures first-frame and traversal behavior with a long saved
transcript and proves complete-conversation Find. Notes proves Preview uses the current draft and
mounting Edit is clean. Prompts proves Basic save preserves every Advanced-only field. Skills proves
trust review is revision-specific and becomes stale under the existing policy when applicable
content changes.

### Representative live verification

Each PR includes a focused live TUI walkthrough at approximately:

| Terminal | Required checks |
| --- | --- |
| 160x50 | All three roles, bounded Library default, list comfort expansion after Library collapse |
| 120x35 | Library-responsive collapse, full list details, usable work mode |
| 100x30 | deterministic one/two-pane effective state and compact mode control |
| 80x24 | permanent work pane, both restore controls, no compositor overflow |

The verification also checks CSS-bundle loading, runtime widget inventories, truthful footer hints,
and the actual application path rather than only isolated widget harnesses.

## Acceptance criteria for implementation planning

- One Library-local adaptive shell can host Media and each of the four destination migrations
  without owning destination behavior.
- Library and destination list are independently collapsible; the work pane remains mounted.
- Collapsing Library expands the destination list toward its comfort cap without overwriting its
  saved width.
- Shared Library preference and per-destination list preferences behave as specified.
- Conversations, Notes, Prompts, and Skills expose the approved modes and preserve their complete
  capability inventories.
- Selection/loading identity, stale workers, dirty drafts, conflicts, deletion, and recovery follow
  the approved contracts.
- Responsive behavior is stable, overflow-free at supported verification sizes, and performs no
  data work or responsive preference writes.
- All four PRs include focused automated and live verification, with Media regression evidence in
  the first PR.
