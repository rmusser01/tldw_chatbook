# Character Conversation Navigation and Local Meaning Search Design

**Date:** 2026-09-03
**Status:** Revised after Impeccable critique; user review pending
**Workstream:** Console Context, Console `Ctrl+K`, and Roleplay character conversations
**Delivery:** Eight independently reviewable pull requests

## Goal

Make character conversations easy to find and resume across the product without
weakening the Console's role as the live conversation owner.

The design serves two distinct jobs:

1. A first-time user can understand where character chats live, see recent chats
   near the current character, recover an unavailable character safely, and
   resume the exact conversation without learning the application's storage
   model.
2. An experienced user can treat `Ctrl+K` as an operational agent switchboard,
   move among active work and complete history, search all character-backed
   conversations, and use optional local Meaning search without losing keyboard
   speed or target trust.

The entire first release searches **local conversations only**. Keyword and
Meaning use the same data-profile-owned local corpus. Server and cached-server
conversation discovery require a later, separately approved ADR and delivery
programme; no first-release surface suggests that remote or cached results are
included.

## Product decisions

The following decisions are fixed for this design:

- `Ctrl+K` remains an operational agent switchboard. It opens in `Active` and
  blank-query Enter targets the most-recently-used other open tab.
- `Ctrl+K` gains a third `Character chats` mode beside `Active` and `History`.
- The Console Context Character section shows at most four group headers. The
  current character is force-included when present; otherwise the most-recent
  character is the default expanded group.
- Character groups contain character cards only, never Personas.
- Each ordinary character group shows at most five conversations and ends with
  `View all N in Roleplay`.
- Context character search is global, data-profile-safe Keyword search rather than
  a filter of the four visible groups.
- Enter on a conversation resumes that exact conversation in Console.
- Roleplay is the complete, searchable, paginated per-character destination.
- Keyword search is delivered completely before optional Meaning search.
- Meaning search is explicitly enabled, local, and limited to local
  conversations in its first release.
- Eligible semantic text is the selected visible user/assistant branch only.
  System prompts, hidden thinking, tool calls/results, attachments, and
  non-selected branches are excluded.
- `Index existing chats` and `Keep future chats indexed` are separate Settings
  decisions. Indexing also has pause, resume, cancel, rebuild, and delete
  controls.
- The feature ships through the eight-PR sequence defined in this specification.

## Scope

### In scope

- One shared, read-only character-conversation projection for Context,
  `Ctrl+K`, and Roleplay.
- Profile-safe local Keyword discovery.
- Recent character groups in Console Context.
- A global `Character chats` mode in the `Ctrl+K` switcher.
- Complete per-character conversation browsing, search, preview, and exact
  Console resume in Roleplay.
- An `Unavailable character` recovery group.
- A Library-owned, explicitly confirmed repair path for provably local
  unresolved character links.
- Explicitly enabled local Meaning search for local conversations.
- Index consent, progress, failure, repair, and deletion controls in the
  canonical Settings screen.
- Responsive, keyboard-complete, production-stylesheet verification.

### Out of scope

- Personas in any Character chat result set.
- Network retrieval during Context, `Ctrl+K`, or Roleplay search.
- Cached or live server conversation discovery, indexing, totals, or resume.
- Remote embedding APIs or an automatic remote fallback.
- Transcript editing, deletion, branch mutation, or identity repair inside the
  switcher or Roleplay preview.
- Reconstructing a missing character from transcript content.
- Replacing Library's global conversation archive and administrative actions.
- A universal work inbox or standalone workflow-run rows in `Ctrl+K`.
- A blocking onboarding wizard, product telemetry, or a new search dependency
  for Keyword search.

## Relationship to existing decisions

This design extends rather than replaces the already-shipped trust and resume
work:

- TASK-28125's exact-query submission, strict F2 targeting, explicit keyboard
  candidate, visible scrolling, textual state labels, and MRU-other behavior are
  hard invariants.
- ADR-046 continues to own historical display identity and exact ID-only
  Roleplay-to-Console resume. PR 1 amends it with the cancellable activation
  result and pre-navigation draft-veto contracts defined here. A current card
  never substitutes for saved historical identity.
- ADR-031 continues to own reserved keys and truthful hints. F3 remains the
  switcher-local mode key and now cycles three modes.
- [Resume prior character chats from Roleplay](2026-08-26-roleplay-resume-prior-character-chat-design.md)
  and [Discover older local character conversations in Roleplay](2026-08-27-task-22453-older-roleplay-conversations-design.md)
  remain the implementation predecessors: their exact ID-only resume,
  active-session reuse, read-only preview, and stable local keyset pagination
  are extended rather than rebuilt.
- Library remains the global archive and management destination. Roleplay gains
  complete browsing only within one selected character identity.

PR 1 creates a new path-specific ADR. Its collision-safe numerical prefix is
assigned at filing time after the repository-wide task/ADR sweep; its canonical
filename suffix is
`character-conversation-navigation-and-local-semantic-search.md`. That PR also
amends these existing ADRs by full filename:

- `004-personas-destination-native-workbench.md`: Roleplay owns complete
  per-character browsing; Library retains global browsing and archive actions.
- `030-derived-index-lifecycle-and-atomic-media-migrations.md`: its media index
  remains media-specific, while the new ADR applies the same authoritative-
  source and eventual-cleanup principles to a separately owned conversation
  index.
- `037-roleplay-assistant-identity-and-persona-user-profile-separation.md`:
  projection and search keys use typed local character identity; unresolved
  legacy rows never inherit the current card.
- `046-roleplay-chat-display-identity-and-template-provenance.md`: exact resume
  becomes a cancellable, result-typed activation whose modal stays mounted
  until Console confirms the destination.
- `083-console-edge-rails-and-workspace-tree-ownership.md`: the Context
  Character section becomes an always-composed conversation-navigation surface
  directly after Conversations and no longer depends on avatar visibility.
- `085-console-activity-receipts-and-switcher-ownership.md`: `Ctrl+K` gains the
  third Character chats mode plus the activation state machine without moving
  receipt or final destination ownership.

ADR-031 is linked but not amended: F3 remains the only switcher-local mode key,
and late Meaning results use a visible control rather than a new function-key
binding. ADR-033 is linked as the unchanged Settings commit-model authority;
the staged fields and immediate reviewed actions below follow its labels.

## Information architecture and ownership

### Surface responsibilities

| Surface | Primary job | May mutate conversations? |
| --- | --- | --- |
| Console | Live chat, agent activity, exact resume, and open-tab ownership | Yes, through existing canonical paths |
| Context Character section | Ambient recent continuation and global Keyword entry | No |
| `Ctrl+K` | Operational switching across Active, History, and Character chats | No; it activates Console targets |
| Roleplay | Character-card management plus complete per-character browse/preview | Character card operations only; conversations remain read-only |
| Library | Global archive, cross-domain browsing, and existing administrative actions | Through existing Library contracts |

Context, `Ctrl+K`, and Roleplay never create their own resume or conversation-
mutation path. They capture an immutable target and ask the existing Console
opener to recheck and activate it. Roleplay's existing card editor remains the
only card-mutation owner.

### Workflow walkthroughs

A first-time continuation path is:

1. Context places Character immediately after Conversations and initially opens
   it when character context exists and no preference has been saved.
2. The current character is the sole expanded group; the user reads recent chat
   titles and relative dates without encountering storage terminology.
3. Selecting a chat reveals its exact state; Enter says `Opening…`, preserves
   the visible row, and dismisses only after Console shows that exact chat.
4. An unresolved chat says why it cannot resume and routes to Library for safe
   inspection or repair.

An experienced switchboard path is:

1. `Ctrl+K`, Enter switches to the MRU other tab exactly as before.
2. F3 moves Active → History → Character chats while preserving the appropriate
   per-visit query and stable selection.
3. Keyword responds from the complete local FTS corpus; when locally enabled,
   Meaning performs direct vector retrieval and never silently rearranges a
   painted list.
4. Enter opens only the committed highlight. `View all N in Roleplay` moves into
   complete, paginated per-character history, and `Back to Console` restores the
   originating focus anchor.

### Shared projection

A single application service, referred to here as the **Character Conversation
Projection**, provides immutable read models. It is read-only and has no UI
dependencies.

Its serialized identity is a closed tagged union rather than a tuple with
nullable or polymorphic fields:

```text
ResolvedLocalCharacterKey(
    data_authority_id,
    character_id,
)

UnresolvedConversationKey(
    data_authority_id,
    conversation_id,
)
```

An activatable conversation target is:

```text
LocalCharacterConversationTarget(
    character=ResolvedLocalCharacterKey(...),
    conversation_id,
)
```

`data_authority_id` is the selected conversation database's existing durable
`local_authority_id` from ADR-037—not the Console runtime's absolute database
path and not a RAG configuration-profile ID. IDs remain opaque outside that
authority. Titles, card names, current-card selection, and routing labels never
establish identity. `UnresolvedConversationKey` can be listed and sent to
Library, but it is never activatable or silently promoted into a resolved key.
The union is versioned at its serialization boundary and rejects unknown tags.

Unresolved result state, separate from entity identity, carries one of
`MISSING_CARD`, `DELETED_CARD`, `MISSING_CHARACTER_AUTHORITY_LINK`, or
`AMBIGUOUS_LEGACY_LINK`. A diagnosis may change without changing focus, row
identity, or selection. Serialized row keys include the union tag and every key
component, so a resolved and unresolved row cannot collide even when their raw
conversation IDs match.

The adapter reuses the database's canonical bounds rather than inventing a UI
normalizer: character IDs are integers from 1 through `2^63 - 1`; data-authority
and conversation IDs are nonblank canonical text of at most 256 UTF-8 bytes.
No identifier is truncated, case-folded, derived from a filesystem path, or
parsed from display text.

The projection supplies three narrow query contracts:

1. recent character groups with bounded conversation previews and exact totals;
2. data-authority-filtered global Keyword results; and
3. complete, keyset-paginated conversations for one exact character group.

Each result contains only presentation-safe metadata and an immutable activation
target: stable identities, saved display snapshot, title, timestamps, open/current
state, local-source label, and optionally a selected-detail excerpt. It does not
carry credentials, prompts, hidden thinking, tool payloads, attachments, or an
assembled transcript.

### Local source and Data Profile scope

**Data Profile** means the app-selected local conversation database and its
`data_authority_id`. **RAG configuration profile** means the independently
selectable retrieval-settings profile in Settings. These names are never
shortened to the same unqualified “profile” in architecture, persistence, or
diagnostic contracts.

The active Data Profile is the outermost conversation authority boundary.

- Rows come only from that Data Profile's authoritative local conversation
  database.
- Its `data_authority_id` plus a monotonic data revision are captured with every
  query and rechecked before commit and activation.
- A Data Profile or revision change invalidates outstanding generations and
  selected targets.
- Server identities, authentication state, and caches do not enter the first-
  release projection.

Every search surface names the real corpus: `This profile · Local chats`.

“Complete” means complete within the active Data Profile's authoritative local
database. Roleplay and `View all N` never imply a remote total.

### Historical identity and unavailable characters

The projection resolves a conversation only against the exact saved
data-authority/character identity. It may use the historical saved character
name for display, but never uses a same-named current card as identity.

A legacy character-authority-null row may be persistently backfilled only when the
same local character database proves a unique matching character and supplies
its durable data authority. That deterministic migration belongs to PR 2 and
is transactionally tested. Merely opening or searching does not mutate the row.

Legacy rows without unique local proof, missing cards, deleted cards, and other
unresolvable identities appear under `Chats with unavailable characters`. They
never inherit the current card or a name match.

The recovery group offers:

- `Open in Library`, preserving the exact conversation identity;
- `Repair in Library` only when compatible cards in the same exact local
  authority can be enumerated; and
- a plain reason such as `Card deleted`, `Character source changed`, or
  `Historical identity incomplete`.

Library owns the confirmation and mutation flow. It shows the old and proposed
identity, writes through the canonical provenance service, and requires an
explicit user action. Context, `Ctrl+K`, and Roleplay only navigate there; they
never repair identity. The service never guesses, and the action is absent when
no data-authority-safe candidate exists.

The navigation payload is a versioned `LibraryCharacterRepairContext` containing
the stable `UnresolvedConversationKey`, expected conversation version, saved
historical identity snapshot, and return anchor. The projection service may
enumerate only live Character cards from the same `data_authority_id`; it never
preselects by name. Library shows old versus user-selected identity and commits
through a compare-and-set provenance mutation. A version mismatch, deleted
conversation, or changed candidate leaves the record untouched and focuses
Refresh. Successful repair invalidates the FTS/Meaning projection, returns to
the repaired Library row, and exposes the original return path.

## Console Context Character section

### Browse state

The existing Character section becomes a compact tree and moves directly after
Conversations, before Model. It is always composed; its presence and
conversation controls do not depend on the incumbent `show character avatar`
preference. That preference controls only the optional image. The tree never
waits for image rendering.

The collapsed header summarizes the current character when present and the
number of local character chats, for example `Character · Samira · 12 chats`.
The expanded body starts with a compact identity line naming the current
character, local source, and open state. The avatar is supporting content and
is suppressed before identity, groups, conversations, search, empty/recovery
copy, or primary actions.

On the first Console visit with no saved Character disclosure preference, the
section defaults open when the current conversation has a resolved character or
the Data Profile has any character chats. Once the user explicitly opens or closes
it, that choice wins. With no cards and no character chats, the collapsed
header says `Character · No chats`; opening it says `No character chats yet`
and offers `Open Roleplay`.

The preference contract adds a versioned `character_disclosure_explicit`
marker; the existing `character_open: bool` cannot distinguish default false
from a user close. A new Data Profile starts with `explicit=false`. Manual open
or close writes the Boolean plus `explicit=true`. A legacy preference record
without the marker preserves its stored Boolean and is treated as
`legacy-preserve` until the next manual toggle, avoiding a surprise auto-open
for existing users. Responsive forced collapse never writes either field.

- At most four group headers are visible.
- The current character, when resolvable, is first and force-included even if it
  is not among the four most recent or has no prior chat. A zero-chat current
  group says `No chats with Samira yet` and offers `Start in Console`; it does
  not render a meaningless `View all 0` action.
- Remaining groups are ordered by their newest conversation activity.
- `Chats with unavailable characters`, when nonempty, consumes one of the four
  slots; it is not an extra fifth header.
- Conversation ordering within a group is descending effective date:
  `last_modified`, then `created_at`, then stable conversation ID.
- The tree is an accordion with at most one expanded group. The current group
  is initially expanded; without one, the most-recent resolvable group is
  expanded. If only the unavailable group exists, it is expanded. Opening a
  different group collapses the prior group and retains the new choice for the
  Console session.
- An expanded ordinary group shows at most five conversations and ends with
  `View all N in Roleplay`.
- The unavailable group shows at most five rows and ends with
  `View all N in Library`.

The section has one outer scroll owner. Character groups do not introduce
nested scroll containers.

At widths where the incumbent responsive policy collapses Context, the section
does not leave hidden focus targets or advertise unreachable controls. Starting
with PR 5, the globally available `Ctrl+K` switcher is the complete narrow-
terminal fallback: its direct `Character chats` mode button and F3 cycle expose
the same local Keyword corpus. PR 4 remains honest in isolation by relying only
on the already-shipped Roleplay destination and making no narrow-screen claim
about a switcher mode that has not landed yet.

### Interaction

- Click, Enter, or Space on a header toggles it.
- Left collapses an expanded group; Right expands a collapsed group.
- Click selects a conversation; double-click or Enter resumes it in Console.
- `View all N in Roleplay` sends a resolved-local-character deep link.
- A selected unavailable row exposes its recovery detail and actions without
  pretending it can resume normally.

Focus remains on the same stable group/row across refresh when it still exists.
If it disappears, focus falls to its group header, then the next visible group,
then the Character section header.

### Search state

`Search character chats…` performs global, data-profile-safe local Keyword
search. It does not filter only the four browse groups.

A nonblank query replaces the browse tree with at most eight flat conversation
results. Each result shows title, character, local source, and age. Context
never shows transcript snippets. PR 4 ships without a cross-surface query-
continuation control. PR 5 capability-gates and adds
`Continue search in Character chats`, which transfers the validated query and
opens `Ctrl+K` directly in Character chats mode. A partial deployment therefore
never shows a dead destination.

Clearing the query or pressing Escape restores the exact pre-search disclosure,
selection, scroll, and focus snapshot when its identities remain valid.

Explicit states are:

- no character chats yet;
- no Keyword matches;
- searching;
- character source changed;
- index unavailable or rebuilding; and
- unavailable-character recovery.

## `Ctrl+K` operational switchboard

### Modes and first-use guidance

The modal title becomes `Switch or resume`. It has three direct mode buttons:

```text
Active | History | Character chats
```

Every ordinary `Ctrl+K` open starts in Active. F3 advances
`Active → History → Character chats → Active`; the visible hint names the next
mode. A validated Context handoff may open Character chats directly with a
validated query. Active and History share one operational query because Active
may explicitly reveal matching History rows; Character chats owns a separate
query. F3 restores the destination mode's per-visit query, selection, and
scroll. All of this state is discarded on close.

For a nonblank Active query with zero Active matches, the list may show exact
History matches under the unmistakable mode line
`Active · showing History matches`. Enter still activates only the visibly
highlighted immutable row; F3 commits the same query into History. Character
chats never auto-widens into another corpus.

Inline copy teaches the mode currently shown. There is no blocking tutorial:

- Active: work requiring attention, running work, new results, and open tabs;
- History: all persisted conversations in bounded local history; and
- Character chats: conversations attached to Character cards, not Personas.

At 52×20 the modal is a complete, non-scrolling shell with a four-result
viewport. Its 20 rows are reserved as follows: one title-border row, one top
padding row, one mode row, one search row, one scope/search-strategy row, one divider,
eight result rows (four two-line results), two selected-detail rows, one
action/paging row, one combined hint-and-Cancel row, one bottom padding row, and
one bottom-border row. The 50 inner columns retain at least 48 content columns
after horizontal padding. Inline teaching collapses into the scope/status line
and F1; mode buttons, focus, the primary action, paging, and Cancel never
disappear. Taller layouts increase only the result viewport up to the existing
35-row modal ceiling.

In Character chats, the scope/search-strategy row contains the focusable
`Keyword | Meaning · Local` choice plus the local-corpus state; Active and
History use the same reserved row for their scope/count. Compact Active results
inline their operational group as the textual status token (`[WAITING]`,
`[WORKING]`, `[NEW]`, `[CURRENT]`, or `[OPEN]`) and deliberately omit separate
one-row group headings. This narrow-only presentation supersedes headings, not
the incumbent consequence ordering. Wider layouts retain headings. The fixed
action row fits Previous, `Apply Meaning results` when offered, and Next without
changing height.

Meaning remains focusable when off or unavailable, labeled with its exact state
(`Meaning off`, `Model missing`, or `Index needs repair`). Enter opens
Settings > RAG > Character chat search with a return anchor; it never enables,
downloads, or starts indexing from the switcher.

DOM and Tab order is mode buttons → search → Keyword/Meaning → results →
apply/paging → Cancel, while opening the modal moves initial focus directly to
search. Shift+Tab reaches the mode buttons. Noninteractive selected detail is
announced when selection changes and never enters Tab order.

### Enter and mode-specific commands

TASK-28125's target rules remain exact:

- On blank-query Active with no explicit row navigation, Enter activates the
  MRU other open native tab.
- After explicit navigation or with a nonblank query, Enter activates only the
  highlighted result for the committed query generation.
- History and Character chats Enter always activate the highlighted immutable
  result.
- Repeated Enter is ignored while that target is opening.
- A single pointer click on a `Ctrl+K` result activates it, matching the
  incumbent command-switcher grammar. Pointer-down captures the stable result
  key; if reconciliation moves or removes that key before click, activation is
  cancelled rather than retargeted.
- F2 Rename exists only for eligible open native tabs. It is absent in
  Character chats rather than disabled ambiguously.
- Hints and actions use `OPEN TAB`, `RESUME CHAT`, and `VIEW DETAILS`
  consistently; they do not call a saved conversation an active session.

### Character chat result grammar

Character chats searches all eligible local character-backed conversations in
the active Data Profile. Personas are excluded.

Each result uses two stable lines:

```text
[status] Conversation title
Character name · Local · Relative age
```

Only the highlighted result may show a detail line containing a safe message
excerpt. Its fixed two-row detail region also gives an absolute timestamp and
state, for example `Updated 2026-09-03 14:22 PDT · Saved`; rows retain relative
age for scanning. Unselected rows never expose transcript text, and their
absolute timestamp never consumes list space. Unicode truncation is terminal-
cell-aware; the selected detail exposes the full title through F1 when 48
columns cannot contain it.

States include current tab, other open tab, saved/closed, opening, unavailable,
deleted, and character source changed. Color supplements but never replaces
these labels.

### Search and asynchronous ordering

Every query receives a new generation containing data-authority/revision,
mode, query, and modal visit. Stale work cannot commit.

When Character chats mode is active and Meaning search is ready, the user can
explicitly choose `Meaning · Local`.
Keyword and Meaning both search local conversations, but only Meaning computes
a local query embedding and performs direct vector retrieval. Meaning never
uses a lexical candidate prefilter. It queries the ready generation's ANN index
for at most 200 eligible chunks, aggregates each conversation by its lowest
cosine distance, retains at most 50 conversations, revalidates them in SQLite,
and orders equal distances by descending effective date and then stable conversation
ID. It may perform one bounded refill for at most 400 examined chunks total
when revalidation removes candidates. Blank queries, modal opening, and F3 mode
changes do not launch vector work.

Meaning starts Keyword fallback and vector retrieval together. A 120 ms
coalescing gate delays only a ready Keyword list, preventing a transient lexical
list from becoming reading state:

- valid vector results may paint immediately as the first list;
- a nonempty Keyword list ready before 120 ms waits until the gate expires for
  Meaning;
- after 120 ms, whichever valid vector or nonempty Keyword list is ready—or
  becomes ready first—paints;
  and
- once any result list has painted, late Meaning results never reorder it
  automatically, even if selection has not moved.

A zero-match Keyword fallback does not end a Meaning query while vector work is
still healthy; the status remains `Finding meaning…`. If vector retrieval then
fails, the surface paints the Keyword fallback (including an honest empty
state) with `Meaning unavailable · Keyword results`. A zero-match vector result
is the authoritative `No Meaning matches` state.

A late result exposes a visible, focusable `Apply Meaning results` control. It
preserves the selected stable conversation ID when still present and restores
focus plus the nearest valid scroll position. No F4 binding is introduced.
The control appears with the textual status `Meaning results ready` but does not
steal focus or change the list's height. Navigation or activation can never be
retargeted by late work.

The presentation state is total:

| Selected search | Work/result state | Visible status | Painted list/action |
| --- | --- | --- | --- |
| Keyword | Running | `Searching local chats…` | Prior query is cleared; no stale rows |
| Keyword | Ready/empty | `Keyword · N results` / `No Keyword matches` | Keyword rows / empty recovery |
| Meaning | Neither leg ready | `Finding meaning…` | No stale rows |
| Meaning | Keyword wins after gate | `Meaning searching · showing Keyword` | Keyword rows; no reordering |
| Meaning | Vector is first paint | `Meaning · Local · N results` | Meaning rows; late Keyword ignored |
| Meaning | Vector ready after Keyword paint | `Meaning results ready` | Keyword rows plus `Apply Meaning results` |
| Meaning | Vector ready with zero matches | `No Meaning matches` | Empty state; late Keyword ignored |
| Meaning | Query/model/index failure | `Meaning unavailable · Keyword results` | Keyword fallback when ready plus Open Settings/Retry |
| Meaning | Ready snapshot, maintenance off | `Meaning · Snapshot from <date>` | Snapshot rows; selected detail carries exact date |

Every transition retains the same query-generation fence. A late result from
the losing leg may populate diagnostics but cannot merge into, append to, or
replace the painted list except through the explicit apply action above.

### Activation state machine

The switcher remains mounted until the Console opener returns a typed result.
The request holds the immutable target plus a cancellation token and advances
through `IDLE`, `OPENING_CANCELLABLE`, `COMMITTING`, and
`FAILURE_VISIBLE`. During either opening phase, duplicate Enter, mode changes,
search edits, and result movement are disabled without removing the highlighted
target.

The opener returns exactly one of:

```text
OPENED
CANCELLED_PRECOMMIT
NOT_FOUND
DATA_PROFILE_CHANGED
CHARACTER_UNAVAILABLE
FAILED
```

The opener's atomic `commit_started` acknowledgement is the cancellation
linearization point. If cancellation wins before it, the opener guarantees that
no Console tab, draft, focus, or current target changed and returns
`CANCELLED_PRECOMMIT`. If `commit_started` wins, the switcher enters
`COMMITTING`, says `Finishing…`, and ignores later Escape. The commit either
returns `OPENED` after the exact destination is current and visible or rolls
back to the unchanged prior Console state and returns `FAILED`; partial target
changes are not an outcome.

The total transition/recovery contract is:

| State/event | Opener result | Next presentation | Focus/action |
| --- | --- | --- | --- |
| Idle + activate | — | `OPENING_CANCELLABLE · Opening…` | Frozen highlighted row |
| Escape wins before commit | `CANCELLED_PRECOMMIT` | Idle, prior query/list restored | Search with stable highlight |
| Target disappears before commit | `NOT_FOUND` | `FAILURE_VISIBLE · Conversation no longer exists` | Refresh results; Open Library when valid |
| Data Profile changes before commit | `DATA_PROFILE_CHANGED` | `FAILURE_VISIBLE · Profile changed` | Refresh results |
| Character link becomes invalid | `CHARACTER_UNAVAILABLE` | `FAILURE_VISIBLE · Character unavailable` | Open Library / Repair in Library |
| Commit begins | — | `COMMITTING · Finishing…` | No interactive action |
| Atomic commit succeeds | `OPENED` | Exact Console destination | Modal dismisses |
| Opener fails with no committed change | `FAILED` | `FAILURE_VISIBLE · Could not open chat` | Retry |

Retry or Refresh transitions back through Idle with a new generation; Open
Library dismisses only after Library accepts the immutable context. Every
failure/cancellation restores query, stable highlight, scroll, and the specified
focus. `OPENED` alone dismisses directly to Console.

User-visible failures say `Conversation no longer exists`, `Profile changed`,
`Character unavailable`, or `Could not open chat`; technical authority details
remain in diagnostics. State-specific actions are Retry, Refresh results, Open
in Library, or Repair in Library. Failure never falls back to the current row,
a same-named conversation, another card, or another authority.

## Roleplay destination

### Deep-link contract

Context and `Ctrl+K` use a typed Roleplay deep link containing:

- a `ResolvedLocalCharacterKey`;
- optional conversation ID;
- optional validated search query;
- return destination; and
- return focus anchor.

An app-owned navigation coordinator receives the link before Roleplay changes
selection or unmounts an editor. It captures one aggregate
`RoleplayDraftSnapshot` from the incumbent guard, covering form edits,
character-visual authoring, shared-Persona visual authoring, Persona visual
authoring, attachments, and every in-flight save owner. The aggregate state is
`NO_CHANGES`, `DIRTY`, or `SAVE_IN_FLIGHT`. A dirty destination lists the
affected draft domains and presents `Save and continue`,
`Discard and continue`, and `Stay`. The pending link remains coordinator-owned
until one choice finishes:

- Save disables duplicate actions, invokes/awaits the incumbent aggregate save,
  re-snapshots every owner, and navigates only when all domains are clean. A
  failed or partially successful aggregate save keeps remaining drafts mounted,
  reports each failed domain, and focuses Retry/Stay.
- Discard requires its explicit action, names every affected draft domain,
  clears only the current Roleplay screen's aggregate drafts, then navigates.
- Stay abandons the pending link and returns focus to the original editor.
- A save already in flight waits for that same operation and never starts a
  second write; success re-snapshots, while failure follows the same
  draft-preserving recovery.

On success, Roleplay selects the exact local character card, opens its
Conversations view, and focuses the requested conversation or search input.
Missing or changed identity does not select a different card. A visible
`Back to Console` action uses the return destination/focus anchor; it appears
only for a valid originating Console link.

### Character-card browse and search

Only Character cards receive a Conversations view. Persona records do not.

The Roleplay surface distinguishes:

- `Find a character` for the card list; and
- `Search Samira’s chats` for the selected exact character.

The Conversations view provides complete keyset-paginated results and exact
data-authority-filtered totals. It uses the same effective date ordering as
Context.
Concurrent creation or ordering-key changes become visible on refresh/reselect;
deletion may remove a row on the next read. Paging does not skip or repeat an
unchanged row.

Keyword search is available for every resolved local character. Meaning search
appears only for local characters when a ready local semantic index exists. It
uses the same generation, 120 ms first-paint gate, direct bounded ANN retrieval,
and visible late-apply rules as `Ctrl+K`.

Keyword or Meaning matches may show the matching excerpt in the selected
preview/details region. Search snippets do not appear in collapsed list rows.

At 52×20 Roleplay uses one keyboard-complete pane at a time. Its complete graph
is `Character list → Card workspace → Conversations → Preview`. Ordinary Enter
on a character opens the incumbent Card workspace; it never bypasses card
editing. Card workspace adds a visible `Conversations (N)` action. A validated
conversation deep link may enter Conversations directly. Back/Escape returns
Preview → Conversations → Card workspace → Character list while restoring
selection by stable ID, scroll, and focus.

The current pane title, local scope, primary action, paging, and Back are always
visible; card art and secondary metadata yield first. The existing card editor,
visual/attachment authoring, preview, import/export, and `Send transcript to
Console draft` behavior remain owned by Card workspace and are neither hidden
nor retired by this programme. At wider sizes, the same owners may render side
by side, but focus order and actions remain identical.

### Preview and resume

- Click or Enter selects a row and opens its read-only preview.
- `Resume chat` is the primary action for a closed conversation.
- `Go to open chat` is the primary action for an already-open conversation.
- The contextual `r` shortcut invokes the exact visible primary action and is
  advertised only when available.
- Resume passes the validated conversation ID through the existing canonical
  Console opener. It does not copy or reconstruct a transcript.
- `Open in Library` remains available for archive/administrative work.
- Roleplay does not edit or delete conversation content.

An empty character history says `No chats with Samira yet` and offers
`Start in Console`. The new session receives the exact
`ResolvedLocalCharacterKey` through the existing character-session creation
path.

Historical saved identity remains authoritative. The current card name may
decorate a result only after exact identity resolution; it never rewrites the
conversation's saved identity snapshot.

## Keyword search foundation

PR 2 completes data-profile-safe local Keyword search before any semantic
implementation becomes reachable.

One pure **Selected Branch Eligibility Projector** owns the searchable-content
decision for both FTS and Meaning. Given an immutable conversation/message
snapshot, it returns the ordered eligible message IDs plus plain visible text,
or a typed exclusion reason. Both indexers consume this output; neither
reimplements role, visibility, branch, deletion, or attachment policy.

The existing broad `messages_fts` indexes all non-deleted message text and is
therefore not eligible for reuse. PR 2 creates a separately versioned derived
FTS generation whose schema and rebuild lifecycle are owned by this feature.
It must:

- include only eligible local conversations in the active Data Profile;
- key every record by resolved local character and conversation identity;
- index title, character display identity, and eligible visible user/assistant
  message text;
- exclude system, hidden thinking, tools, attachments, and unselected branches;
- support global and exact-character queries with data-authority filters applied
  inside the storage query rather than after result truncation;
- return deterministic ranks with stable identity/date tie-breakers;
- paginate with bounded keyset or rank-aware continuation; and
- revalidate candidate visibility and authority before display and activation.

FTS absence, migration, corruption, and rebuild have explicit status. Context
and Roleplay may fall back to date browsing; search never silently returns an
empty success state when its index is unavailable. A version mismatch builds a
new generation and atomically swaps it after validation; it never mixes rows
from two eligibility policies.

Keyword indexing is automatic local database maintenance, not a consent toggle.
PR 2 installs the schema and dormant `ensure_keyword_index()` service but starts
no background build and shows no UI on its own. The first owning surface in PR 3
invokes it and owns visible progress/failure while date browsing stays
available. After that explicit consumer activation, initial backfill runs off
the event loop and subsequent authoritative message, branch-selection,
conversation, and character-link commits enqueue idempotent derived updates
plus reconciliation. A prior ready FTS generation remains queryable during a
rebuild, labeled with its snapshot time; deleted or newly ineligible rows are
still suppressed by SQLite revalidation.

## Local Meaning search

### Consent and provider boundary

Meaning search is off by default. Enabling it requires:

- an explicitly supported local embedding provider;
- a locally present model artifact;
- a visible storage estimate and local-conversation eligibility count; and
- an explicit user action in the canonical Settings screen.

Existing general RAG or embedding configuration does not imply consent. A cloud
provider configured elsewhere is never selected for conversation indexing.
There is no remote fallback, automatic download, or indexing/query network
request. A missing local model blocks the build with a direct recovery path.

Settings identifies model name, provenance, artifact fingerprint, estimated
index size, and eligible local conversations. It also says
`Local chats only · nothing is uploaded` adjacent to the enabling action.

These controls live in a dedicated `Character chat search` group within the
canonical user-visible Settings > RAG category (stable category ID
`library-rag`). The group has its own consent, state, and
`data_authority_id`-keyed configuration; the active RAG configuration profile,
Library RAG search mode/model, and assistant-access settings do not enable or
namespace it implicitly.

### Eligible text

The semantic source text is **complete eligible visible conversation text**,
not a serialized full transcript.

Eligible content is limited to:

- user messages on the selected visible branch; and
- assistant messages or selected assistant variants on that branch.

It excludes:

- system prompts and instruction messages;
- hidden reasoning, chain-of-thought, and thinking fields;
- tool calls, arguments, results, and internal execution messages;
- attachments, derived attachment text, and attachment metadata;
- non-selected branches or response variants;
- deleted, trashed, hidden, or inaccessible messages; and
- known credential-bearing fields.

The eligibility policy is shared with conversation FTS so Keyword and Meaning
do not disagree about whether sensitive or hidden content is searchable.

The projector reads one SQLite transaction and records its source revision. It
walks `parent_message_id` from the conversation's local
`active_leaf_message_id` to the root and then restores root-to-leaf order. If a
legacy conversation has no active leaf, a single provable live leaf may be used;
multiple leaves are ambiguous. A dangling/cross-conversation parent, cycle,
ambiguous leaf, deleted selected node, or variant group without exactly one live
selected variant excludes **message body text for that conversation** and emits
a repairable diagnostic; it never falls back to indexing every message. The
non-sensitive title and saved character display identity may remain eligible.
Any branch/variant mutation invalidates the prior projection revision before a
new derived update can become ready.

### Chunk and storage model

Stable message- or exchange-level chunks replace the incumbent single joined
conversation document. Chunk identity includes the resolved local conversation
identity and stable branch/message identity. Unchanged chunks retain identity
across an incremental update.

The persistent backend reuses the existing Chroma client construction and safe
collection-naming utilities, but **not** the generic `VectorStore` protocol: that
protocol requires/stores document plaintext and collapses query failures into
empty results. PR 6 adds a narrow `CharacterConversationVectorStore` with
embeddings-only add/replace, metadata-filtered query, and delete-generation
operations. Its Chroma collection is scoped by data authority and generation,
hard-pins HNSW cosine distance at creation, and never supplies `documents`.

Queries return a typed `RESULTS`, `UNAVAILABLE`, `DAMAGED`, or `QUERY_ERROR`
outcome; `RESULTS([])` alone means no semantic matches. Ranking uses raw cosine
distance ascending, converts to no user-visible pseudo-probability, and
aggregates a conversation by its lowest eligible chunk distance. If the
optional local vector dependency is absent, Meaning is unavailable rather than
silently using a linear interaction-time scan. The vector store contains
embeddings plus the minimum identity/ranking metadata needed to retrieve
candidates. It does not duplicate transcript plaintext.
Titles and display labels remain in authoritative SQLite. A visible excerpt is
read from SQLite only after current Data Profile, data authority, branch selection,
and deletion state pass revalidation.

### Initial build and ongoing maintenance

Settings presents two separate controls:

1. `Index existing chats` builds a complete initial generation from eligible
   local conversations.
2. `Keep future chats indexed` subscribes to subsequent eligible create, append,
   edit, branch-selection, restore, and delete events.

The maintenance preference may be selected before the first build but activates
only after a complete generation is ready. A future-only incomplete corpus is
not offered because it would make corpus completeness unknowable.

If maintenance is turned off after a successful build, the ready corpus remains
queryable and is labeled `Snapshot from <date>`. Re-enabling maintenance first
reconciles changes since the recorded source revision.

Controls are:

- Index existing chats;
- Pause;
- Resume;
- Cancel current build;
- Rebuild;
- Delete semantic index; and
- enable/disable Keep future chats indexed.

Cancel deletes only the incomplete staging generation and preserves any prior
ready generation. Delete removes ready and staging generations, disables
ongoing maintenance, clears semantic query caches, and requires explicit
confirmation. It never deletes source conversations.

### Atomic generations

Every semantic generation records a manifest containing:

- data-authority identity;
- content authority (`local` in the first release);
- local model artifact digest;
- vector dimension and normalization;
- distance metric (`cosine`), distance-semantics version, and conversation-
  aggregation version;
- chunk configuration and version;
- eligibility-policy version;
- conversation-projection version; and
- source revision or equivalent content watermark.

Builds write into a staging generation. Validation confirms manifest
compatibility, expected chunk ownership, counts, and readable vectors before an
atomic ready-pointer swap. Partial generations are never queried. A failed or
cancelled rebuild leaves the previous ready generation active.

A model, vector dimension, normalization, chunk policy, eligibility policy, or
projection-version change creates a separate generation rather than mutating an
incompatible one in place.

### Lifecycle and deletion

Authoritative SQLite commits first. Best-effort post-commit events then upsert or
remove derived chunks for:

- message append or edit;
- selected branch/variant change;
- conversation title or resolved card-identity change;
- conversation delete, trash, restore, or hard purge;
- local character-link resolution change;
- Data Profile deletion;
- model removal; and
- detected index corruption.

The same authoritative transaction increments the conversation's
`character_search_revision` and writes a durable outbox record. Every vector
chunk carries that projection revision and an eligible-content digest. A worker
writes a complete replacement chunk set under the new revision, verifies its
count/digests, then advances the per-conversation ready-revision fence; old
chunks are cleaned afterward. Until that fence advances, SQLite revalidation
suppresses the whole conversation whenever an outbox record is pending or a
candidate revision/digest differs from the source. A crash between old/new
chunk operations can therefore cause temporary omission, never a mixed or stale
ready conversation. Idempotent outbox replay completes or cleans the update.

Every candidate is revalidated against SQLite before display and again before
navigation. Source deletion or loss of local identity resolution therefore
makes content unavailable immediately even while vector cleanup is pending. A
durable reconciliation ledger repairs missed best-effort events. Cleanup
failure is visible as `Cleanup pending`; it never makes stale content eligible.

Data Profile deletion removes that authority's semantic generations. Model removal
leaves source data untouched and marks dependent generations unavailable until
the model is restored or the user deletes/rebuilds the index.

### Settings interaction

`Character chat search` presents status before configuration. Its first screen
contains, in this order:

1. a plain summary such as `Meaning search: Off`, `Index: Not built`, and
   `Scope: 1,240 local chats · nothing is uploaded`;
2. the installed local model and storage estimate;
3. exactly one state-appropriate primary action: `Save settings`,
   `Index existing chats`, `Resume indexing`, `Retry`, or `Rebuild now`;
4. the independent `Keep future chats indexed` choice and its effective state;
5. progress/error detail; and
6. secondary pause, cancel, rebuild, and destructive delete controls.

At 52×20 the category uses the Settings detail pane's single scroll owner.
Status and the primary action precede every advanced field; nested scrolling,
side-by-side columns, and off-screen-only recovery are prohibited. `Delete
semantic index` remains a separately focused destructive action with the
confirmation copy `Deletes the search index, not your chats`. Model provenance,
fingerprint, batch tuning, and diagnostics sit under an initially collapsed
`Advanced` disclosure and remain reachable by keyboard and `/` Settings search.

Installed-model selection and `Keep future chats indexed` are ordinary staged
Settings fields and become authoritative only after `Save settings`. When future
maintenance is saved on before the first complete build, its effective state
says `Waiting for initial index`; it does not create an unknowable future-only
corpus. Index, pause, resume, cancel, rebuild, and delete are explicit immediate
job commands with ADR-033's inline label
`applies immediately - no Save needed`. Index, Rebuild, and Delete are disabled
while relevant Settings fields are dirty and always capture the last saved
configuration. Pause, Resume, and Cancel remain available for an existing job
and never read draft values. Command outcomes appear in the same group and never
depend on a second category-level Save.

Category Revert changes only staged model/maintenance preferences and cannot
undo completed job commands. `Delete semantic index` is unavailable with a
dirty draft; once invoked, it atomically disables the saved future-maintenance
preference and refreshes both original and draft state to Off. The incumbent
RAG `Backfill` command remains separately labeled `Library RAG backfill`; it
never builds or repairs this Character-chat index.

### User-visible states

Compact search surfaces show only the state needed for the current action:

- `Keyword`
- `Meaning off`
- `Building 342 of 1,240`
- `Paused`
- `Meaning ready`
- `Snapshot from Sep 3`
- `Rebuilding · previous index active`
- `Local model missing`
- `Storage full`
- `Index needs repair`
- `Cleanup pending`

Settings or F1 owns detailed explanation and recovery. Result rows do not repeat
index diagnostics.

## Concurrency, performance, and boundedness

- Database, FTS, embedding, and vector work runs off the Textual event loop.
- Every async operation carries exact data-authority/revision, surface,
  mode, query, selection, and generation ownership.
- Scope changes cancel or invalidate work; late completion cannot repaint a new
  profile, character, modal visit, or query.
- Context mounts at most four headers, five rows per expanded group, and eight
  search results.
- `Ctrl+K` preserves its 35-row modal ceiling, shows four results at 52×20, and
  fetches at most 50 conversations per result page.
- Roleplay fetches at most 20 conversations per keyset page and never mounts the
  entire corpus.
- Semantic queries embed only the query in the interaction path and obey the
  200-chunk/400-with-one-refill ANN bound above; they never scan or embed the
  corpus during interaction.
- Initial and reconciliation builds stream at most 128 chunks per batch with
  bounded memory, durable progress, and idempotent chunk writes.
- Pausing reaches a safe batch boundary; resuming continues from durable
  progress without duplicating ready chunks.
- Search and activation post visible busy state within 100 ms and do not block
  the Textual event loop for more than one 50 ms scheduling slice.
- On a recorded reference machine with 10,000 conversations and 250,000
  eligible messages, warm Keyword P95 is at most 300 ms and ready-index Meaning
  P95 is at most 2 seconds. A versioned fixture/query manifest fixes 30 IDs:
  eight title-keyword, eight body-keyword, eight semantic-only/no-shared-token,
  three no-match, and three Unicode/long-title queries. Each mode runs five
  discarded warm-ups followed by ten repetitions per query; P95 is the nearest-
  rank 95th percentile of the 300 measured durations. The evidence records
  hardware, model digest, fixture/query-manifest digest, corpus size, cold/warm
  state, and raw timings rather than treating the threshold as portable to every
  machine.
- Build progress becomes durable and visible at least every 128 chunks or one
  second, whichever comes first. Excluding the loaded embedding model and the
  vector backend's documented native cache, coordinator/batch RSS growth stays
  below 256 MiB on the reference fixture.

## Accessibility and usability

- Every status and state has a textual label; color is supplementary.
- Focus, candidate, expanded, current, and unavailable states remain visually
  distinguishable in theme variants.
- Controls are keyboard reachable in logical reading order.
- Headers implement Enter/Space and Left/Right tree semantics.
- Result lists implement Arrow, Home, End, Page Up, and Page Down behavior where
  paging exists.
- Escape restores the previous safe state or cancels a pre-commit action.
- Pointer and keyboard activation use the same immutable target path.
- Dynamic hints advertise only currently implemented actions.
- Search labels name their corpus: `Search character chats…`,
  `Search Samira’s chats`, `Keyword`, and `Meaning · Local`.
- Compact layouts omit secondary metadata before truncating the target title or
  hiding the primary action.
- First-use copy uses the user's concepts—Character card, chat, and open tab.
  `Authority`, `generation`, `ANN`, and `projection` remain diagnostic or
  implementation terms and never appear in primary workflow copy.

## Error and recovery matrix

| Condition | Presentation | Allowed recovery |
| --- | --- | --- |
| No conversations for a valid character | Honest empty state | Start in Console |
| Keyword index unavailable | `Keyword search unavailable` | Retry/rebuild; date browse remains |
| Semantic index off | `Meaning off` | Open Settings |
| Semantic build paused/cancelled | Exact progress state | Resume/restart |
| Local model missing | No remote fallback | Restore/select installed local model |
| Storage full | Preserve prior ready generation | Free space, Retry, or Delete index |
| Index damaged | Never query partial corrupted generation | Rebuild or Delete |
| Conversation deleted during activation | Preserve prior Console target | Refresh results/Open Library when applicable |
| Profile changed during work | Do not open or substitute | Return to the original profile or Refresh |
| Character card missing/source changed | Unavailable character | Open Library; Repair in Library when safely available |
| Cleanup failed | Candidate remains ineligible | Retry cleanup; source stays authoritative |

## Delivery sequence

Each PR receives one atomic Backlog task created during plan execution after a
fresh repository-wide task-ID sweep. No task may depend on a later PR.

### PR 1 — Decisions and contract alignment

- Create the new path-specific ADR.
- Amend ADR-004, ADR-030, ADR-037, ADR-046, ADR-083, and ADR-085 by exact
  filename.
- Link ADR-031 and ADR-033 as preserved contracts.
- Define terminology, ownership, source scope, and semantic consent before code.

### PR 2 — Authority-safe projection and Keyword search

- Build the shared read-only projection.
- Add deterministic data-authority backfill and Unavailable classification.
- Add same-data-authority repair candidate enumeration and compare-and-set
  provenance mutation services without exposing UI.
- Add the canonical Selected Branch Eligibility Projector and a separate derived
  local FTS generation; do not reuse broad `messages_fts`.
- Complete data-authority filtering, ordering, totals, paging, generations,
  and revalidation.
- Expose a dormant `ensure_keyword_index()` adapter for future surfaces without
  starting background work or changing UI behavior.

### PR 3 — Navigation recovery and Roleplay vertical slice

- Add the app-owned pre-navigation draft veto and typed cancellable Console
  activation result used by every later surface.
- Add Library's typed repair context, confirmation UI, refresh/failure focus,
  and return-anchor handling before any later surface advertises Repair.
- Extend the incumbent Roleplay conversation browser with the typed local deep
  link, exact identity selection, local-scope copy, per-character Keyword
  search, requested-row focus, and its 52×20 one-pane progression.
- Preserve and reuse its shipped complete local keyset browse, preview, exact
  Console resume, active-session reuse, and Library handoff paths.
- Preserve existing Roleplay card operations and Library handoff.

### PR 4 — Console Context Character section

- Add the bounded four-header tree, five-row groups, Unavailable recovery group,
  global Keyword search, versioned explicit-disclosure preference/migration,
  first-use behavior, empty state, state restoration, and Roleplay links.
- Move Character directly after Conversations and decouple it from avatar
  visibility. This PR does not render a Character-chats continuation link.

### PR 5 — `Ctrl+K` Character chats

- Add the third mode, F3 cycle, direct deep link, two-line result grammar,
  selected-only detail, 52×20 geometry, typed activation states, and preserved
  trust fixes.
- Add the capability-gated `Continue search in Character chats` control to
  Context.
- Keyword search only; no semantic UI is reachable yet.

### PR 6 — Semantic foundation, default-off and unreachable

- Add local-provider validation, eligibility/chunk contracts, no-plaintext
  embeddings-only Chroma/HNSW storage, typed query outcomes, generation
  manifests, atomic cutover, per-conversation revision fences/outbox,
  reconciliation, deletion, corruption handling, and direct bounded ANN query
  contracts.
- Keep the feature default-off and unreachable from production UI.

### PR 7 — Settings lifecycle and Roleplay Meaning slice

- Add consent, model/storage details, Index existing, Keep future, pause/resume,
  cancel, rebuild, delete, 52×20 single-scroll layout, and status/recovery
  controls in Settings > RAG > Character chat search.
- Ship the first end-to-end Meaning search in Roleplay for local characters.

### PR 8 — `Ctrl+K` Meaning integration and cross-surface hardening

- Add explicit `Meaning · Local`, the 120 ms first-paint gate, and visible
  `Apply Meaning results` handling for late vectors.
- Keep Context FTS-only with query transfer to `Ctrl+K`.
- Complete cross-surface real-TUI, lifecycle, documentation, and integration
  evidence. Compact, keyboard, focus, copy, and targeted rendering evidence
  remains an owning gate in each UI PR rather than being deferred here.

## Verification strategy

### Projection and authority tests

- Same character and conversation IDs in two local authorities never merge or
  leak.
- Server and cached-server rows never enter first-release browse, Keyword,
  Meaning, count, or activation results.
- A Data Profile/revision switch during query/backfill cannot commit old results.
- Local authority-null backfill succeeds only with unique same-database proof.
- An unresolved row never receives the current character merely because its
  display name or numeric ID matches.
- Missing/deleted cards enter Unavailable with only valid recovery actions.
- Repair enumerates only same-data-authority cards, requires an explicit user
  choice, rejects stale expected versions, and invalidates derived indexes after
  a successful compare-and-set mutation.

### Eligible-content tests

Seed a branched conversation containing unique canaries in user, assistant,
system, thinking, tool, attachment, and non-selected-branch content. Keyword and
Meaning search must retrieve only the selected visible user/assistant canaries.

Server/cached canaries must remain absent from both derived FTS and every
semantic generation regardless of authentication state elsewhere in the app.

### Generation and lifecycle tests

- Model, dimension, normalization, chunk-policy, eligibility-policy, and
  projection-version changes produce incompatible separate generations.
- A partial, failed, paused, or cancelled generation never affects ranking.
- Vector writes contain no document plaintext; cosine/scoring manifest mismatch
  rejects the generation, and query failure cannot masquerade as zero matches.
- Rebuild failure leaves the previous ready generation queryable.
- Pause/cancel/restart/resume does not duplicate chunks.
- Delete removes all generations and disables future indexing without deleting
  conversations.
- Delete, purge, branch change, profile delete, character-link change, and model
  removal make candidates unavailable immediately.
- Missed events converge through reconciliation.
- Failure injection between replacement chunk writes and ready-revision advance
  suppresses the affected conversation until idempotent outbox replay completes;
  old and new chunks never mix in a result.
- A cold index/query performs no network request and no unauthorized write.

### Interaction and trust tests

- Blank Active Enter targets the MRU other open tab.
- Explicit navigation and nonblank queries target only the committed highlighted
  identity.
- Double Enter cannot duplicate resume.
- F2 cannot rename a Character chats row.
- Context search restores browse disclosure/focus/scroll on clear/Escape.
- A dirty Roleplay card cannot be replaced before Save and continue, Discard
  and continue, or Stay completes; save failure preserves the draft.
- Successful deep-link navigation exposes Back to Console and restores its
  originating focus anchor.
- Meaning may win only before the 120 ms first-paint gate; after any list paint
  it requires `Apply Meaning results` and preserves stable selection.
- Deletion, profile change, or character-source change during activation
  preserves the prior Console
  tab and never falls back.
- Escape cancels activation only before opener commit; the switcher stays
  mounted until a typed result is returned, and double Enter never duplicates
  work.
- Active zero-match widening is labeled, shares only the Active/History query,
  and Character chats never auto-widens.

### Scale and rendering tests

- Large, branched conversations index incrementally with bounded memory.
- Meaning retrieval reaches a semantic canary that is absent from the query's
  lexical terms, proving direct vector retrieval rather than lexical candidate
  reranking.
- Roleplay paging does not skip or repeat unchanged rows.
- Context and switcher mount limits remain enforced.
- Production hierarchy and stylesheet compositor tests cover 52×20, 72×35,
  80×24, and 120×50 terminal cells.
- A real TUI pass verifies keyboard-only and pointer workflows, visible focus,
  exact resume, unavailable recovery, semantic status, and no hidden controls.
- iTerm2 and Windows Terminal evidence uses equal cell dimensions where the
  owning ADR/task requires terminal parity.
- Each UI PR records its own 52×20 keyboard, focus, truncation, empty/error, and
  primary-action evidence; PR 8 verifies their integrated composition.
- A moderated first-use check with at least three participants unfamiliar with
  the feature records whether at least two can find and resume an existing
  character chat within two minutes without intervention, correctly explain
  Character card versus chat versus open tab, and recover from one unavailable
  row. Failure returns the owning UI PR to revision rather than becoming a
  post-launch note.

## Documentation and completion gates

The final sequence updates:

- the Console sessions/tabs/workspaces guide;
- Roleplay character and conversation guidance;
- Settings local search/indexing guidance;
- F1 switcher and recovery copy;
- affected ADR links and Backlog tasks; and
- reproducible QA evidence.

No implementation PR is complete until its task's acceptance criteria,
Implementation Notes, targeted tests, static checks, documentation, ADR hygiene,
self-review, and task status meet the repository Definition of Done. The full
repository test suite is run only with explicit user approval; each PR must run
the complete targeted and reachable suites for its changed contracts.

## Acceptance contract for the completed eight-PR programme

The programme is complete when:

1. A first-time user can discover recent character conversations in Context,
   understand the distinction between Character cards, conversations, and
   Console tabs, and resume an exact conversation without a tutorial modal.
2. A power user can open `Ctrl+K`, preserve MRU-other blank Enter, switch among
   Active/History/Character chats, search the complete eligible local corpus,
   and activate only the committed highlighted target.
3. Character is always composed directly after Conversations, independent of
   avatar visibility; only its current or most-recent group starts expanded.
4. Context shows at most four headers. Every nonempty resolved group shows at
   most five date-sorted chats and ends with the exact local total in
   `View all N in Roleplay`.
5. Roleplay provides complete per-character browse/search, selected preview,
   52×20 one-pane progression, and exact Console resume without taking
   ownership of transcript mutation.
6. Dirty Roleplay card state is saved, explicitly discarded, or retained before
   navigation; a failed save never loses the draft.
7. Unresolved identities use the typed recovery variant, never inherit a
   current card/data authority, and expose mutation only through Library.
8. Keyword uses the canonical selected-branch eligibility projector and a
   separate local FTS generation before Meaning is enabled.
9. Meaning is local-only, opt-in, network-free, and direct bounded vector
   retrieval—not lexical reranking—and excludes system, thinking, tool,
   attachment, and non-selected-branch content.
10. Existing and future indexing consent is separate; pause, resume, cancel,
    rebuild, and delete behavior is truthful and atomic.
11. A painted result list never reorders silently. Meaning may become the first
    paint; later results require the visible apply action and preserve stable
    selection.
12. The switcher stays mounted through typed cancellable activation and closes
    only after the exact destination is current and visible.
13. Partial or stale index generations never affect ranking, and authoritative
    deletion or identity invalidation takes effect before asynchronous cleanup.
14. The specified first-use, identity, race, scale, latency, memory, keyboard,
    pointer, compact-layout, real-TUI, and equal-cell terminal evidence is
    recorded without weakening an incumbent gate.
