# ADR-120: Character conversation navigation and local semantic search

Status: Accepted
Renumbering: provisional ADR-116 moved to ADR-120 for Keyword release isolation;
the shipped Schedules ADR retains 116. Local/remote ref and worktree allocation
review on 2026-09-05 confirms 117–119 occupied and 120 owned by this programme.

Delivery status: navigation and Keyword only in the 2026-09-05 delivery scope.
Meaning and its Settings/runtime contracts below remain deferred requirements.
Date: 2026-09-03
Related Task: [TASK-31241](../tasks/task-31241%20-%20Align-character-conversation-navigation-decisions.md)
Related Spec: [Character Conversation Navigation and Local Meaning Search Design](../../Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md)
Preserves: ADR-031, ADR-033
Amends: ADR-004 (`004-personas-destination-native-workbench.md`), ADR-030,
  ADR-037, ADR-046, ADR-083, ADR-085

## Decision

Character-conversation discovery and navigation use one read-only application
projection over the active Data Profile's authoritative local conversation
database. The first release does not browse, count, index, search, or resume live
server or cached-server conversations. Character cards are eligible; Personas
are not.

The owned contracts are:

```text
Identity: ResolvedLocalCharacterKey | UnresolvedConversationKey
Activation: Console-owned, cancellable before commit_started, result typed
Repair: Library-only, same-data-authority, explicit confirmation, CAS
Keyword: separate selected-branch FTS generation
Meaning: local-only, opt-in, embeddings-only, atomic ready generations
Surfaces: Context bounded; Ctrl+K operational; Roleplay complete browse
```

### Authority and identity

**Data Profile** means the selected local conversation database and its durable
`local_authority_id`, exposed to this contract as `data_authority_id`. **RAG
configuration profile** means the independently selectable retrieval-settings
profile. Architecture, persistence, and diagnostics must use those qualified
terms; an absolute database path, display label, current card, character name,
server routing identity, or RAG configuration-profile ID never establishes
conversation authority.

Serialized character-conversation identity is a closed, versioned tagged union:

```text
ResolvedLocalCharacterKey(data_authority_id, character_id)
UnresolvedConversationKey(data_authority_id, conversation_id)
```

An activatable target combines a `ResolvedLocalCharacterKey` with the exact
conversation ID. An unresolved key remains listable and can be sent to Library,
but it is never activatable or silently promoted. Every query and activation
captures the Data Profile authority plus revision and fails closed if either
changes. Server and cached-server identities do not enter this union in the
first release.

### Selected-branch corpus and derived search generations

One Selected Branch Eligibility Projector determines the searchable content for
both Keyword and Meaning. It includes only visible user messages and selected
assistant messages or variants on the selected live branch. It excludes system
and instruction content, hidden thinking, tool calls and results, attachments,
non-selected branches or variants, and deleted, hidden, inaccessible, invalid,
or credential-bearing content.

Keyword owns a separately versioned FTS generation; the broad incumbent
`messages_fts` is ineligible because it indexes content outside this policy.
FTS generations are scoped by `data_authority_id`, built separately, validated,
and atomically made ready. Authority filters apply in storage queries before
result bounds, and candidates are revalidated against authoritative SQLite
state before display and activation.

Meaning is off by default and requires an explicit action in canonical Settings,
a supported local embedding provider, and a locally present model artifact.
General RAG or embedding configuration does not imply consent. Indexing and
queries make no network request, have no remote fallback or automatic model
download, and store embeddings plus minimum identity/ranking metadata without
transcript plaintext.

Meaning performs bounded direct ANN retrieval using cosine distance; it is not
lexical candidate reranking. Builds write incompatible or incomplete work into
staging generations. Only a compatible, validated generation becomes queryable
through an atomic ready-pointer swap. Failed, paused, or cancelled work leaves
any prior ready generation intact. Authoritative SQLite commits and revision
fences make deletions, branch changes, identity invalidation, and Data Profile
changes ineligible before asynchronous derived cleanup completes.

### Activation, repair, and surface ownership

Context, `Ctrl+K`, and Roleplay capture immutable targets and call the canonical
Console opener. They do not create resume paths or mutate transcripts. The
opener returns exactly `OPENED`, `CANCELLED_PRECOMMIT`, `NOT_FOUND`,
`DATA_PROFILE_CHANGED`, `CHARACTER_UNAVAILABLE`, or `FAILED`. Its atomic
`commit_started` acknowledgement is the cancellation linearization point:
cancellation that wins before it guarantees no Console target, tab, draft, or
focus change; once commit starts, the caller waits for success or atomic rollback.
Only `OPENED`, after the exact destination is current and visible, dismisses the
calling activation surface.

Roleplay deep links pass through an app-owned navigation coordinator before
selection changes. The coordinator snapshots all incumbent Roleplay draft and
in-flight save owners as one aggregate and requires Save and continue, Discard
and continue, or Stay. Navigation proceeds only after every owned draft domain
is clean; failure keeps drafts mounted and recoverable.

Unresolved identity repair is owned by Library. It accepts the exact
`UnresolvedConversationKey`, expected conversation version, saved historical
identity snapshot, and return anchor; enumerates only live character cards from
the same `data_authority_id`; shows the old and proposed identity; requires an
explicit user selection and confirmation; and commits through compare-and-set.
Context, `Ctrl+K`, and Roleplay may navigate to that flow but cannot repair.
Names never select or preselect a repair target.

Task4 release clarification (2026-09-05): unavailable inspection and unavailable
browse deep links originate only from Console Context and retain its Character
return anchor. Both their typed objects and wire payloads reject other origins;
their native return action is **Back to Console**. Incumbent repair continues to
accept its existing Console and Roleplay return targets. This scopes the two
new Task4 routes, not future navigation origins; later delivery must explicitly
define any extension. Returning reveals Character transiently and does not write
manual disclosure preferences.

Task5 admission correction (2026-09-06, TASK-31245): a switcher-to-Library
inspection is accepted only after Library prepares the exact immutable local
conversation and admits its existing save guards. Generic destination stack
ownership is not evidence that the requested inspection was accepted. Library
owns preparation through its existing bounded local conversation locator; the
app's existing navigation coordinator owns the one-way commit and screen
transfer. Preparation must not replace the retained Library view or dismiss the
source switcher. It captures the database, authority, exact conversation,
navigation generation, and originating visit, and rejects stale or cancelled
requests before commit.

Cold and retained Library screens use the same prepared inspection contract.
Commit installs the validated route and exact reader selection, without a
second competing admission lookup on mount. Later transcript rendering remains
Library-owned and is not confused with route admission. Until commit begins,
rejection or cancellation preserves the switcher's query, highlight, scroll,
and recovery identity; after commit begins, source controls cannot cancel or
retarget the transfer. This does not broaden repair authority or return origins,
create a parallel navigation service, or redesign generic overlay rollback for
an unrelated late synchronous screen-switch failure. The existing Context
Character return anchor remains unchanged.

Surface roles remain distinct:

- Console Context owns a bounded ambient Character section directly after
  Conversations. It is always composed, independent of avatar visibility, and
  exposes bounded recent groups plus global local Keyword entry.
- `Ctrl+K` remains an operational switchboard. Active remains the default,
  blank Active Enter preserves MRU-other behavior, and F3 cycles Active,
  History, and Character chats under [ADR-031](031-tui-keybinding-and-footer-hint-conventions.md).
- Roleplay owns complete, keyset-paginated browsing, search, preview, and exact
  Console resume for one resolved local character. Library retains the global
  archive and conversation-administration role.

Keyword and Meaning asynchronous results share authority, visit, mode, query,
and generation fences. Meaning may win first paint, but once any list paints, a
late Meaning result is exposed through a visible, focusable Apply action and
never silently reorders or retargets the list.

### Settings ownership

Installed-model selection and `Keep future chats indexed` are saved staged
preferences. The future-maintenance preference may be saved before an initial
build, but its effective state remains `Waiting for initial index`. Maintenance
does not subscribe or write before a complete ready initial generation is
published by `Index existing chats`. A future-only incomplete corpus is never
offered.

`Index existing chats`, Pause, Resume, Cancel, Rebuild, and Delete semantic
index are explicit immediate maintenance commands labeled `applies immediately
- no Save needed` under the unchanged
[ADR-033](033-settings-commit-models-three-honestly-labeled.md) commit-model
contract. Immediate commands capture saved configuration, never draft values.
Index, Rebuild, and Delete are unavailable while relevant Settings fields are
dirty; Pause, Resume, and Cancel remain available for an existing job and never
read draft values.

Delete requires explicit confirmation and atomically removes ready and staging
generations, disables the saved `Keep future chats indexed` preference, and
clears semantic query caches. It refreshes both original and draft Settings state
to Off and never deletes source conversations. No maintenance worker may
repopulate the deleted index until the user saves a new preference and
completes a new initial ready generation.

## Context

Recent character conversations need to be discoverable from Console Context,
the operational switcher, and Roleplay without creating three identity,
navigation, or search implementations. Existing historical-display and exact
ID-only resume behavior provides a trustworthy base, but it does not define
local authority scoping, unresolved identity, cross-surface cancellation,
Roleplay draft veto, selected-branch indexing, or semantic consent and
generation lifecycle.

This decision establishes those boundaries before runtime work so the eight-PR
programme can ship independently without a partial surface inventing a broader
authority or mutation right.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Include cached-server conversations in the first release | Cached and live server discovery need explicit remote authority, completeness, authorization, paging, and resume contracts. Local totals and labels would become misleading before those contracts exist. |
| Reuse broad `messages_fts` | It indexes content outside the selected visible user/assistant branch and cannot provide the feature's eligibility-policy generations or authority-safe lifecycle. |
| Use lexical candidates and rerank them for Meaning | Semantic-only matches with no shared token would remain undiscoverable. Meaning is direct bounded ANN retrieval over the eligible local corpus. |
| Use remote embeddings or silently fall back to them | Conversation content would leave the device without the separate consent and provider boundary this feature promises. Meaning is local-only and fails closed when its local model is unavailable. |
| Let each surface own resume | Parallel openers would drift on revalidation, cancellation, rollback, open-tab reuse, and destination visibility. Console remains the single activation owner. |
| Repair unresolved identity by character name | Names are mutable and non-unique and cannot prove data authority. Repair requires same-authority candidates, explicit confirmation, and compare-and-set. |
| Automatically apply late Meaning results | Reordering a painted list can move selection or change the target under keyboard and pointer activation. Late results require a visible Apply action. |

## Consequences

- First-release character conversation labels and totals always mean the active
  Data Profile's local corpus; server and cached-server expansion requires a new
  decision.
- Runtime tasks must share the closed identity union, eligibility projector,
  Console opener, and authoritative revalidation rather than duplicate them in
  widgets.
- Keyword can ship before Meaning because it owns an independent complete FTS
  generation and no semantic consent state.
- Meaning requires explicit local lifecycle, storage, failure, deletion, and
  rebuild controls; optional dependencies degrade to visible unavailability.
- A ready snapshot may remain queryable while maintenance is off or a rebuild
  stages, but partial or stale content never becomes eligible.
- ADR-031 continues to own global/reserved keys and truthful hints; this decision
  introduces no new global binding.
- ADR-033 continues to own Settings transaction labels; this decision assigns
  each preference or maintenance action to an existing commit model.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md)
- [TASK-31241](../tasks/task-31241%20-%20Align-character-conversation-navigation-decisions.md)
- [ADR-004: Personas destination-native workbench](004-personas-destination-native-workbench.md)
- [ADR-031: TUI keybinding and footer-hint conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-033: Settings commit models](033-settings-commit-models-three-honestly-labeled.md)
