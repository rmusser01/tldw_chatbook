# Resume prior character chats from Roleplay

- **Date:** 2026-08-26
- **Status:** Proposed
- **Revision:** 4 — resolves the ID-only navigation copy contract during planning
- **Scope:** Local Roleplay conversation preview and native Console resume
- **Main task:** [TASK-22988](../../../backlog/tasks/task-22988%20-%20Resume-prior-character-chats-from-Roleplay.md)
- **Related follow-up:** [TASK-22453](../../../backlog/tasks/task-22453%20-%20Make-older-local-character-conversations-discoverable-in-Roleplay.md)

## Problem

Roleplay lists recent conversations for the selected local character, but selecting
one only opens a read-only preview. The available Console action copies a bounded
transcript into draft context; it does not reopen the persisted conversation, its
message tree, or an existing live Console session.

Users need to continue a prior character chat from the surface where they found it
without turning that conversation into RAG/context or silently refreshing its
behavior from the current character card.

## Decisions

1. Roleplay remains discovery and read-only preview; Console remains the only owner
   of writable live chat sessions.
2. **Resume chat** opens the existing live session for the selected persisted
   conversation or hydrates it through Console's canonical resume path.
3. Resume carries only a validated local conversation ID in fresh-screen navigation
   context. It does not use ChatHandoffPayload, copied transcript text, or RAG
   scope.
4. The persisted conversation is authoritative for historical character behavior.
   The current card cannot refresh its prompt, template inputs, greeting, provider,
   or behavioral settings.
5. Earlier pending Console intents receive their existing terminal or
   transient-release attempt first. Resume then runs last and becomes the final
   active-session target for that navigation.
6. Only the current 20 recent local conversations are in scope. Older discovery is
   tracked in TASK-22453.

## Goals

- Resume a recent local character conversation directly from its Roleplay preview.
- Activate an already-open matching Console session without duplication or loss of
  its live draft/settings.
- Otherwise restore the saved conversation tree within canonical safety limits,
  including its persisted prompt, roleplay provenance, active leaf, policies,
  speech preferences, and pinned prefill.
- Keep Resume distinct from **Send transcript to Console draft**, which remains a
  bounded context-staging action.
- Preserve the previously active Console session and draft when resume fails or is
  cancelled before commit.

## Non-goals

- Server-backed character conversations.
- Discovering more than the current 20 recent local conversations.
- Editing the transcript in Roleplay.
- Cloning, forking, importing, or resending the transcript.
- Persisting provider/model/sampling controls that the canonical resume path
  currently inherits from live/config state.
- Changing RAG scope, message-tree semantics, or the Console persistence model.
- Reconstructing historical raw template inputs that legacy conversations never
  stored.

## User experience

### Preview state

Selecting a saved conversation continues to open its read-only transcript preview.
Selection leaves focus in the conversation list so arrow-key browsing remains
stable. Transcript completion must not steal focus after the user has moved
elsewhere.

While the transcript preview is active:

- The inspector keeps character identity and the conversation list visible.
- Card-level actions such as **Chat now**, character-level
  **Send to Console draft**, exports, and delete are hidden.
- **Back to card** restores the card view and its actions.
- A non-scrolling note states:
  **Preview shows up to 200 messages. Resume opens the saved chat in Console.**

The note uses the helper's real unit: the existing limit applies to persisted
messages before UI pairing. It is always shown, so this feature does not add an
unreliable sentinel/count API. The separate 6,000-character transcript-draft cap
continues to drive only the staged handoff's truncation flag.

Preview load failures are distinguishable from an empty conversation:
**Couldn't load this preview. You can still resume the saved chat.** Resume remains
available because it does not depend on preview text.

### Action hierarchy and layout

The conversation action block uses three explicit rows at all supported widths:

1. Full-width primary **Resume chat**.
2. Full-width secondary **Send transcript to Console draft**.
3. Equal-width subdued **Back to card** and **Open in Library**.

The block is content-height rather than fixed to the current three-line toolbar.
The transcript keeps the remaining 1fr. No action is placed in an overflow menu,
and labels remain explicit rather than relying on memorized icons.

At the compact breakpoint and 80×24, the third row must remain contained without
horizontal clipping. Production CSS determines exact cell padding, but it must
preserve the three-row structure and may not truncate either label. The two
subdued controls divide the available width equally; compact spacing, rather than
a fourth row or overflow, keeps them within the supported minimum.

### Keyboard and focus

- The conversation list keeps focus after selection and asynchronous preview load.
- When preview is active, Resume becomes the center pane's first dynamic F6 target.
- The contextual footer truthfully advertises the existing F6 pane route.
- Within the center pane, traversal is:
  **Resume → Send transcript → Back → Open in Library → transcript scroll**.
- Escape from the preview keeps its existing Back-to-card behavior.
- Successful resume focuses the Console composer.
- Failed resume restores focus to the previously active Console composer.

No new terminal-convention shortcut is added.

### Resume activation

Resume is enabled as soon as a valid local conversation row opens. It is not gated
on transcript loading or provider readiness.

On press, the handler copies the preview's current conversation ID into a local,
immutable target, verifies that the preview still represents that row, and sets an
in-flight guard. The button becomes disabled and reads **Opening Console…**.
Repeated presses are ignored.

Posting navigation is fire-and-forget, so Roleplay does not pretend to receive a
destination acknowledgement. A short source-side fallback resets the button and
guard only if the same Roleplay screen remains mounted after dispatch; successful
navigation unmounts that source. The busy label receives an app-tier disabled style
that meets the project's measured contrast floor rather than inheriting the generic
stacked dimming.

### Success and failure

On success, Console displays the selected session and focuses its composer.

A stale source selection stays in Roleplay and reports:

> This conversation is no longer available. Refresh conversations and try again.

A missing or unreadable destination keeps the prior Console session active and
reports. The copy does not quote the title because the navigation contract
deliberately carries only the conversation ID:

> Couldn't resume this saved conversation: it was deleted or couldn't be read.
> Your previous Console chat is still active.

Use the existing notification system with a deliberately durable/long presentation
where supported; do not add a new recovery component solely for this feature.
Failure does not fall back to a copied transcript, clone, or blank chat.

## Architecture and ownership

Roleplay owns discovery and preview. Console owns live-session selection and saved
conversation hydration, consistent with
[ADR-026](../../../backlog/decisions/026-retire-chat-tab-conversation-entry-chain.md).

Resume uses a dedicated memory-only Console navigation-context key containing only
the normalized local conversation ID. The fresh ChatScreen receives that context
before mount, consistent with
[ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md).
Its context handler performs synchronous shape/bounds validation and captures the
target only. It cannot query widgets, hydrate, switch sessions, or render.

No new handoff channel, resume service, result class, or Roleplay-side loader is
introduced.

## Mount ordering

When a resume context is present, ChatScreen.on_mount() uses one ordered startup
branch instead of scheduling the session-switching consumers and Resume as
independent same-delay timers:

1. Establish the existing Console runtime/controller and mounted DOM.
2. Let older pending Console consumers run in their existing order and reach their
   normal acknowledge or transient-release outcome.
3. Prevent those consumers from claiming final composer focus or final active-session
   presentation while the explicit Resume target remains pending.
4. Resume the requested persisted conversation last.
5. Perform the final active-session synchronization and composer focus.
6. Resume ordinary non-conflicting startup timers/workers unchanged.

A transiently released earlier handoff remains owned by its existing channel; it is
not cleared, copied into the resumed chat, or represented as successfully settled.
It also does not override Resume during this navigation.

Ordinary initial transcript projection is deferred while the ordered branch runs, so
the prior or intermediate session is not presented as the selected target.

If the screen unmounts or its worker is cancelled before Resume commits, the ordered
branch stops. A newly created partial runtime session is rolled back; a pre-existing
live session is never deleted.

## Session selection and outcome

Resume reuses open_console_workspace_conversation() and canonical hydration. The
public opener should return the tri-state outcome its internal resume path already
uses rather than adding a new result abstraction:

- True: an existing session was activated or hydration committed successfully.
- False: the durable conversation is missing/terminally unavailable.
- None: a transient service/load outcome was already reported or released.

Cancellation is not folded into the tri-state result. ``asyncio.CancelledError``
propagates after the resume attempt restores the prior active session and removes
only the partial runtime session created by that attempt.

Matching is deterministic:

1. If the active live session already has the requested persisted conversation ID,
   keep and synchronize it.
2. Otherwise activate the first existing live session with that ID, preserving the
   store's current creation-order tie break for legacy duplicates.
3. Otherwise hydrate the persisted conversation once.

Active-match-first avoids switching away from a matching active duplicate. This
feature does not attempt to deduplicate sessions that already exist.

Hydration reads the saved tree within the existing 10,000-root/depth safety limits.
The feature does not call a provider. It must not describe this as literally
unbounded loading.

## Restoration authority

| State | Existing live match | Newly hydrated session | Current character card |
| --- | --- | --- | --- |
| Messages, branches, active leaf | Keep live state | Restore persisted tree within canonical limits | No authority |
| Resolved system prompt | Keep live value | Restore persisted conversation value | No authority |
| Raw roleplay template, user-name override | Keep live provenance | Restore versioned conversation metadata | No authority |
| Historical character name used by templates | Keep trusted snapshot | Restore trusted snapshot when present; do not guess for legacy data | No authority |
| Provider/model/sampling controls | Keep live values | Inherit canonical active/config defaults | No authority |
| Speech preferences, pinned prefill, library policy | Keep live values | Restore existing persisted values | No authority |
| Composer draft | Keep live draft | Use canonical newly-hydrated draft behavior; no durable draft is promised | No authority |
| Avatar/presentation asset | Keep current presentation | May resolve by matching local character ID | Presentation only |

## Historical character authority

Current resume code fetches the current card name and places it on the restored
session. Trusted-template projection can then re-expand {{char}} with that edited
name. To preserve historical behavior,
[ADR-046](../../../backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md)
must be amended before implementation:

- Add optional character_name_snapshot to console_roleplay_context.
- Advance new writes to metadata version 2.
- Version-2 readers accept versions 1 and 2; version 1 has no name snapshot.
- New character conversations persist the name used to materialize their saved
  prompt and trusted greeting projections.
- Existing version-1 conversations are never backfilled from the current card.
  Later owned-metadata writes may upgrade them to version 2 with the optional
  snapshot still absent.
- A valid snapshot supplies session.character_name and template projection input.
- For legacy conversations without a snapshot, the already-resolved saved system
  prompt remains authoritative. Raw templates are not re-expanded with a guessed
  current-card name.
- Current-card lookup may provide presentation-only assets. It cannot merge the
  card's current name, prompt, description, scenario, greeting, provider, or other
  behavior into the resumed session.

The snapshot uses the existing character display-name validation boundary. This is
an additive metadata evolution, not a database schema migration. Older builds treat
version 2 as unknown and keep the existing fail-closed no-overwrite behavior.

## Runtime failure atomicity

Before hydration, capture the prior active session ID and live session IDs. Track
whether this invocation created a restored session. Resume commits only after
hydration, policy reconciliation, roleplay restoration, and final core-state sync
succeed.

Before commit, failure or cancellation:

- removes only the exact runtime session created by this invocation;
- reactivates the prior session when it still exists;
- never closes or mutates a session that existed before Resume;
- never deletes or rewrites the durable conversation;
- never claims rollback across database boundaries.

A focused store/controller cleanup helper should remove that exact partial restored
session. Do not reuse a broad user-driven close/delete path or a helper restricted to
pristine sessions.

Any idempotent derived projection reconciliation reached by canonical hydration keeps
its existing owner and recovery policy. Runtime cleanup does not pretend to undo a
commit in another database.

## Data flow

    Recent local conversation row
      -> read-only Roleplay preview; card actions hidden
      -> Resume chat(captured conversation_id)
      -> NavigateToScreen(Console, resume-local-conversation context)
      -> pre-mount ChatScreen target capture only
      -> ordered post-mount branch
           -> older pending Console intents reach ack/transient-release outcome
           -> active matching session? keep + final sync
           -> other live matching session? activate + final sync
           -> otherwise hydrate persisted tree within canonical limits
                -> restore saved roleplay identity/settings/policies
                -> commit session switch
           -> focus composer

    Send transcript to Console draft
      -> existing 6,000-character bounded transcript handoff
      -> unchanged context staging; never aliases Resume

## Verification strategy

Only targeted tests related to modified paths are required unless the user later
requests a full sweep.

### Roleplay behavior

- Preview mode hides card-level chat/export/delete actions; Back restores them.
- Action block has the three specified rows and one primary action.
- At 80×24 and standard widths, all actions are contained and the transcript retains
  a usable scroll region under production consolidated CSS.
- Preview copy truthfully says up to 200 persisted messages; no sentinel API is added.
- Preview failure is distinct from empty, while Resume remains enabled.
- Resume captures the open row ID, ignores repeated presses, and resets its busy state
  if navigation leaves the source mounted.
- Busy disabled styling meets the project's measured contrast floor.
- Selection/preview completion does not steal focus; F6 lands on Resume and traversal
  follows the specified order.

### Console behavior

- Pre-mount navigation handling captures state without querying DOM or switching
  sessions.
- Earlier pending consumers reach their existing outcome first; Resume is the final
  active target and receives final focus.
- The canonical opener returns True, False, or None consistently to direct
  notification, focus, and rollback; cancellation rolls back and propagates.
- Active matching session wins over other duplicates and preserves its live draft.
- A closed conversation restores its branched tree, active leaf, persisted state, and
  roleplay provenance without a provider call.
- Cancellation/unmount before commit restores the prior active session and removes
  only a newly created partial runtime session.

### Historical authority and failure injection

- Metadata tests cover version-1 parse, version-2 round trip, future-version
  fail-closed behavior, sibling preservation, and name-snapshot validation.
- Renaming, editing, or deleting the current card does not alter a resumed saved
  prompt.
- Version-2 snapshots support stable template projection; legacy conversations keep
  their resolved prompt without guessing.
- Missing/load/post-create failure preserves the prior session and conversation/
  message authority.
- **Send transcript to Console draft** remains bounded staging and **Open in Library**
  remains unchanged.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Fresh-chat and resume actions compete | Hide card-level actions during preview; Resume is the sole primary |
| Compact controls clip | Explicit three-row layout, content-height block, production 80×24 containment test |
| Preview worker steals keyboard focus | Keep list focus and make Resume the dynamic center F6 target |
| Older Console intent overrides Resume | Ordered mount branch attempts older intents first and resumes last |
| Navigation failure leaves button stuck | Source-mounted fallback resets busy state |
| Post-create failure leaves partial session | Exact runtime rollback restores prior active session |
| Edited card changes historical behavior | Version-2 name snapshot; legacy saved prompt remains authoritative |
| Preview cap is misstated | Static “up to 200 messages” disclosure using the helper's real unit |
| Provider setup blocks history | No provider readiness gate or network call |

## ADR check

- **ADR required:** Yes, by amendment of an existing decision.
- **ADR path:**
  [ADR-046](../../../backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md)
  for the historical character-name projection input and metadata version 2.
- **Related decisions:**
  [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md)
  for fresh-screen memory-only navigation and destination ownership, and
  [ADR-026](../../../backlog/decisions/026-retire-chat-tab-conversation-entry-chain.md)
  for Console ownership of conversation loading.
- **Reason:** The feature changes a cross-module navigation contract and durable
  provenance needed to keep historical character behavior stable. It extends
  ADR-046 and follows ADR-026/ADR-033, so no new ADR is needed.

## Recommended implementation boundaries

1. Roleplay preview action hierarchy, copy, focus, and three-row layout.
2. One validated Console navigation-context key and ordered mount branch.
3. Tri-state return and active-match-first hardening on the canonical opener.
4. Exact runtime rollback for a newly created partial restored session.
5. ADR-046 amendment and version-2 roleplay metadata snapshot.
6. Targeted behavior, failure-injection, accessibility, and production-layout tests.

Do not add another handoff channel, queue, resume service, result class,
Roleplay-side hydration path, transcript-count service, or overflow menu.
