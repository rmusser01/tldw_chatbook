# Resume prior character chats from Roleplay

- **Date:** 2026-08-26
- **Status:** Proposed
- **Scope:** Local Roleplay conversation preview and native Console resume
- **Related follow-up:** [TASK-22453](../../../backlog/tasks/task-22453%20-%20Make-older-local-character-conversations-discoverable-in-Roleplay.md)
- **Superseding decision:** None

## Problem

Roleplay shows recent conversations for the selected local character, but selecting
one only opens a read-only transcript. The available Console action sends a bounded
copy of that transcript as draft context. It does not reopen the persisted
conversation, its message tree, or its existing live Console session. Users therefore
cannot continue a prior character chat from the surface where they found it.

## Goals

1. A user can resume a recent local character conversation directly from its
   Roleplay preview.
2. Resume activates an already-open Console session for the conversation when one
   exists; otherwise it hydrates the complete persisted conversation into Console.
3. The resumed chat continues with its saved character prompt, roleplay template
   provenance, display-name inputs, tree, and every conversation setting already
   owned by persistence rather than refreshing behavioral state from the current
   character card. An already-open session also keeps its live-only settings and
   draft.
4. The existing read-only preview, bounded **Send to Console draft**, and **Open in
   Library** behaviors remain available and semantically distinct.
5. Navigation, hydration, and failures do not duplicate sessions, target the wrong
   conversation, overwrite a draft, or leave a partial restored session active.

## Non-goals

- Server-backed character conversations.
- Discovering more than the current 20 recent local conversations. That work is
  tracked separately in TASK-22453.
- Turning the Roleplay preview into an editable transcript.
- Cloning, forking, or importing a transcript as a new conversation.
- Changing RAG scope, library policy, provider settings, message-tree semantics, or
  the Console persistence model.
- Reconstructing historical raw template inputs for legacy conversations that never
  stored them.

## User experience

### Conversation selection

Selecting a recent conversation continues to open its read-only transcript preview.
The preview action area becomes responsive and uses two rows rather than squeezing
four controls into the existing single row:

1. A full-width primary **Resume chat** button.
2. Secondary **Back to card**, **Send to Console draft**, and **Open in Library**
   actions.

The Resume control is available as soon as a valid local conversation row is
selected. It does not wait for transcript-preview loading or provider readiness:
resuming is local persistence work, and the user may configure a provider later.

The action is bound to the immutable conversation ID represented by the displayed
preview, not to a mutable controller selection read after the press. On activation it
becomes disabled and reads **Opening...** until navigation hands off successfully or
the source remains mounted after a failure. Repeated activation while it is in flight
is ignored.

The preview remains bounded to its existing 200 user/assistant-pair display. Its
worker reads one sentinel entry beyond the display bound so it can distinguish an
actually truncated result from a conversation with exactly 200 entries. When the
sentinel is present, the preview displays an accessible note: **Preview limited;
Resume loads the complete conversation.** The note avoids implying that Resume will
continue from only the visible excerpt.

### Successful resume

Resume opens the native Console and makes the selected conversation active. If that
conversation already has a live Console session, Console activates it and preserves
its current in-memory draft and state. If it is not open, Console loads its complete
persisted tree, active leaf, persisted prompt/prefill, policies, speech preferences,
and roleplay provenance through the existing saved-conversation hydration path. Other
provider/model controls retain the canonical resume behavior of inheriting from the
active Console session or configuration defaults; this feature does not claim they
were historically persisted. Focus moves to the composer after the destination has
rendered the selected conversation.

### Errors

A stale or invalid source selection produces one clear Roleplay notification and does
not navigate. A deleted or unreadable destination conversation produces one clear
notification in Console. The previously active session and its draft remain active.
Roleplay never falls back to sending transcript context, creating a clone, or opening
a blank chat, because those outcomes would misrepresent Resume.

## Architecture and ownership

Roleplay owns discovery and the read-only preview. Console remains the sole writable
owner of live chat sessions and saved-conversation hydration, consistent with
[ADR-026](../../../backlog/decisions/026-retire-chat-tab-conversation-entry-chain.md).

Resume uses a dedicated, memory-only Console navigation context containing only the
normalized local conversation ID. It does not use `ChatHandoffPayload`, the RAG scope,
or a copied transcript. Navigation context is appropriate because this is an explicit
destination selection, not queued content to be staged into an arbitrary chat. The
context remains subject to the fresh-screen and destination-ownership rules in
[ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md).

The incoming `ChatScreen` is constructed and receives navigation context before it is
mounted. Its context handler therefore performs only synchronous shape validation and
captures the target. It must not query widgets, hydrate a session, switch the store,
or render. A single post-mount coordinator performs the resume once the Console DOM
and controller are ready.

## Navigation and mount sequencing

The resume target is the exclusive active-session navigation intent for that
`ChatScreen` construction:

1. Roleplay validates that the active runtime is local, the preview still represents
   the pressed row, and the conversation ID is a nonblank bounded string.
2. Roleplay posts `NavigateToScreen(TAB_CHAT, <resume context>)` and marks the action
   in flight.
3. The new Console screen captures and validates the context before mount.
4. On mount, a coordinator processes Resume before the ordinary initial transcript
   sync and before any startup consumer that can switch the active session.
5. Competing draft, prompt, or fleet session-switch intents cannot override Resume on
   that mount. Their source-owned pending state is not silently cleared or retargeted;
   the coordinator leaves it unsettled for its existing lifecycle/retry semantics.
6. After Resume succeeds or fails atomically, Console performs one authoritative UI
   sync and focuses the composer only on success.

Delaying the ordinary first transcript sync when a resume target exists prevents a
one-frame flash of the previously active conversation. Central sequencing also avoids
depending on the ordering of several same-delay Textual timers, which is not a stable
coordination contract.

## Session selection and hydration

Console reuses `open_console_workspace_conversation()` and the canonical conversation
hydration service rather than adding a Roleplay-specific loader.

Matching is deterministic:

1. If the active live session already has the requested persisted conversation ID,
   keep it active and synchronize the view.
2. Otherwise activate the first existing live session with that persisted ID,
   preserving the store's established creation-order tie break for legacy duplicate
   sessions.
3. Otherwise load and hydrate the persisted conversation exactly once.

The active-session-first rule avoids switching away from a matching active duplicate.
The in-flight guard prevents two Roleplay presses from concurrently creating two
sessions. This design does not attempt to deduplicate legacy sessions that already
exist; it merely avoids creating another.

Resume loads the full persisted tree independently of the bounded Roleplay preview.
It preserves the stored active leaf and uses the existing Console projection and
policy reconciliation paths. Provider availability is not a resume prerequisite and
no provider call occurs during hydration.

## Historical character authority

The persisted conversation, not the current character card, is authoritative for a
resumed chat's behavioral state. Resume restores the saved resolved system prompt,
raw character system-template provenance where available, user display-name override,
speech and Console settings, and saved character identity.

Current resume code resolves a local character name from the current card. That name
can be fed back into trusted-template projection, so renaming or deleting a card can
silently change `{{char}}` expansion after resume. To keep the saved chat stable,
[ADR-046](../../../backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md)
must be amended before implementation:

- Add an optional `character_name_snapshot` to the versioned
  `console_roleplay_context` metadata object.
- Advance writes to metadata version 2. Version-2 readers accept both versions 1 and
  2; version 1 has no name snapshot. Existing future-version fail-closed behavior and
  merge-safe preservation of unrelated outer metadata remain unchanged.
- New character conversations persist the character name that was used to materialize
  the saved prompt and trusted greeting projections.
- Existing version-1 conversations are not backfilled from the current card. A later
  owned-metadata write may upgrade their object to version 2 while leaving the optional
  snapshot absent; absence remains meaningful historical uncertainty.
- On resume, a valid snapshot supplies `session.character_name` and the display label
  used for trusted-template projection. The current card cannot replace it.
- For legacy conversations without a snapshot, the already-resolved saved system
  prompt remains authoritative. Resume must not re-expand a raw template with a name
  fetched from the current card. Missing historical source data is left unresolved
  rather than guessed.
- A matching current card may still provide optional, presentation-only assets such as
  an avatar. Its current prompt, description, scenario, greeting, name, provider, or
  other behavior settings do not merge into the resumed session.

This is an additive metadata evolution and requires no database schema migration.
The snapshot uses the existing character-display-name validation boundary. Older
builds see version 2 as unknown and retain the existing fail-closed behavior rather
than overwriting it.

## Failure atomicity

Known service-unavailable, load, and missing-record outcomes already return without
replacing the active session. The design also covers unexpected failures after a
restored in-memory session has been created:

1. Before hydration, capture the prior active session ID and the set of live session
   IDs.
2. Track whether this invocation created a new restored session.
3. Do not consider Resume committed until hydration, policy reconciliation, character
   state restoration, and the destination's core-state sync have succeeded.
4. If an unexpected exception occurs before commit, remove only the exact session
   created by this invocation and reactivate the prior session when it still exists.
5. Never roll back, close, or mutate a live session that existed before Resume.
6. Never delete or rewrite the durable conversation as part of runtime rollback.

A focused store/controller helper should implement this cleanup rather than reusing a
broad close-session path whose side effects are intended for user-driven deletion or
whose preconditions only cover pristine sessions.

This is runtime-session atomicity, not a cross-database transaction claim. Any
idempotent derived projection reconciliation reached by canonical hydration keeps its
existing owner and recovery policy; runtime cleanup neither pretends to undo a commit
in another database nor mutates the conversation/message authority.

## Accessibility and responsive behavior

- **Resume chat** is the first action in keyboard traversal and retains an explicit
  text label; color is not its only primary-action cue.
- Busy state is announced by both the disabled state and **Opening...** text.
- Preview-limit and error copy is rendered as text that screen readers and terminal
  capture can observe, not as color-only decoration.
- The two-row action layout is verified at the app's supported narrow and standard
  terminal sizes with the production consolidated stylesheet.
- Existing global and screen keybinding conventions remain unchanged; the feature
  adds no terminal-convention shortcut.

## Data flow

```text
Recent local conversation row
  -> read-only Roleplay preview
  -> Resume chat(conversation_id captured from preview)
  -> NavigateToScreen(Console, resume-local-conversation context)
  -> pre-mount ChatScreen context capture only
  -> post-mount exclusive resume coordinator
       -> active matching session? keep + sync
       -> other live matching session? activate + sync
       -> otherwise load complete persisted tree
            -> hydrate inactive session
            -> restore saved roleplay identity/settings/policies
            -> commit session switch
       -> one authoritative render + composer focus

Send to Console draft
  -> existing bounded transcript handoff
  -> unchanged; never aliases Resume
```

## Verification strategy

Only targeted tests related to the modified paths are required unless a later
implementation request explicitly asks for a full suite.

### Unit and integration coverage

- Roleplay controller/screen: a selected local preview exposes Resume immediately;
  the event carries the preview-bound conversation ID; a second press while in flight
  is ignored; invalid/stale/server targets do not navigate.
- Roleplay preview: a 201st sentinel entry displays the limit disclosure, exactly 200
  entries do not falsely claim truncation, and Resume remains available when preview
  loading fails because it does not consume preview text.
- Console navigation context: applying the context before mount only captures state;
  no widget query, hydration, or session switch occurs until mount.
- Mount coordinator: Resume wins over same-mount session-switch consumers, ordinary
  initial sync is deferred, and exactly one final sync occurs.
- Existing-session behavior: the active matching session is kept; a non-active match
  is activated; no duplicate session is created and its in-memory draft survives.
- Hydration behavior: a closed conversation restores the complete branched tree,
  active leaf, saved settings and policies without a provider call.
- Metadata compatibility: version-1 parse, version-2 round trip, future-version
  fail-closed behavior, unrelated metadata preservation, and character-name snapshot
  validation.
- Historical authority: renaming, editing, or deleting the current card does not
  change a resumed saved prompt. A version-2 snapshot supports stable template
  projection; a legacy conversation without one keeps its resolved prompt and does
  not guess from the current card.
- Failure injection: missing/load failures and exceptions after session creation leave
  the prior session/draft active, remove only the newly-created partial runtime
  session, and do not mutate the durable conversation/message authority. No test
  should infer cross-database rollback from this runtime guarantee.
- Regression: **Send to Console draft** remains bounded context staging and **Open in
  Library** remains unchanged.

### Production-shaped UI verification

Use the real app hierarchy and `ConsolidatedCSSApp`/production stylesheet bundle to
verify the two-row action area at narrow and standard terminal sizes. Exercise the
keyboard path from conversation selection through Resume, assert the busy label and
focus result, and confirm that a capped preview visibly explains that Resume loads the
complete conversation.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Resume races another Console startup consumer | One exclusive mount coordinator gives Resume active-session precedence. |
| Previous transcript flashes before the selected chat | Defer ordinary initial sync until Resume settles. |
| Two presses create duplicate restored sessions | Preview-bound immutable ID plus an in-flight guard and existing-session recheck. |
| Unexpected post-create failure leaves a partial session | Scoped rollback removes only the session created by this invocation and restores the prior active session. |
| Edited character card changes an old chat | Persist the historical character-name projection input; legacy chats keep the saved resolved prompt. |
| Preview looks like the full resume context | Explicit disclosure at the preview bound; hydration always loads the full tree. |
| Provider setup blocks local history access | Resume performs no provider readiness gate or network call. |
| Four actions overflow narrow terminals | Responsive two-row action area verified under production CSS. |

## ADR check

- **ADR required:** Yes, by amendment of an existing decision.
- **ADR path:**
  [ADR-046](../../../backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md)
  (historical character-name projection input and metadata version 2).
- **Related existing decisions:**
  [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md)
  for memory-only navigation/destination ownership and
  [ADR-026](../../../backlog/decisions/026-retire-chat-tab-conversation-entry-chain.md)
  for Console ownership of conversation loading.
- **Reason:** The feature changes a cross-module navigation contract and the durable
  provenance needed to keep resumed character behavior historically stable. No new
  ADR is needed because the decision extends ADR-046 and follows ADR-026/ADR-033.

## Recommended implementation boundaries

Keep the change narrow:

1. Roleplay preview action and responsive layout.
2. One dedicated Console resume navigation context and post-mount coordinator.
3. Small hardening changes to canonical session matching and hydration rollback.
4. ADR-046 plus versioned roleplay-metadata snapshot support.
5. Targeted behavior, failure-injection, and production-shaped layout tests.

Do not introduce another handoff store, resume service, Roleplay-side hydration path,
or transcript-copy abstraction.
