# tldw_chatbook Characters, Persona Profiles, And Main-Chat Runtime Alignment Vertical Design

**Date:** 2026-04-19  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next parity vertical after chat/conversations: align `tldw_chatbook` characters, persona profiles, exemplar systems, and character/persona session runtime behavior with `tldw_server`, while keeping `tldw_chatbook` standalone and local-first and making main chat the primary runtime surface for launched character-backed and persona-backed sessions.

## Context

The current `tldw_chatbook` already has substantial local character support:

- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- `tldw_chatbook/UI/Conv_Char_Window.py`
- `tldw_chatbook/UI/Screens/ccp_screen.py`
- `tldw_chatbook/UI/CCP_Modules/`
- `tldw_chatbook/Widgets/CCP_Widgets/`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`

That local character stack is not empty or provisional. It already supports:

- character card CRUD
- card import/export
- character-linked conversations and messages
- alternate greetings on character cards
- a dedicated CCP discovery surface

At the same time, the broader server-side character and persona surface is now more structured and split across multiple endpoints:

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/persona.py`
- `tldw_server/tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- `tldw_server/tldw_Server_API/app/api/v1/schemas/character_schemas.py`
- `tldw_server/tldw_Server_API/app/api/v1/schemas/persona.py`

The server now distinguishes at least four related but different domains:

- character cards
- character chat sessions and messages
- character-owned exemplars
- persona profiles and persona-profile exemplars

`tldw_chatbook` also already carries some of the newer assistant identity contract in the general conversation model:

- `assistant_kind`
- `assistant_id`
- `character_id`
- `persona_memory_mode`

However, the runtime, discovery, and management surfaces are still fragmented:

- CCP is still oriented around older local assumptions
- main chat and CCP do not yet share one explicit launched-session contract
- there is no obvious app-wide local/server source abstraction for this domain
- local persona profile and local exemplar support do not appear to be fully modeled yet
- some legacy local character save paths still encode assistant identity using display names instead of stable IDs

This is the parity gap this vertical addresses.

## Product Decisions

The following scope and behavior decisions are fixed for this vertical:

- This vertical is intentionally broader than a character-only seam. It covers:
  - characters
  - persona profiles
  - character exemplars
  - persona-profile exemplars
  - character/persona session launch and runtime handoff into main chat
- `tldw_chatbook` remains a standalone local-first application.
- `Server mode` is the reference behavior for this vertical where server support exists.
- In `server mode`, supported reads, writes, and execution are server-authoritative.
- In `server mode`, writes are server-primary only. The client updates local UI/cache after confirmed server success.
- In `local mode`, the same conceptual entities must still exist and remain usable locally.
- Cross-surface entity, session, exemplar, greeting, and preset IDs are string-first canonical IDs.
- `assistant_id` must always store a stable canonical entity ID, never a display name.
- There is no mixed local/remote CCP list view in this vertical. Mode determines the backing source for the active CCP entity area.
- CCP becomes a dual-entity management surface, not just a character screen.
- Main chat becomes the primary runtime surface for launched character-backed and persona-backed sessions.
- CCP remains the discovery and launch surface for character-backed and persona-backed sessions.
- Character-backed sessions must not appear in the general main-chat history list.
- Persona-backed sessions must not appear in the general main-chat history list.
- Persona-backed chats are only visible in the CCP context of the currently selected persona profile.
- Character-backed chats are only visible in the CCP context of the currently selected character.
- Runtime backend is session-bound at creation or launch time. Flipping local/server mode later must not silently rewrite an already open session.
- Changing CCP backing mode must clear entity selection, session selection, and cached discovery results before refetching from the newly active backend.
- Restored character/persona tabs may remain open and runnable even when CCP is currently focused on a different entity, but CCP discovery lists remain selected-entity scoped.
- A running character-backed or persona-backed session in main chat does not hard-lock provider/model/runtime controls.
- Greetings and presets must be execution-compatible in this vertical, but their UI remains mostly default-driven rather than fully surfaced.
- Full persona-profile CRUD is in scope, not just persona exemplar CRUD.
- Character exemplars and persona-profile exemplars are separate systems and must stay separate in the design and implementation.
- Sync, dual-write, and local/remote reconciliation remain out of scope.
- The general chat history remains for ordinary non-character, non-persona conversations only.

## In Scope

- Add server-backed `tldw_api` coverage for:
  - character CRUD
  - character session CRUD
  - character message CRUD
  - character exemplar CRUD/search/debug
  - persona profile CRUD
  - persona exemplar CRUD/import/review
  - character-session greeting listing/selection
  - character-session preset listing/create/update/delete
- Add local storage and service coverage for:
  - persona profile CRUD
  - character exemplars
  - persona exemplars
  - mode-aware session launch metadata
- Introduce domain-specific local/server service layers for characters and persona profiles, parallel to the recent notes/chatbooks service pattern.
- Refactor CCP into a dual-entity management surface with first-class areas for:
  - characters
  - persona profiles
- Add explicit launch flows from CCP into main chat for:
  - character-backed sessions
  - persona-backed sessions
- Add explicit runtime metadata so main chat can distinguish:
  - local-backed sessions
  - server-backed sessions
  - character-backed sessions
  - persona-backed sessions
- Add explicit discovery/visibility metadata so main chat can run these sessions without leaking them into the ordinary history/search surfaces.
- Backfill or default legacy conversation rows and restored-tab payloads so new runtime/discovery fields exist before the new filters and handoff rules go live.
- Normalize legacy name-shaped assistant identity onto canonical string IDs before CCP-to-main-chat launch work.
- Normalize CCP session/conversation identity to string-first contracts before handoff work.
- Preserve ordinary general-chat behavior for non-character, non-persona sessions.
- Add regression coverage for:
  - mode-aware CCP reads/writes
  - launch/handoff correctness
  - history visibility rules
  - selected-entity scoping rules
  - local persona/profile exemplar behavior

## Out Of Scope

- Sync and dual-write semantics
- Cross-device or local/server reconciliation
- Mixed local and remote entity lists in the same CCP view
- Full greeting-management UI
- Full preset-management UI beyond minimal execution-compatible controls
- Replacing CCP with a completely new product surface
- Major redesign of the visible main chat layout
- Prompts, dictionaries, and unrelated CCP domains unless touched incidentally by required refactors
- Hermes-inspired global job centers or approval centers

## Approaches Considered

### Option A: Thin adapter vertical

Keep CCP mostly as-is, add local/server character clients and adapters, and bolt on an `Open In Chat` bridge.

Why not chosen:

- Preserves too many duplicated local runtime paths
- Makes later convergence harder
- Under-specifies persona profile management
- Encourages more temporary compatibility shims

### Option B: Contract-first convergence

Normalize characters and sessions onto the shared conversation/session seam while keeping CCP primarily a management and launch surface.

Why not chosen:

- Better than the adapter path, but still too conservative for the requested broader server-led alignment
- Risks treating the server model as optional instead of reference behavior

### Option C: Server-led character and persona surface

Make the server-side entity model and runtime behavior the reference shape for this domain, mirror it locally where needed, and converge launched sessions onto the main chat runtime.

Why chosen:

- Matches the explicit request to use the broader server stack as the reference
- Avoids baking in another temporary CCP-specific runtime model
- Aligns with the conversation/session parity work already landed
- Produces clearer future sync boundaries because backing mode and discovery ownership are explicit

## Chosen Model

This vertical treats the server-side entity model as the reference shape and treats local support as a mirrored local-first implementation of that same conceptual model.

That means:

- characters and persona profiles are both first-class CCP-managed entities
- CCP owns discovery and launch of their sessions
- main chat owns session runtime after launch
- launched character/persona sessions carry explicit metadata for runtime backend and discovery ownership
- general chat history remains scoped to ordinary conversations only
- local support for persona profiles and exemplars is not optional glue code; it is part of the vertical itself

The goal is not just to add more server endpoints. The goal is to produce one coherent runtime model in which:

- CCP knows what to list
- main chat knows what it is running
- general history knows what to exclude
- local and server modes can implement the same behavior with different backends

## Architecture

### 1. Domain Mode And Service Layer

This vertical must not assume an app-wide source-mode abstraction already exists for this domain. The implementation should add one explicitly for characters/persona runtime instead of scattering mode checks across CCP and chat handlers.

At minimum, the design requires:

- a `local` or `server` backing mode for this domain
- a server-backed service layer for character/persona resources
- a local-backed service layer for the same conceptual resources
- one thin mode-aware facade that CCP and main chat can call

This should follow the service pattern already used elsewhere in the repo:

- `tldw_chatbook/Notes/server_notes_workspace_service.py`
- `tldw_chatbook/Chatbooks/server_chatbook_service.py`

The UI should not call `tldw_api` methods directly.

### 2. Entity Separation

The implementation must keep these as separate concepts:

#### Characters

- character card CRUD
- character session discovery/launch
- character-owned exemplars
- character greetings

#### Persona Profiles

- persona profile CRUD
- persona session discovery/launch
- persona-profile exemplars
- persona-specific import/review flows

#### Conversations And Runtime

- one launched-session contract for main chat runtime
- explicit discovery ownership rules
- explicit runtime backend rules

This is not one generic `assistant profile` object. The spec and code should preserve the domain distinctions the server already uses.

### 3. Session Contract And Runtime Metadata

The main chat session contract must grow explicit metadata beyond the current conversation identity fields.

The minimum required metadata for this vertical is:

- `conversation_id`
- `assistant_kind`
- `assistant_id`
-   canonical string ID for the selected character or persona
-   never a display name
- `character_id`
-   optional legacy/local bridge while integer-backed local character rows still exist
-   not the canonical cross-surface identity
- `persona_memory_mode`
- `runtime_backend`
  - `local`
  - `server`
- `discovery_owner`
  - `general_chat`
  - `ccp_character`
  - `ccp_persona`
- `discovery_entity_id`
  - selected character id for character-backed sessions
  - selected persona id for persona-backed sessions

Rationale:

- `runtime_backend` prevents server-backed sessions from falling through local persistence flows accidentally
- `discovery_owner` prevents launched sessions from leaking into the wrong history surfaces
- `discovery_entity_id` enforces the rule that character/persona chats only show inside the currently selected entity context in CCP

Canonical identity rules:

- all entity, session, exemplar, greeting, and preset IDs exposed across CCP, main chat, services, and `tldw_api` are string-first contracts
- local integer primary keys may remain internal storage details, but adapters must translate them before server calls, launch handoff, or restored tab state serialization
- display titles and labels must come from dedicated display fields rather than overloading `assistant_id`

Session lifecycle rules:

- `runtime_backend` is bound when a session is created or launched and does not change retroactively when the user flips source mode later
- restored tabs keep their stored runtime/discovery metadata even when the currently selected CCP entity or mode differs
- launching a session that is already open focuses the existing tab keyed by `(runtime_backend, conversation_id)` rather than creating a duplicate tab
- if the backing entity or session has been deleted or is unavailable, the open/restored tab remains visible but must surface a degraded read-only or relaunch-required state instead of silently mutating into a different session

### 4. CCP Information Architecture

CCP becomes a dual-entity management surface rather than a catch-all mixed local screen.

The visible structure should become:

- `Characters`
  - list/search/select
  - card details
  - edit/create/delete
  - exemplars
  - session discovery/launch
- `Persona Profiles`
  - list/search/select
  - profile details
  - edit/create/delete
  - exemplars
  - session discovery/launch

The spec should not require a full visual redesign, but it must require enough UI structure that persona profiles are not hidden as an afterthought inside character flows.

Switching the CCP backing mode must:

- clear selected character, persona, and session state
- clear cached search results and entity-scoped session lists
- refetch from the newly active backend before the user can take actions against the new selection

### 5. Main Chat Runtime Handoff

Launching from CCP into main chat must be treated as a first-class supported runtime path, not an ad hoc “open existing conversation” shortcut.

For both character-backed and persona-backed sessions:

- CCP selects the entity and optionally the existing session
- CCP resolves the backing mode through the domain service facade
- CCP opens or focuses a main-chat tab with the normalized launched-session contract
- main chat runs the session with the correct backend and identity metadata

Main chat must not assume these launched sessions belong in the ordinary search/history flows.

### 6. History And Discovery Rules

The discovery rules must be explicit and testable.

#### General Main Chat History

Must include only:

- ordinary non-character, non-persona conversations

Must exclude:

- character-backed sessions
- persona-backed sessions launched under CCP ownership

#### CCP Character Session Discovery

Must include only sessions for the currently selected character.

#### CCP Persona Session Discovery

Must include only sessions for the currently selected persona profile.

These rules must be implemented with explicit metadata contracts, not client-side heuristics based only on `character_id` presence.

History inclusion and exclusion must live in DB/service queries and launch/discovery facades, not only checkbox-driven client-side filtering.

The implementation must backfill or safely default existing local conversation rows so legacy character/persona conversations receive explicit `runtime_backend`, `discovery_owner`, and `discovery_entity_id` values before the new history filters go live.

### 7. Local Storage Requirements

The local DB already supports:

- character cards
- conversation assistant metadata
- persona-memory metadata on conversations

But this vertical requires additional local modeling for full parity:

- local persona profile storage
- local character exemplar storage
- local persona exemplar storage
- local metadata needed for discovery ownership and runtime backend
- migration/backfill support for legacy conversation rows and restored tab payloads that predate the new runtime/discovery contract

The implementation plan should treat these as foundational schema/service tasks and explicit prerequisites for CCP UI expansion, not late UI polish.

### 8. Greeting And Preset Compatibility

Server endpoints already exist for:

- greetings list/select
- prompt preset list/create/update/delete

This vertical should integrate enough of those seams that:

- a launched server-backed session can use greeting selection correctly
- a launched server-backed session can use preset-aware execution correctly
- local mode preserves comparable data where needed

But the UI should stay minimal:

- no broad preset-management product area
- no large greeting-management workflow
- only the controls strictly needed for correct execution and debugging

## Compatibility Targets

### Character Card Alignment

Local and server character card handling should converge on preserving:

- identity and display fields
- personality/scenario/system-prompt fields
- greetings and alternate greetings
- tags and extensions
- import/export compatibility with current card formats

### Persona Profile Alignment

This vertical must define a real local/server persona profile shape instead of treating persona support as only a conversation flag.

The minimum target shape is:

- `id`
- `name`
- profile body or instruction fields needed by the server contract
- `created_at`
- `last_modified`
- `version`
- local/server mode-aware lifecycle support

### Character Exemplars

Character exemplars must align with the server-owned character exemplar CRUD/search/debug model.

The local shape should preserve:

- stable exemplar ID
- character ownership
- content
- labels or tags
- enabled/disabled state where applicable
- selection/debug compatibility

### Persona Exemplars

Persona-profile exemplars must align with the persona endpoint model rather than the character-owned exemplar model.

The local shape should preserve:

- stable exemplar ID
- persona-profile ownership
- kind
- content
- source metadata
- notes/review state
- enabled/disabled/deleted state where applicable

## Risks And Guardrails

### Risk: Vertical turns into a CCP rewrite

Guardrail:

- keep the visible layout additive
- change the entity model and service layer first
- change only the UI structures needed to make persona profiles first-class

### Risk: Handoff path breaks because CCP and chat disagree on ID shape

Guardrail:

- normalize CCP conversation/session IDs to string-first contracts before any user-visible launch work

### Risk: Canonical identity drifts back to legacy name-shaped assistant IDs

Guardrail:

- require `assistant_id` to hold the canonical string entity ID
- never populate `assistant_id` from character or persona display names
- keep display-name derivation separate from identity storage

### Risk: History leakage

Guardrail:

- require explicit `discovery_owner` and `discovery_entity_id`
- do not rely on loose `character_id` filtering

### Risk: Server-backed sessions get saved locally by mistake

Guardrail:

- require explicit `runtime_backend`
- route persistence through backend-aware services only

### Risk: Restored tabs and mode switches behave inconsistently

Guardrail:

- persist runtime/discovery metadata in tab state and restore it verbatim
- keep `runtime_backend` session-bound instead of recalculating it from the current toggle on restore
- clear CCP selections, searches, and cached lists on mode change before refetch

### Risk: Persona support becomes fake parity

Guardrail:

- include full persona profile CRUD now
- include local persona profile and persona exemplar storage now
- do not reduce persona support to session flags only

### Risk: Exemplar systems get flattened incorrectly

Guardrail:

- keep character exemplar and persona exemplar contracts separate at every layer

### Risk: Relaunches duplicate tabs or deleted owners produce confusing session drift

Guardrail:

- focus existing tabs by `(runtime_backend, conversation_id)` when possible
- surface an explicit unavailable/relaunch-required state when the backing owner or session can no longer be resolved
- do not silently retarget an open tab to a different entity or backend

## Recommended Implementation Sequence

The implementation plan for this vertical should follow this order:

1. Define canonical string identity rules and legacy bridge behavior
2. Add and migrate local data contracts for persona profiles, exemplars, and runtime/discovery metadata
3. Backfill legacy conversations and restored tab payloads to the new runtime/discovery defaults
4. Add `tldw_api` server client coverage for characters, persona profiles, sessions, messages, exemplars, greetings, and presets
5. Add local/server domain services plus the mode-aware facade
6. Add main-chat launched-session runtime metadata, persistence, restore semantics, and duplicate-tab rules
7. Normalize CCP identity/state, mode-switch invalidation, and dual-entity management areas
8. Add session discovery and launch flows
9. Add greeting/preset execution integration
10. Add focused docs and verification sweep

## Success Criteria

This vertical is successful when:

- CCP can manage both characters and persona profiles as first-class entities
- both character exemplars and persona exemplars have real local and server-backed CRUD paths
- CCP can discover only the sessions for the currently selected character or persona
- main chat can run those launched sessions with explicit local/server runtime behavior
- launched sessions persist and restore with stable canonical IDs plus explicit backend/ownership metadata
- those launched sessions do not leak into general chat history
- switching CCP modes does not reuse stale selections or mix local/server discovery state
- ordinary non-character, non-persona chat behavior remains intact
