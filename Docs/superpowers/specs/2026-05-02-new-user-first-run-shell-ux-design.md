# New-User First-Run Shell UX Design

Date: 2026-05-02
Status: Approved for spec review
Scope: First-run orientation, top-level information architecture, navigation clarity, capability-aware onboarding, and core screen framing
Primary Persona: New user
Primary Interaction Bias: Keyboard-first, mouse-safe, compact TUI
Platform Constraint: Stay within the current Textual/TUI product model

## Summary

`tldw_chatbook` already has substantial capability, but it does not currently present a clear front door for new users. The main UX problem is not lack of features or styling quality. It is orientation debt.

The current product appears to expose too much of its internal structure too early, relies on several labels that assume prior context, and asks new users to infer which path matters before they have completed a meaningful action. This creates hesitation, failed setup attempts, and avoidable first-run drop-off.

The recommended direction is a `state-aware onboarding shell` built around:

- a skippable `Home` screen as the default first-run landing surface
- a simplified top-level information architecture that distinguishes core workflows from expert tooling
- capability-aware guidance so the app recommends only actions the current installation can complete
- a consistent page frame and empty-state model across primary screens

This is not a cosmetic refresh. It is a product-structure and workflow-priority redesign for new-user comprehension.

## Problem Statement

The current shell and screen system appears to create first-run confusion in these ways:

1. There is no clear beginner entry point that explains what the product is and what to do first.
2. The app presents too many areas at equal visual importance, including advanced and diagnostic workflows.
3. Several user-facing labels reflect implementation history or expert shorthand more than beginner mental models.
4. Setup prerequisites for providers and optional capabilities are not surfaced early enough in the journey.
5. Empty states are likely too passive in a product where value depends on setup, import, or prior content.
6. The shell currently risks competing navigation patterns or duplicated chrome in transitional areas of the codebase.

For a new user, this does not read as “powerful.” It reads as “large, dense, and unclear.”

## Goals

- Help a new user understand what the product does within the first minute.
- Help a new user choose the right next step without needing to inspect Settings first.
- Preserve product breadth while reducing first-run cognitive load.
- Separate core workflows from expert or low-frequency tooling.
- Make setup state and capability gaps visible before they cause dead-end interactions.
- Keep `Characters/Personas` as a top-level destination because they are a first-class product concept.

## Non-Goals

- Rebuild every screen in this slice.
- Remove advanced features from the product.
- Force onboarding completion before users can use the app.
- Convert the app into a web-style dashboard with large card-heavy layouts.
- Change stable backend capabilities or feature ownership.

## Design Principles

- One clear front door
- Recognition before recall
- Core workflows first, expert workflows second
- Status before failure
- Compact, keyboard-first orientation
- Progressive disclosure over full upfront exposure
- Stable navigation labels
- One primary navigation system per screen

## Recommended Approach

### Option A: Navigation cleanup only

Improve labels, grouping, and empty states but keep the current landing behavior.

Why not recommended:

- does not solve the missing front door problem
- still asks users to infer the right next step from the shell alone

### Option B: Guided home plus simplified IA

Add a compact, skippable, state-aware `Home` screen and reduce first-run prominence of advanced workflows.

Why recommended:

- addresses the biggest orientation problem directly
- creates a coherent beginner model without removing expert depth
- fits TUI constraints if kept compact and task-led

### Option C: Full task-based app re-architecture

Reorganize the entire app around end-user goals in one larger rewrite.

Why not recommended now:

- higher implementation risk
- mixes shell repair with deeper architectural migration
- unnecessary before validating a clearer first-run shell

**Chosen approach:** Option B, guided home plus simplified IA.

## Top-Level Information Architecture

The top-level shell should prioritize user goals over implementation history.

### Recommended Top-Level Destinations

- `Home`
- `Chat`
- `Content`
- `Import`
- `Search`
- `Characters`
- `Models`
- `Advanced`
- `Settings`

### Destination Roles

- `Home`: orientation, state-aware next steps, setup status, and recent work
- `Chat`: direct interaction with LLMs and conversation runtime
- `Content`: saved material such as notes, media, conversations, and chatbooks
- `Import`: ingestion entry point for files, URLs, media, and source material
- `Search`: retrieval across imported and saved material
- `Characters`: personas, reusable behavior, prompts, and related identity shaping workflows
- `Models`: provider setup, model availability, and “AI is ready/not ready” status
- `Advanced`: evals, coding, STTS, subscriptions, logs, stats, and other expert tooling
- `Settings`: application-level preferences and configuration not required for a first successful task

### `Models` Versus `Settings` Boundary

This boundary must be explicit in both labels and behavior.

- `Models` owns anything required to make AI functionality work, including provider setup, model availability, and “ready/not ready” runtime status.
- `Settings` owns application preferences, customization, and lower-priority configuration that should not block a first useful task.

New users should not have to guess whether “make chat work” lives under `Models` or `Settings`. It should live under `Models`.

### Why `Content` Instead Of `Library`

`Library` risks sounding too abstract given the current object mix. `Content` is plainer, more direct, and easier to connect to notes, media, conversations, and imported material.

### Why `Characters` Stays Top-Level

`Characters/Personas` are a core part of product identity rather than a hidden secondary feature. They should remain directly accessible at the shell level, but their page framing should explain the concept clearly to beginners.

## Route Compatibility And Migration Map

This design introduces a simpler top-level model, but implementation must remain compatible with the current screen inventory.

### New-To-Current Mapping

| New destination | Current routes/screens it wraps, renames, or groups |
|---|---|
| `Home` | new screen |
| `Chat` | `chat` |
| `Content` | grouped access to `notes`, `media`, `chatbooks`, and conversation-oriented browsing currently surfaced through `ccp` |
| `Import` | `ingest` |
| `Search` | `search` |
| `Characters` | character/persona/prompt workflows currently surfaced through `ccp` |
| `Models` | `llm` plus any provider/model setup currently buried in settings flows |
| `Advanced` | `coding`, `stts`, `evals`, `subscriptions`, `logs`, `stats` |
| `Settings` | `tools_settings`, `customize` |

### Migration Rules

- Current route IDs may remain stable internally during the first implementation slice.
- The shell may relabel, regroup, or wrap existing screens without requiring an immediate backend or route rewrite.
- `ccp` is not a usable new-user label and must not remain a primary shell label.
- If `Content` or `Characters` temporarily launch into existing composite screens, the entry state must land the user in the relevant subsection rather than a generic legacy overview.
- The implementation plan must explicitly separate `presentation relabeling` from `deeper screen decomposition`.

## First-Run Entry Rules

### Default Landing Rule

New users should land on `Home`, not `Chat`.

### Exit Rule

`Home` must always be skippable.

### Persistence Rule

`Home` should not permanently own app entry once the user has completed a meaningful action. After the user completes a first successful task, the default reopen behavior should prefer:

1. last active screen, or
2. explicit user preference

`Home` remains available as an always-accessible dashboard and orientation layer.

### Upgrade Behavior Rule

Existing users must not be treated as first-run users by default.

A user should bypass default-first-run `Home` behavior if any strong evidence of prior meaningful use exists, such as:

- existing saved conversations
- existing notes or media
- existing characters/personas
- existing chatbooks
- prior successful provider/model configuration
- stored last-active-screen state

For those users, default reopen behavior should remain:

1. last active screen when available
2. prior default destination behavior if no last-active-screen state exists

`Home` may still be introduced to existing users as:

- a one-time dismissible “new home available” entry point, or
- a visible primary destination in navigation

but it should not forcibly interrupt established workflows.

### Meaningful Action Definition

Any of the following should count as first-use completion:

- successfully starting a chat
- successfully configuring a working model/provider
- successfully importing content
- successfully opening and using an existing content area with saved data

## Home Screen Design

`Home` should be treated as a compact decision surface, not a decorative dashboard.

### Purpose

The Home screen should answer four questions immediately:

1. What is this app for?
2. What is ready right now?
3. What should I do next?
4. Where do I go for the most common workflows?

### Required Home Sections

#### 1. Product Summary

One short sentence in plain language explaining the product.

Example direction:

“Use AI chat, saved content, search, and personas in one terminal workspace.”

This must stay brief. It is orientation copy, not marketing copy.

#### 2. Primary Recommended Action

One dominant CTA based on current app state.

Examples:

- `Set up a model`
- `Start your first chat`
- `Import content`
- `Search your knowledge`

There must be only one visually dominant primary action at a time.

The implementation must derive this action from a deterministic priority order rather than ad hoc conditions.

#### 3. Guided Paths

Show 3 to 4 compact paths that explain the main things the product can do.

Recommended paths:

- `Talk to a model`
- `Import files or media`
- `Search saved content`
- `Use characters/personas`

Each path should include a one-line explanation and a direct jump.

#### 4. Setup Status

Show compact readiness indicators for the parts of the app most likely to block success.

Examples:

- Models: ready / not configured
- Import capabilities: basic / enhanced / unavailable
- Search: ready / limited until content exists
- Personas: available / no saved characters yet

#### 5. Recent Work

If the user has meaningful existing activity, Home should shift from onboarding to re-entry.

Show small recent items such as:

- recent conversations
- recent notes
- recent media/import activity

This section should stay compact and should not displace the primary recommended action.

## Home Behavior Rules

- Home must be useful in both empty and partially configured states.
- Home must never recommend an action that the current installation cannot complete.
- Home should adapt as soon as setup state changes.
- Home should remain compact enough to fit comfortably in a normal terminal viewport.
- Home should not require scrolling on standard terminal sizes unless recent work becomes unusually long.
- Home should support both keyboard navigation and mouse click selection.

## Capability-Aware Guidance

The onboarding layer must be capability-aware.

### Capability Inputs

The shell should derive recommendations from:

- configured provider/model availability
- optional dependency availability
- presence or absence of imported content
- presence or absence of saved characters/personas
- presence or absence of prior user work

### Recommendation Rules

- If no model is configured, `Set up a model` should win over all other CTAs.
- If a model is ready but no content exists, recommend either `Start your first chat` or `Import content`.
- If content exists but no search history or search setup is evident, recommend `Search your knowledge`.
- If characters exist, Home may surface them as a guided path but should not make them the first CTA unless they are the user’s main recent workflow.

`Search` should remain visible in global navigation even on first run, but it should not be promoted as the primary recommended action until searchable content exists.

### CTA Decision Matrix

The `Home` primary CTA should be chosen by this priority order:

1. `Set up a model`
Condition:
No working provider/model is available for chat.

2. `Start your first chat`
Condition:
Model is ready and the user has no successful chat history.

3. `Import content`
Condition:
Model is ready, chat is possible, but no meaningful saved/imported content exists.

4. `Search your knowledge`
Condition:
Searchable content exists and search can provide useful results.

5. `Resume recent work`
Condition:
The user has clear recent activity and no more urgent setup blockers exist.

Tie-break rules:

- unresolved setup blockers outrank content-based suggestions
- first successful core action outranks exploratory features
- recent-work resume outranks generic discovery only after the user is clearly beyond first run
- characters/personas may be surfaced as a guided path without taking over the primary CTA unless recent workflow evidence strongly supports it

### Failure Prevention

If a screen depends on unavailable capability, the user should see a clear readiness message before attempting the task, not after a failed action.

## Primary Navigation Rules

- There must be one clear primary navigation system on each screen.
- Core destinations should appear before advanced destinations.
- Advanced destinations should be contained under `Advanced`, not displayed as equal-weight first-run peers.
- Active location must always be visually obvious.
- Abbreviations and implementation shorthand should not be primary user-facing labels.

## Screen Contract For Core Destinations

Every primary destination should follow one compact page grammar.

### Required Page Frame

Each major screen should contain:

1. page title
2. one-line purpose statement
3. one primary action
4. optional status/help line
5. main work area

This frame should reduce relearning cost across the app.

### Empty-State Rules

Empty states should not simply report absence. They should route users forward.

#### Chat

- explain how to choose or confirm a model
- provide a clear action to start chatting
- keep advanced settings collapsed by default for new users

#### Content

- explain what kinds of items live there
- provide links to import content or open recent items

### `Content` Internal Structure

`Content` must not be implemented as a vague bucket label.

In the first implementation slice, `Content` should behave as a container screen with a local section switcher for:

- `Notes`
- `Media`
- `Conversations`
- `Chatbooks`

Rules:

- one subsection should be selected by default based on recent activity, or `Conversations` if no better signal exists
- each subsection may initially wrap an existing screen or legacy subsection
- the user must be able to tell which content type they are browsing at all times
- if deeper screen decomposition is deferred, the shell still needs to provide a coherent `Content` landing frame above the wrapped legacy content

#### Import

- explain supported source types and expected output
- show what happens after import completes

#### Search

- explain what is searchable
- explain why results may be empty on first run
- point to import when no content exists
- avoid presenting empty search as a broken feature

#### Characters

- explain what a character/persona does in this product
- provide a direct action such as create, import, or start chat with persona

#### Models

- act as the guided place to make AI functionality work
- clearly distinguish ready, missing, and optional setup steps

## Advanced Containment

`Advanced` should not become a junk drawer. It should contain destinations that meet at least one of these tests:

- expert-oriented
- low-frequency
- diagnostic
- configuration-heavy
- not required for first-run success

Likely candidates:

- evals
- coding
- STTS
- subscriptions
- logs
- stats

`Advanced` should still be easy to reach, but it should not dominate the first-run shell.

### Advanced Discoverability Rules

`Advanced` features must remain discoverable when contextually relevant.

Rules:

- Core screens may link into specific advanced tools when the user’s current task logically leads there.
- These links should be contextual and explicit, for example:
  - Chat or Models linking to `Evals`
  - Chat linking to `Coding`
  - Models or Chat linking to `STTS`
  - Content or Search linking to `Subscriptions` when monitoring sources is relevant
- Contextual entry points should use plain-language helper copy and should not require users to guess that the feature lives under `Advanced`.
- `Advanced` should act as containment, not concealment.

## Risks And Mitigations

### Risk: Home becomes decorative

Mitigation:

- enforce one dominant CTA
- limit guided paths
- keep summary copy short
- treat Home as a decision surface, not a showcase

### Risk: `Advanced` becomes a catch-all mess

Mitigation:

- define qualification rules for advanced destinations
- keep naming stable
- audit items periodically against first-run value

### Risk: `Content` remains too broad

Mitigation:

- use clear sub-areas inside Content
- surface item types explicitly in screen copy and filters

### Risk: guidance becomes inaccurate when capabilities vary

Mitigation:

- derive recommendations from real runtime availability and saved state
- never hardcode onboarding steps that assume optional dependencies

### Risk: onboarding irritates returning users

Mitigation:

- make Home skippable
- switch to last-used-screen behavior after first successful task
- allow user preference for reopen behavior later

## Success Metrics

The redesign should be considered successful if it improves these outcomes:

- time to first successful chat
- time to first successful import
- percent of users who reach a useful action without visiting Settings first
- blank-search rate on first run
- number of navigation hops before the first successful task
- qualitative ability of a new user to explain the product within the first minute

## Measurement Plan

The implementation plan should include a lightweight measurement pass tied to the success metrics.

### Baseline Requirements

Before shipping the redesigned shell, capture the current baseline where feasible for:

- startup destination behavior
- time to first successful chat
- time to first successful import
- empty search frequency
- Settings-first navigation frequency

### Suggested Event Sources

- app launch and resolved landing destination
- model/provider setup success
- first successful chat send/response completion
- first successful import completion
- search submitted with zero available searchable content
- navigation sequence before first successful task

### Measurement Constraints

- If full telemetry is not available, use local metrics, structured logs, or deterministic event counters.
- Measurement should be good enough to compare before/after behavior for the redesign slice.
- Instrumentation work should stay minimal and should not block shell implementation unless no outcome verification is otherwise possible.

## Implementation Phases

### Phase 1: Shell foundation

- define final top-level destination model
- remove competing navigation chrome
- standardize top-level labels

### Phase 2: Home screen

- add state-aware Home
- implement setup status and CTA logic
- wire first-run entry rules

### Phase 3: Core screen framing

- apply the page frame pattern to Chat, Content, Import, Search, Characters, and Models
- improve empty states and primary actions

### Phase 4: Advanced containment

- move expert workflows behind `Advanced`
- verify discoverability without first-run overload

### Phase 5: Optimization

- tune reopen behavior
- refine recommendation logic
- measure first-run success metrics

## Acceptance Tests For The Design

- A brand-new user can identify a sensible first action from Home without opening Settings.
- A user with no model configured is guided toward model setup before hitting a failed chat or search path.
- A user with configured models but no content sees chat and import as the primary options.
- A user with existing content but no recent usage can re-enter through recent work or search guidance.
- `Characters/Personas` remain visible as a top-level destination and are explained clearly for new users.
- Advanced workflows are still available, but they are no longer presented as equal first-run priorities.

## Final Recommendation

Proceed with a shell-first UX slice centered on:

- `Home` as a compact, state-aware first-run landing surface
- a simplified top-level IA that preserves `Characters` as first-class
- capability-aware recommendations
- a shared screen contract for core workflows
- reduced first-run prominence for expert tooling

This gives `tldw_chatbook` a coherent front door without flattening the depth that makes the product valuable.
