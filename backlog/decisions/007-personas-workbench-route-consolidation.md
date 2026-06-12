# ADR-007: Personas Workbench Route Consolidation

Status: Accepted
Date: 2026-06-10
Related Task: [backlog/tasks/task-90 - Personas-workbench-foundation-and-route-consolidation-ADR.md](../tasks/task-90%20-%20Personas-workbench-foundation-and-route-consolidation-ADR.md)
Supersedes: N/A

## Decision

The top-level `personas` route is the durable product destination for characters, persona profiles, prompts, dictionaries, lore, import/export, and Console handoff; `ccp` remains a compatibility route only while its reusable handlers and widgets are incrementally adapted behind destination-native Personas state and messages.

## Context

Personas currently has two user-facing surfaces. The `personas` route is the shell that appears in the unified top navigation, but it only exposes a behavior-context snapshot and can send users to `ccp` for deeper work. The `ccp` route already contains much of the functional character/persona editor behavior, but it carries legacy naming and route ownership that does not match the approved top-level product model.

The product direction is to make Personas a first-class destination that users can understand without knowing CCP. Characters, persona profiles, prompts, dictionaries, lore, card import/export, conversation preview, and Console attach/start-chat actions should live under that destination. This foundation must not force a broad rewrite or break existing working CCP handlers, so the transition needs a shared state/message layer that lets later UI slices move features without coupling new widgets directly to `CCPScreen`.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep `ccp` as the real destination and make `personas` a permanent wrapper | This preserves legacy terminology and makes the top-level Personas tab misleading because meaningful work still happens in another route. |
| Rewrite all CCP handlers and widgets in one PR | Too risky for character/persona CRUD, import/export, prompt, dictionary, and conversation flows; it would make review and rollback difficult. |
| Split Personas and Characters into separate top-level destinations | This conflicts with the current shell IA and separates closely related behavior-shaping concepts that users need to compare and attach to Console together. |
| Move prompts, dictionaries, and lore into Settings | These are reusable behavior assets, not global app preferences; Settings should configure defaults and safety boundaries, not own authoring workflows. |

## Consequences

New Personas work should target reusable `Persona_Widgets` state/messages and the top-level `personas` destination. Existing CCP handlers/widgets may be reused as implementation detail, but new code should avoid adding fresh dependencies from destination-native Personas widgets back into `CCPScreen`.

Visible UI migration should happen in small screenshot-approved slices. Until those slices land, `ccp` can continue serving compatibility paths so existing tests and user workflows do not regress. Later work can map `ccp`, `characters`, `prompts`, and related aliases to the destination-native Personas workbench once feature parity and QA are complete.

## Links

- [CCP destination-native route replacement plan](../../Docs/superpowers/plans/2026-05-20-ccp-destination-native-route-replacement.md)
- [Personas loading recovery task](../tasks/task-60.5%20-%20Fix-Personas-destination-indefinite-behavior-context-loading-state.md)
