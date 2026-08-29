# TASK-19873 Dead CCP Handler Retirement Design

## Summary

Delete two CCP handlers that production code never constructs and remove five
known-broken operations from the deprecated, unrouted Tools Settings UI. Keep
the remainder of `Tools_Settings_Window.py` because current tests and utility
consumers still import it. This closes the dead-code decision without reviving
features that have no production entry point or expanding the task into a
legacy-settings migration.

## Evidence and Decision

Repository-wide reference searches on current `origin/dev` show that
`CCPConversationHandler` and `CCPDictionaryHandler` are defined and re-exported
but never constructed by production code. `PersonasScreen`, the live CCP
destination, constructs only `CCPCharacterHandler` and `CCPPersonaHandler`.
The conversation handler is constructed only by a unit test, while the
dictionary handler has no production construction path.

The handlers' original worker dispatches could not run. TASK-19563 later
repaired their dispatch spelling with bound callables and undecorated plain
worker targets, but did not add a production constructor or route. The deletion
decision rests on that continuing absence of production use, not on claiming
the handlers are still mechanically broken at removal time.

The legacy `ToolsSettingsScreen` is not registered as a destination:
`tools_settings` resolves to `MCPScreen`. Its wrapped
`ToolsSettingsWindow` nevertheless remains imported by focused tests and
supplies tested backup, export, configuration, and database-path utilities.
Therefore the whole module is not eligible for deletion in this task.

The owner decision is:

- Delete `CCPConversationHandler` and `CCPDictionaryHandler` rather than wire
  them into the live Personas destination.
- Delete the five broken legacy operation families rather than repair them:
  single-database vacuum, backup, restore, and integrity-check operations, plus
  legacy Chatbook import.
- Retain the rest of `ToolsSettingsWindow` until its remaining utility and test
  consumers are migrated in a separately scoped task.

## Scope

### CCP modules

- Delete `ccp_conversation_handler.py` and `ccp_dictionary_handler.py`.
- Remove both classes from `UI.CCP_Modules` imports and `__all__`.
- Remove tests that instantiate or describe the deleted handlers.
- Narrow `test_legacy_entrypoints_retired.py` to the live handler inventory and
  make the two deleted module paths positive retirement guards.
- Remove the deleted dictionary handler from the file-picker source
  parameterization.
- Keep live character/persona handlers, shared CCP messages, and standalone
  dictionary-library helpers that still serve production code.

### Deprecated Tools Settings operations

- Remove the individual-database vacuum, backup, restore, and integrity-check
  buttons from the deprecated database-tools view.
- Remove their event-dispatch branches and the corresponding picker, wrapper,
  worker, and operation-specific helpers that become unreferenced.
- Delete or rewrite tests and private-SQLite inventory entries whose only
  contract is one of those retired operation families. A focused test by itself
  is not an independent consumer and does not justify retaining dead workers.
- Remove the legacy Chatbook-import button, dispatch branch, picker, worker,
  and helpers used only by that path.
- Keep bulk database maintenance and the canonical Chatbooks destination's
  import workflow unchanged.
- Keep any utility in `Tools_Settings_Window.py` that has a caller outside the
  five retired operation families. Retained behavior must still have focused
  coverage, but coverage of a retired family is updated or removed with it.

### Documentation and task evidence

- Correct stale CCP refactoring documentation so it no longer presents the
  deleted handlers as live architecture.
- Correct the current data-compatibility map so it no longer presents the
  conversation handler as part of the live CCP runtime.
- Regenerate and review the production diagnostic inventory after deleting the
  handler logger statements.
- Re-measure the CCP pre-import budget and regenerate its snapshot with the
  repository script, tightening the snapshot rather than manually removing
  module names.
- Record the repository-search evidence and deletion boundary in TASK-19873's
  implementation notes. The notes must distinguish the historically impossible
  dispatches from TASK-19563's later dead-code repair, and record that no
  production construction path existed through deletion. This preserves why
  the code was deleted without leaving inert comments or compatibility exports
  in production code.

## Behavioral Contract

No reachable application behavior changes. The live Personas destination
continues to use the character and persona handlers. The canonical Chatbooks
screen continues to own Chatbook import. Reachable bulk database maintenance
continues to work through its existing implementations.

The deprecated Tools Settings window, when mounted directly by tests or
external code, no longer advertises controls whose dispatch was guaranteed to
fail. It retains only operations supported by an executable implementation.

## Verification

Targeted regression coverage will prove:

- the deleted handler modules and package exports are absent;
- live CCP handler tests still pass;
- the deprecated database-tools view no longer composes the removed individual
  maintenance or legacy Chatbook-import controls;
- its button dispatcher no longer references the removed operations;
- retained Tools Settings utilities and reachable bulk operations still pass
  their focused tests;
- the private-SQLite inventory contains only retained SQL seams;
- the production diagnostic inventory is regenerated and passes its repository
  consistency check;
- `scripts/update_boot_budget_snapshots.py` re-measures and tightens the CCP
  pre-import snapshot after the two modules leave the eager import census;
- Ruff lint/format and both working-tree and base-branch diff checks pass.

Tests will be adjusted before production deletion so the first run demonstrates
the old dead-code contract is no longer the desired one. No full-suite run is
part of this focused deletion unless explicitly requested.

## Alternatives Rejected

### Repair and wire the handlers and operations

Rejected because it would create new product behavior for code that has never
had a production construction or routing path. There is no acceptance criterion
requiring those features to exist.

### Delete all of `Tools_Settings_Window.py`

Rejected for this task because the module still supplies independently tested
utilities and remains in import-closure contracts. Whole-module retirement
requires migrating or deleting those consumers and is materially broader than
the five known-broken operations.

### Leave compatibility exports or failure stubs

Rejected because they would preserve the false impression that the handlers
and operations remain supported. Their absence is the intended contract.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is dead-code removal that enforces existing navigation and
ownership decisions. It introduces no new storage, service, security, runtime,
or cross-module boundary.
