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
- Remove the operation-specific `Last Backup` labels and update the view copy;
  without single-database backup, those labels have no writer and would remain
  stuck at `Loading...`.
- Remove their event-dispatch branches and the corresponding picker, wrapper,
  worker, and operation-specific helpers that become unreferenced.
- Remove `_validate_maintenance_path`, `_get_schema_version`, and
  `_update_last_backup_status` once their only callers are deleted, along with
  imports used exclusively by the retired paths. Keep `_get_database_path`,
  database size/record reporting, and other helpers used by bulk or advanced
  operations.
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
- Remove the now-orphaned `settings.schema`, `settings.single_backup`,
  `settings.pre_restore_backup`, and `settings.restore` policies from the
  private-SQLite owner registry. Retain `settings.vacuum` and
  `settings.integrity` because the bulk workers still use them, and point their
  inventory evidence at those bulk workers.
- Preserve generic `private_sqlite` backup/restore coverage by switching it to
  equivalent retained owner policies where the test exercises shared storage
  behavior rather than the retired Tools Settings caller. Delete only tests
  whose contract is specifically the removed UI operation or owner ID.

### Documentation and task evidence

- Correct stale CCP refactoring documentation so it no longer presents the
  deleted handlers as live architecture.
- Correct the current data-compatibility map so it no longer presents the
  conversation handler as part of the live CCP runtime.
- Correct the current Database Tools implementation summary and the Chatbook
  importer test commentary so neither advertises the retired controls as live
  call sites.
- Update the private-SQLite owner inventory and pragma-test narrative to remove
  retired owner IDs while retaining the bulk vacuum/integrity owners.
- Regenerate and review the production diagnostic inventory after deleting all
  affected handler and Tools Settings diagnostic statements.
- Re-measure the CCP pre-import budget and regenerate its snapshot with the
  repository script's `--only preimport` option, tightening only that snapshot
  rather than manually removing module names or rewriting unrelated budgets.
- Record the repository-search evidence and deletion boundary in TASK-19873's
  implementation notes. The notes must distinguish the historically impossible
  dispatches from TASK-19563's later dead-code repair, and record that no
  production construction path existed through deletion. This preserves why
  the code was deleted without leaving inert comments or compatibility exports
  in production code.
- Leave historical task files and superseded Superpowers plans/specs unchanged;
  they remain evidence of the code state and decisions at the time they were
  written. Only documents that claim to describe the current architecture are
  corrected.

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
- the private-SQLite registry, inventory, and tests contain only retained owner
  policies, with bulk vacuum/integrity still covered;
- the production diagnostic inventory is regenerated and passes its repository
  consistency check;
- `scripts/update_boot_budget_snapshots.py --only preimport` re-measures and
  tightens only the CCP pre-import snapshot after the two modules leave the
  eager import census, and the corresponding pre-import guard passes;
- Ruff lint/format and both working-tree and base-branch diff checks pass.

Tests will be adjusted before production deletion so the first run demonstrates
the old dead-code contract is no longer the desired one. No full-suite run is
part of this focused deletion unless explicitly requested.

The untouched `origin/dev` baseline currently has one focused failure:
`test_restore_refuses_a_dangerous_backup_path_via_path_validation` assumes the
ChaChaNotes path does not exist after mounting the app, but app startup creates
it. The failure reproduces in isolation and belongs entirely to the retired
single-restore contract. It will be removed with that contract rather than
patched or counted as a branch regression.

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
