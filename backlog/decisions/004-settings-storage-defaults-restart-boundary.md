# ADR 004: Settings Storage Defaults Restart Boundary

Status: Accepted
Date: 2026-06-08
Related Task: [backlog/tasks/task-81 - Functionalize-Settings-Storage-defaults.md](../tasks/task-81%20-%20Functionalize-Settings-Storage-defaults.md)
Supersedes: N/A

## Decision

Settings may edit persisted local storage path defaults under the existing `database` configuration section, but those edits are restart-required and must not move files, create migration jobs, or reconnect active database handles.

Editable Storage defaults are limited to:

- `database.USER_DB_BASE_DIR`
- `database.chachanotes_db_path`
- `database.prompts_db_path`
- `database.media_db_path`
- `database.research_db_path`
- `database.writing_db_path`
- `database.library_collections_db_path`
- `database.workspaces_db_path`

Settings can validate the draft paths for basic safety and parent-directory readiness. It can save valid persisted defaults to config and update `app_config` for display consistency. Runtime storage services remain the owner of active database connections until the next application launch.

## Context

Settings currently exposes Storage as a validation-only panel. Users can check local paths, but they cannot configure database path defaults from the main configuration hub. This leaves Settings incomplete for a core configuration domain.

Storage paths are high-risk because live file movement or handle reconnection can corrupt data, invalidate open SQLite connections, or create confusing partial migrations. The existing configuration already has a stable `[database]` section and load path, so Settings can safely provide persisted defaults without solving migration in the same PR.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep Storage validation-only | This preserves safety but leaves Settings unable to configure a core app domain. |
| Move database files immediately on save | This needs backups, locks, rollback, cross-service coordination, and explicit user confirmation. It is too risky for a Settings defaults slice. |
| Reconnect active database handles after save | Active services have independent lifecycles. Silent reconnection from Settings would create hidden runtime side effects and likely break ongoing workflows. |
| Add a new top-level storage config section | The existing `database` section is already the source of truth for these defaults. A new section would duplicate state. |

## Consequences

The Storage Settings UI must label changes as next-launch defaults. Save success copy must not imply data was moved or active services changed. The Storage check can evaluate draft values and parent-directory readiness, but cannot create directories or mutate files during validation.

Future live migration or storage relocation can be implemented as a dedicated guided flow with backups, file locks, rollback, and screenshot-approved UX. That future flow should reference or supersede this ADR instead of quietly changing Settings save semantics.

## Links

- [Implementation plan](../../Docs/superpowers/plans/2026-06-08-settings-storage-defaults.md)
- [Backlog task TASK-81](../tasks/task-81%20-%20Functionalize-Settings-Storage-defaults.md)
- [Settings configuration hub design](../../Docs/superpowers/specs/2026-05-24-settings-configuration-hub-design.md)
